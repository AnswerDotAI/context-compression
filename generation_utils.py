import itertools
from typing import Optional, Tuple
from pathlib import Path

import torch
import torch._dynamo.config
import torch._inductor.config

import argparse
from model import Transformer, find_multiple
from tokenizer import TokenizerInterface


default_device = "cuda" if torch.cuda.is_available() else "cpu"


def add_generation_arguments(parser: argparse.ArgumentParser):
    group = parser.add_argument_group("generation_args")
    # Generation hparams
    group.add_argument(
        "--checkpoint_path",
        type=Path,
        default=Path(__file__).resolve().parent
        / "checkpoints/Qwen/Qwen2-1.5B-Instruct/model.pth",
        help="Model checkpoint path.",
    )

    group.add_argument("--profile", type=Path, default=None, help="Profile path.")

    group.add_argument(
        "--compile", action="store_true", help="Whether to compile the model."
    )

    group.add_argument(
        "--device", type=str, default=default_device, help="Device to use"
    )


def compute_max_seq_length(model, prompt_lens, max_new_tokens) -> int:
    max_prompt_length = max(len(prompt_lens[i]) for i in range(len(prompt_lens)))
    max_seq_length = max_prompt_length + max_new_tokens
    if max_seq_length > model.config.block_size:
        print(
            f"Warning: The longest prompt puts the desired max_seq_length at {max_seq_length}, which is greater than models max of {model.config.block_size}."
        )
        print(f"Setting to model's max_seq_length of {model.config.block_size}.")
        max_seq_length = model.config.block_size
    print(f"Maximum context length of {max_seq_length} tokens.")
    return max_prompt_length, max_seq_length


def device_sync(device):
    if "cuda" in device:
        torch.cuda.synchronize(device)
    elif ("cpu" in device) or ("mps" in device):
        pass
    else:
        print(f"device={device} is not yet suppported")


def multinomial_sample_one_no_sync(
    probs_sort,
):  # Does multinomial sampling without a cuda synchronization
    q = torch.empty_like(probs_sort).exponential_(1)
    return torch.argmax(probs_sort / q, dim=-1, keepdim=True).to(dtype=torch.int)


def logits_to_probs(logits, temperature: float = 1.0, top_k: Optional[int] = None):
    logits = logits / max(temperature, 1e-5)

    if top_k is not None:
        v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
        pivot = v.select(-1, -1).unsqueeze(-1)
        logits = torch.where(logits < pivot, -float("Inf"), logits)
    probs = torch.nn.functional.softmax(logits, dim=-1)
    return probs


def sample(
    logits: torch.Tensor,
    next_token: torch.Tensor = None,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
):
    probs = logits_to_probs(logits[0, -1], temperature, top_k)
    if next_token is None:
        idx_next = multinomial_sample_one_no_sync(probs)
    else:
        idx_next = next_token
    return idx_next, probs


def greedy(logits, next_token):
    probs = torch.nn.functional.softmax(logits[0, -1], dim=-1)
    if next_token is None:
        idx_next = torch.argmax(probs, keepdim=True).to(dtype=torch.int)
    else:
        idx_next = next_token
    return idx_next, probs


def prefill(
    model: Transformer,
    x: torch.Tensor,
    input_pos: torch.Tensor,
    next_token: torch.Tensor = None,
    **sampling_kwargs,
) -> torch.Tensor:
    # input_pos: [B, S]
    causal_mask = (
        torch.tril(torch.ones(len(input_pos), len(input_pos), dtype=torch.bool))
        .unsqueeze(0)
        .unsqueeze(0)
        .to(x.device)
    )
    logits = model(x, input_pos, mask=causal_mask)
    return greedy(logits, next_token)


def decode_one_token(
    model: Transformer,
    x: torch.Tensor,
    input_pos: torch.Tensor,
    next_token: torch.Tensor = None,
    **sampling_kwargs,
) -> Tuple[torch.Tensor, torch.Tensor]:
    # input_pos: [B, 1]
    assert input_pos.shape[-1] == 1
    logits = model(x, input_pos)
    return greedy(logits, next_token=next_token)


def decode_n_tokens(
    model: Transformer,
    cur_token: torch.Tensor,
    input_pos: torch.Tensor,
    num_new_tokens: int,
    terminator_ids: Optional[list] = None,
    prefix: Optional[torch.Tensor] = None,
    **sampling_kwargs,
):
    new_tokens, new_probs = [], []
    for i in range(num_new_tokens):
        with torch.backends.cuda.sdp_kernel(
            enable_flash=False, enable_mem_efficient=False, enable_math=True
        ):  # Actually better for Inductor to codegen attention here
            teacher_force = prefix is not None and i < len(prefix)
            next_token = prefix[i].view(1) if teacher_force else None
            next_token, next_prob = decode_one_token(
                model, cur_token, input_pos, next_token=next_token, **sampling_kwargs
            )

            new_tokens.append(next_token.clone())
            new_probs.append(next_prob.clone())

            if terminator_ids and next_token in terminator_ids and not teacher_force:
                break

            input_pos += 1
            cur_token = next_token.view(1, -1)

    return new_tokens, new_probs


def model_forward(model, x, input_pos):
    return model(x, input_pos)


def normalize_cache_length(
    max_cache_length: float, max_seq_length: int, multiple_of: int = 8
) -> int:
    """
    Computes the absolute cache length given the max_cache_length and max_seq_length.
    """
    if 0 < max_cache_length <= 1:
        max_cache_length = round(max_seq_length * max_cache_length)
    else:
        assert int(max_cache_length) == max_cache_length
        max_cache_length = int(max_cache_length)
        if max_cache_length > max_seq_length:
            print(
                f"Warning: max_cache_length ({max_cache_length}) is greater than max_seq_length ({max_seq_length}). Setting to {max_seq_length}"
            )
            max_cache_length = max_seq_length
    return min(find_multiple(max_cache_length, multiple_of), max_seq_length)


def setup_caches(
    model: Transformer,
    tokenizer: TokenizerInterface,
    device: torch.device,
    max_seq_length: int,
    cache_kwargs: dict = None,
):
    # Normalize max_cache_length to absolute cache length if provided as a fraction of the max seq sequence length
    cache_kwargs["max_cache_length"] = list(
        map(
            lambda l: normalize_cache_length(l, max_seq_length),
            cache_kwargs["max_cache_length"],
        )
    )
    assert (
        model.config.n_layer % len(cache_kwargs["max_cache_length"]) == 0
    ), f'max_cache_length ({len(cache_kwargs["max_cache_length"])}) must be a factor of {model.config.n_layer} layers.'

    tile_size = model.config.n_layer // len(cache_kwargs["max_cache_length"])
    cache_kwargs["max_cache_length"] = [
        item for item in cache_kwargs["max_cache_length"] for _ in range(tile_size)
    ]

    # Gets called twice when model is wrapped in torch.compile which causes an error without the if statement
    if type(cache_kwargs["drop_amount"]) != list:
        cache_kwargs["drop_amount"] = [
            max(int(cache_kwargs["drop_amount"] * l), 1)
            for l in cache_kwargs["max_cache_length"]
        ]

    assert cache_kwargs["global_tokens"] <= min(
        cache_kwargs["max_cache_length"]
    ), "Global tokens must be less than max_cache_length."

    if cache_kwargs["cache_strategy"] == "fastgen":
        # We need to pass the special and punctuation token ids to the cache via cache_kwargs
        cache_kwargs["token_ids"] = {
            "special": tokenizer.special_ids(),
            "punctuation": tokenizer.punctuation_ids(),
        }

    with torch.device(device):
        model.setup_caches(max_batch_size=1, **cache_kwargs)


def reset_caches(model: Transformer):
    model.reset_caches()


def get_cache_stats(model: Transformer, prompt_len: int, gen_len: int):
    return model.get_cache_stats(prompt_len, gen_len)


@torch.no_grad()
def generate(
    model: Transformer,
    prompt: torch.Tensor,
    max_new_tokens: int,
    terminator_ids: Optional[list] = None,
    feed_long_prompts: bool = False,
    **sampling_kwargs,
) -> torch.Tensor:
    """
    Takes a conditioning sequence (prompt) as input and continues to generate as many tokens as requested.
    """

    # create an empty tensor of the expected final shape and fill in the current tokens
    prompt_length = prompt.size(0)

    device, dtype = prompt.device, prompt.dtype

    min_cache_length = model.min_cache_length()
    # Subtract 1 in case we need one generation step over which to compute attention, etc.
    max_prompt_len = min_cache_length - 1
    prefix = None
    # If we asked to have prompt truncated and fed, we need to do split prompt into prompt and prefix
    # We also define a rare yet important edge case: if |prompt| is exactly cache length
    # We might have to start evictions before having had a change to record any state (attentions).
    # In this scenario let's decrement prompt by 1 and start "generating" on the prefix
    if (
        feed_long_prompts and prompt_length > max_prompt_len
    ) or prompt_length == min_cache_length:
        prompt, prefix = prompt[:max_prompt_len], prompt[max_prompt_len:]
        max_new_tokens += len(prefix)
        prompt_length = max_prompt_len
    # create an empty tensor (all -1) of the expected final shape and fill in the current tokens
    # GPT-Fast had this as empty but the values of empty are non-deterministic
    seq = torch.full((prompt_length + max_new_tokens,), -1, dtype=dtype, device=device)
    seq[:prompt_length] = prompt
    input_pos = torch.arange(0, prompt_length, device=device)

    ret = prefill(
        model,
        prompt.view(1, -1),
        input_pos,
        next_token=None if prefix is None else prefix[0].view(1),
        **sampling_kwargs,
    )
    next_token = ret[0].clone()
    next_tok_probs = ret[1].clone()
    seq[prompt_length] = next_token

    input_pos = torch.tensor([prompt_length], device=device, dtype=torch.int)
    generated_tokens, generated_tok_probs = decode_n_tokens(
        model,
        next_token.view(1, -1),
        input_pos,
        max_new_tokens - 1,
        terminator_ids=terminator_ids,
        prefix=None if prefix is None else prefix[1:],
        **sampling_kwargs,
    )
    if len(generated_tokens) > 0:
        seq[prompt_length + 1 : prompt_length + 1 + len(generated_tokens)] = torch.cat(
            generated_tokens
        )

    # Truncate seq to first instance of -1 if -1 is present
    if -1 in seq:
        seq = seq[: torch.where(seq == -1)[0][0]]

    return seq, [next_tok_probs] + generated_tok_probs


def load_model(checkpoint_path, device, precision, use_tp):
    use_cuda = "cuda" in device
    with torch.device("meta"):
        model = Transformer.from_name(checkpoint_path.parent.name)

    if "int8" in str(checkpoint_path):
        print("Using int8 weight-only quantization!")
        from quantize import WeightOnlyInt8QuantHandler

        simple_quantizer = WeightOnlyInt8QuantHandler(model)
        model = simple_quantizer.convert_for_runtime()

    if "int4" in str(checkpoint_path):
        print("Using int4 weight-only quantization!")
        path_comps = checkpoint_path.name.split(".")
        groupsize = int(path_comps[-2][1:])
        from quantize import WeightOnlyInt4QuantHandler

        simple_quantizer = WeightOnlyInt4QuantHandler(model, groupsize)
        model = simple_quantizer.convert_for_runtime()

    checkpoint = torch.load(str(checkpoint_path), mmap=True, weights_only=True)
    if "model" in checkpoint and "stories" in str(checkpoint_path):
        checkpoint = checkpoint["model"]
    model.load_state_dict(checkpoint, assign=True)
    if use_tp:
        from tp import apply_tp

        print("Applying tensor parallel to model ...")
        apply_tp(model)

    model = model.to(device=device, dtype=precision)
    return model.eval()


def get_model_size(model):
    model_size = 0
    for name, child in model.named_children():
        if not isinstance(child, torch.nn.Embedding):
            for p in itertools.chain(child.parameters(), child.buffers()):
                model_size += p.numel() * p.dtype.itemsize
    return model_size
