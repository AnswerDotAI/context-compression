import torch
from abc import ABC, abstractmethod


class PromptCompressor(ABC):
    def __init__(self, head_specific, **kwargs) -> None:
        # Assign each kwarg as an attribute of the class
        for key, value in kwargs.items():
            setattr(self, key, value)

        self.head_specific = head_specific
        assert self.is_compatible(), f"Prompt compressor ({self.__class__.__name__}) is not compatible with the chosen cache strategy."

    @abstractmethod
    def requires_attn(self) -> bool:
        pass

    @abstractmethod
    def __call__(self, *args: torch.Any, **kwds: torch.Any) -> torch.Any:
        pass

    @abstractmethod
    def is_compatible(self) -> bool:
        pass


class PromptCompressorRecentGlobal(PromptCompressor):
    def __init__(self, head_specific, **kwargs) -> None:
        super().__init__(head_specific, **kwargs)

    def is_compatible(self) -> bool:
        # Can be used with any cache
        return True

    def requires_attn(self) -> bool:
        return False

    def __call__(self, input_pos, k_val, v_val):
        # [global; ...; window - global] --> [global; window - global]
        # Indices for first global_tokens tokens and last (window - global_tokens) tokens
        # Making this a tensor seems to give a speedup, but I haven't fully benchmarked
        keep_idxs = torch.tensor(
            list(range(self.global_tokens))
            + list(
                range(
                    input_pos.shape[0] - self.max_cache_length + self.global_tokens,
                    input_pos.shape[0],
                )
            ),
            dtype=torch.long,
            device=k_val.device,
        )
        assert len(keep_idxs) == self.max_cache_length
        k_val = k_val[:, :, keep_idxs]
        v_val = v_val[:, :, keep_idxs]
        return keep_idxs, k_val, v_val


class PromptCompressorSnapKV(PromptCompressor):
    """
    Use SnapKV to compress the prompt
    Inspired by the pseudo code on Page 7 of https://arxiv.org/abs/2404.14469
    """

    def __init__(self, head_specific, **kwargs) -> None:
        super().__init__(head_specific, **kwargs)

        self.kernel_size = 5
        self.observation_len = 16

        self.pool = torch.nn.AvgPool1d(
            self.kernel_size,
            stride=1,
            padding=self.kernel_size // 2,
            ceil_mode=False,
            count_include_pad=False,
        )

    def is_compatible(self) -> bool:
        # Can only be used with head-specific KV-caches
        return self.head_specific

    def requires_attn(self) -> bool:
        return True

    def __call__(self, input_pos, k_val, v_val, attn):
        assert self.head_specific, "SnapKV can only be used with head-specific KV-caches, e.g., placing the same token in different locations across heads)."

        priority = attn[:, :, -self.observation_len :, :].mean(dim=2)
        prev_shape = priority.shape

        # We'll be returning the attention history so we need to keep a copy before it's modified
        attn_history = priority.clone()
        priority = self.pool(priority)
        assert (
            priority.shape == prev_shape
        ), f"Pooling operation should not change the dimension: {prev_shape} -> {priority.shape}"
        priority[:, :, -self.observation_len :] = (
            1.0  # Ensure the observation window is selected
        )
        keep_idxs = (
            priority.topk(self.max_cache_length, dim=-1).indices.sort(dim=-1).values
        )

        attn_history = attn_history.gather(2, keep_idxs)

        keep_idxs_rep = keep_idxs.unsqueeze(-1).expand(-1, -1, -1, k_val.shape[-1])
        k_val_compressed = k_val.gather(2, keep_idxs_rep)
        v_val_compressed = v_val.gather(2, keep_idxs_rep)

        return keep_idxs.squeeze(0), k_val_compressed, v_val_compressed, attn_history


class PromptCompressorL2(PromptCompressor):
    def __init__(self, head_specific, **kwargs) -> None:
        super().__init__(head_specific, **kwargs)

    def is_compatible(self) -> bool:
        return self.head_specific

    def requires_attn(self) -> bool:
        return False

    def __call__(self, input_pos, k_val, v_val):
        key_norm = torch.linalg.vector_norm(k_val, ord=2, dim=-1)

        # Give low score to global and recent tokens
        locality_mask = torch.logical_or(
            input_pos < self.global_tokens,
            input_pos >= input_pos.shape[0] - self.recent_window,
        ).view(1, 1, -1)

        eviction_scores = key_norm.masked_fill(locality_mask, float("-inf"))

        keep_idxs = (
            eviction_scores.topk(self.max_cache_length, dim=-1, largest=False)
            .indices.sort(dim=-1)
            .values
        )

        keep_idxs_rep = keep_idxs.unsqueeze(-1).expand(-1, -1, -1, k_val.shape[-1])
        k_val_compressed = k_val.gather(2, keep_idxs_rep)
        v_val_compressed = v_val.gather(2, keep_idxs_rep)

        return keep_idxs, k_val_compressed, v_val_compressed


def prompt_compressor_constructor(strategy):
    if strategy == "recent_global":
        return PromptCompressorRecentGlobal
    elif strategy == "snapkv":
        return PromptCompressorSnapKV
    elif strategy == "l2":
        return PromptCompressorL2
    else:
        raise ValueError(f"Unknown prompt compression strategy: {strategy}")
