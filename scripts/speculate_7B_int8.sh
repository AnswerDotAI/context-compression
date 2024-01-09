export MODEL_REPO=codellama/CodeLlama-7b-Python-hf
export DRAFT_MODEL_REPO=PY007/TinyLlama-1.1B-intermediate-step-480k-1T
time python generate.py --compile  --draft_checkpoint_path checkpoints/$DRAFT_MODEL_REPO/model_int8.pth  --checkpoint_path checkpoints/$MODEL_REPO/model_int8.pth --speculate_k 4 --prompt "Hi my name is" --max_new_tokens 10 --num_samples 1  --temperature 0.5 --compile_prefill --prompt_file True
