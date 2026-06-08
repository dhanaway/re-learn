# benchmark.py
# Final before/after benchmark on the FIXED val set (val_fens.json), greedy
# decoding -> a clean, deterministic apples-to-apples comparison.
#   python benchmark.py Qwen/Qwen3-1.7B            # baseline
#   python benchmark.py ./qwen3-1.7b-chess-merged  # trained
import os
os.environ["VLLM_ATTENTION_BACKEND"] = "TRITON_ATTN"   # avoid the broken FlashInfer backend

import sys
from vllm import LLM, SamplingParams
from chess_eval import load_val_fens, make_eval_engine, prompt_for, score_completion, summarize
from make_dataset import SYSTEM


def main(model_path):
    fens = load_val_fens()
    llm = LLM(model=model_path, max_model_len=1024, gpu_memory_utilization=0.85)
    tok = llm.get_tokenizer()
    eng = make_eval_engine()

    prompts = [prompt_for(tok, f, SYSTEM) for f in fens]
    outs = llm.generate(prompts, SamplingParams(temperature=0.0, max_tokens=64))  # greedy
    rows = [score_completion(f, o.outputs[0].text, eng) for f, o in zip(fens, outs)]
    eng.quit()

    m = summarize(rows)
    print(f"\n=== {model_path} ===")
    print(f"legal-move rate   : {m['legal_rate']:.1%}")
    print(f"top-1 match        : {m['top1']:.1%}")
    print(f"avg centipawn loss : {m['cp_loss']:.0f}   (lower = better)")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen3-1.7B")
