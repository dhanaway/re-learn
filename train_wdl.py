# train_wdl.py
# Same as train.py but optimizes WIN PROBABILITY (WDL) instead of centipawn loss.
# Trains a SEPARATE model (qwen3-1.7b-chess-wdl-merged) so the centipawn model is
# preserved for an A/B Elo comparison. vLLM ON (TRITON_ATTN; flashinfer uninstalled).
import os
os.environ["VLLM_ATTENTION_BACKEND"] = "TRITON_ATTN"   # avoid the broken FlashInfer backend

import json
import torch
from unsloth import FastLanguageModel
from datasets import load_from_disk
from transformers import TrainerCallback, AutoTokenizer
from trl import GRPOConfig, GRPOTrainer

from chess_reward_wdl import chess_reward_wdl              # <-- win-probability reward
from chess_eval import (load_val_fens, make_eval_engine, prompt_for,
                        score_completion, summarize)
from make_dataset import SYSTEM

MAX_SEQ    = 1024
EVAL_EVERY = 25
MERGED     = "qwen3-1.7b-chess-wdl-merged"                # <-- separate dir (keeps the cp model)

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "Qwen/Qwen3-1.7B",          # [14B] "Qwen/Qwen3-14B"
    max_seq_length = MAX_SEQ,
    load_in_4bit = False,                    # [14B] True
    fast_inference = True,                   # vLLM-backed fast rollouts (flashinfer removed + TRITON_ATTN)
    max_lora_rank = 32,
    gpu_memory_utilization = 0.6,
)

model = FastLanguageModel.get_peft_model(
    model,
    r = 32, lora_alpha = 32,
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj"],
    use_gradient_checkpointing = "unsloth",
    random_state = 3407,
)

dataset = load_from_disk("chess_positions")


class ChessEvalCallback(TrainerCallback):
    """Every EVAL_EVERY steps, score the model on the fixed val set and log it."""
    def __init__(self, tokenizer, fens, system, every=EVAL_EVERY):
        self.tok, self.fens, self.system, self.every = tokenizer, fens, system, every
        self.eng = make_eval_engine()
        self.history = []

    def _run(self, model, step):
        try:
            was_training = model.training
            model.eval()
            prev_cache = getattr(model.config, "use_cache", None)
            model.config.use_cache = True
            rows = []
            for fen in self.fens:
                prompt = prompt_for(self.tok, fen, self.system)
                inputs = self.tok(prompt, return_tensors="pt").to(model.device)
                with torch.no_grad():
                    out = model.generate(**inputs, max_new_tokens=64,
                                         do_sample=False, pad_token_id=self.tok.eos_token_id)
                text = self.tok.decode(out[0][inputs["input_ids"].shape[1]:],
                                       skip_special_tokens=True)
                rows.append(score_completion(fen, text, self.eng))
            model.config.use_cache = prev_cache
            if was_training:
                model.train()
            m = summarize(rows, step)
            self.history.append(m)
            json.dump(self.history, open("eval_history_wdl.json", "w"), indent=2)
            print(f"\n[eval @ step {step}] legal={m['legal_rate']:.0%}  "
                  f"top1={m['top1']:.0%}  cp_loss={m['cp_loss']:.0f}\n")
        except Exception as e:
            print(f"[eval @ step {step}] skipped: {e}")
            try:
                model.train()
            except Exception:
                pass

    def on_train_begin(self, args, state, control, model=None, **kw):
        self._run(model, 0)

    def on_step_end(self, args, state, control, model=None, **kw):
        if state.global_step % self.every == 0:
            self._run(model, state.global_step)


args = GRPOConfig(
    output_dir = "qwen3-1.7b-chess-wdl-grpo",
    per_device_train_batch_size = 8,
    num_generations = 8,
    gradient_accumulation_steps = 4,
    max_prompt_length = 512,
    max_completion_length = 256,
    learning_rate = 1e-5,
    lr_scheduler_type = "constant",
    warmup_steps = 10,
    temperature = 1.0,
    beta = 0.02,
    max_steps = 400,
    logging_steps = 1,
    save_steps = 1000,
    bf16 = True,
    use_vllm = True,
    report_to = "none",
)

trainer = GRPOTrainer(
    model = model,
    processing_class = tokenizer,
    reward_funcs = [chess_reward_wdl],
    args = args,
    train_dataset = dataset,
    callbacks = [ChessEvalCallback(tokenizer, load_val_fens(), SYSTEM)],
)
trainer.train()

model.save_pretrained_merged(MERGED, tokenizer, save_method="merged_16bit")
# unsloth writes a broken tokenizer; overwrite with the (identical) base one
AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B").save_pretrained(MERGED)
print(f"\nDONE -> merged model in ./{MERGED} (tokenizer re-saved; loads without UCI_TOKENIZER)")
