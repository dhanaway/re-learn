# train_smoke.py
# Tier-0 SMOKE TEST: Qwen3-1.7B, 30 GRPO steps (~5-10 min on an H100).
# Goal: prove the GRPO + Stockfish-reward loop runs end-to-end and reward climbs.
# To scale to the real run, change every line marked  [14B].
from unsloth import FastLanguageModel
from datasets import load_from_disk
from trl import GRPOConfig, GRPOTrainer
from chess_reward import chess_reward

MAX_SEQ = 1024                                 # [14B] 2048

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "Qwen/Qwen3-1.7B",            # [14B] "Qwen/Qwen3-14B"
    max_seq_length = MAX_SEQ,
    load_in_4bit = False,                      # [14B] True
    fast_inference = True,                     # vLLM-backed fast rollouts
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

args = GRPOConfig(
    output_dir = "qwen3-1.7b-chess-grpo",
    per_device_train_batch_size = 4,           # must be a multiple of num_generations
    num_generations = 4,                       # [14B] 8  (candidate moves per position)
    gradient_accumulation_steps = 1,
    max_prompt_length = 512,
    max_completion_length = 256,               # [14B] 768
    learning_rate = 1e-5,
    temperature = 1.0,                         # exploration during rollouts
    beta = 0.02,                               # KL to base model
    max_steps = 30,                            # [14B] 500+
    logging_steps = 1,
    save_steps = 1000,                         # effectively skip checkpoints in the smoke run
    bf16 = True,
    use_vllm = True,
    report_to = "none",                        # set "wandb" after `wandb login`
)

trainer = GRPOTrainer(
    model = model,
    processing_class = tokenizer,
    reward_funcs = [chess_reward],
    args = args,
    train_dataset = dataset,
)
trainer.train()

# Merge LoRA into a standalone 16-bit model for easy inference/benchmarking.
model.save_pretrained_merged("qwen3-1.7b-chess-merged", tokenizer, save_method="merged_16bit")
print("\nSMOKE TEST DONE -> merged model in ./qwen3-1.7b-chess-merged")
