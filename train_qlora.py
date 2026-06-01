#!/usr/bin/env python3

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig


DEFAULT_MODEL_ID = "Qwen/Qwen3-4B-Thinking-2507"


def load_jsonl(path: str) -> list[dict[str, Any]]:
    rows = []
    with open(path, "r") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def is_mcq_item(item: dict[str, Any]) -> bool:
    if "is_mcq" in item:
        return bool(item["is_mcq"])
    return bool(item.get("options"))


def format_options(options: Any) -> str:
    if not options:
        return ""
    if isinstance(options, dict):
        return "\n".join(f"{k}. {v}" for k, v in options.items())
    if isinstance(options, list):
        letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        return "\n".join(f"{letters[i]}. {opt}" for i, opt in enumerate(options))
    return str(options)


def box_answer(ans: Any) -> str:
    if isinstance(ans, list):
        if len(ans) == 1:
            ans = ans[0]
        else:
            ans = ", ".join(str(x) for x in ans)

    ans = str(ans).strip()
    if "\\boxed{" in ans:
        return ans
    return f"\\boxed{{{ans}}}"


def clean_teacher_response(text: str) -> str:
    text = str(text).strip()
    if not text:
        return ""
    if "\\boxed{" in text:
        return text
    return f"\\boxed{{{text}}}"


def build_messages(item: dict[str, Any], answer_key: str) -> list[dict[str, str]]:
    question = item["question"]
    options = item.get("options")
    is_mcq = is_mcq_item(item)

    if answer_key not in item:
        raise KeyError(f"Missing answer_key={answer_key!r}. Existing keys: {list(item.keys())}")

    if answer_key == "answer":
        assistant_answer = box_answer(item[answer_key])
    else:
        assistant_answer = clean_teacher_response(item[answer_key])

    format_note = (
        "Output format rules: Use exactly one final \\boxed{} answer. "
        "For MCQ, the box must contain only the option letter, not the option value. "
        "For multiple blanks, put all answers in order inside one \\boxed{} separated by commas. "
        "For free-form formulas, prefer Python-style syntax when appropriate: use * for multiplication, "
        "** for powers, exp(...) for exponentials, sqrt(...) for square roots. "
        "For decimal answers, keep high precision and do not round aggressively."
    )

    if is_mcq:
        system = (
            "You are solving a multiple-choice math problem. "
            "Think carefully and compare all options. "
            "Return the final answer as exactly one capital letter inside \\boxed{}."
        )
        user = (
            f"{format_note}\n\n"
            f"Problem:\n{question}\n\n"
            f"Options:\n{format_options(options)}"
        )
    else:
        system = (
            "You are solving a math problem. "
            "Think carefully before answering. "
            "Return the final answer inside \\boxed{}."
        )
        user = f"{format_note}\n\nProblem:\n{question}"

    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
        {"role": "assistant", "content": assistant_answer},
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--train_path", default="data/public.jsonl")
    parser.add_argument("--adapter_dir", default="qwen_qlora_adapter")
    parser.add_argument("--answer_key", default="answer")
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--grad_accum", type=int, default=8)
    parser.add_argument("--max_seq_length", type=int, default=4096)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    print(f"Loading training data from {args.train_path}...")
    rows = load_jsonl(args.train_path)
    if args.limit is not None:
        rows = rows[:args.limit]

    texts = []
    skipped = 0

    for item in rows:
        try:
            messages = build_messages(item, args.answer_key)
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
                enable_thinking=False
            )
            texts.append({"text": text})
        except Exception as e:
            skipped += 1
            print(f"Skipping id={item.get('id')} because: {e}")

    if not texts:
        raise RuntimeError("No training examples created. Check --train_path and --answer_key.")

    print(f"Training examples: {len(texts)} | skipped: {skipped}")
    print("First training example preview:")
    print(texts[0]["text"][:1500])

    dataset = Dataset.from_list(texts)

    print("Loading base model in 4-bit QLoRA mode...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        trust_remote_code=True,
        quantization_config=bnb_config,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )

    sft_config = SFTConfig(
        output_dir=args.adapter_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        logging_steps=10,
        save_steps=100,
        save_total_limit=2,
        bf16=True,
        report_to="none",
        optim="paged_adamw_8bit",
        max_length=args.max_seq_length,
        dataset_text_field="text",
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=dataset,
        peft_config=lora_config,
        processing_class=tokenizer,
    )

    print("Starting training...")
    trainer.train()

    print(f"Saving adapter to {args.adapter_dir}...")
    trainer.model.save_pretrained(args.adapter_dir)
    tokenizer.save_pretrained(args.adapter_dir)

    print("Done. Check adapter folder for adapter_model.safetensors and adapter_config.json.")


if __name__ == "__main__":
    main()
