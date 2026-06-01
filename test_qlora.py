#!/usr/bin/env python3
import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel


DEFAULT_MODEL_ID = "Qwen/Qwen3-4B-Thinking-2507"


def load_jsonl(path: str) -> list[dict[str, Any]]:
    rows = []
    with open(path, "r") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def save_jsonl(rows: list[dict[str, Any]], path: str) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


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


def build_messages(item: dict[str, Any]) -> list[dict[str, str]]:
    question = item["question"]
    options = item.get("options")
    is_mcq = is_mcq_item(item)

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
            "Think carefully and compare every option. "
            "The final answer must be one capital letter inside \\boxed{}."
        )
        user = (
            f"{format_note}\n\n"
            f"Problem:\n{question}\n\n"
            f"Options:\n{format_options(options)}"
        )
    else:
        system = (
            "You are solving a math problem. "
            "Think carefully. "
            "The final answer must be inside \\boxed{}."
        )
        user = f"{format_note}\n\nProblem:\n{question}"

    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def extract_last_boxed(text: str) -> str:
    matches = re.findall(r"\\boxed\{([^}]*)\}", text)
    return matches[-1].strip() if matches else ""


def extract_letter(text: str) -> str:
    boxed = extract_last_boxed(text)
    if boxed:
        m = re.search(r"[A-Z]", boxed.upper())
        if m:
            return m.group(0)

    matches = re.findall(r"\b([A-Z])\b", text.upper())
    return matches[-1] if matches else ""


def force_boxed_if_needed(response: str, is_mcq: bool) -> str:
    response = response.strip()

    if not response:
        return response

    if "\\boxed{" in response:
        return response

    if is_mcq:
        letter = extract_letter(response)
        if letter:
            return f"\\boxed{{{letter}}}"
        return response

    # Do not wrap long unfinished reasoning into a box.
    return response


def score_mcq(response: str, gold_letter: Any) -> bool:
    return extract_letter(response) == str(gold_letter).strip().upper()


def score_free_form(response: str, gold: Any) -> bool:
    gold_list = gold if isinstance(gold, list) else [gold]

    try:
        sys.path.insert(0, ".")
        from judger import Judger  # type: ignore

        judger = Judger(strict_extract=False)
        return bool(
            judger.auto_judge(
                pred=response,
                gold=gold_list,
                options=[[]] * len(gold_list),
            )
        )
    except Exception:
        pred = extract_last_boxed(response) or response.strip()
        return pred in [str(g).strip() for g in gold_list]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--adapter_dir", default="qwen_qlora_adapter")
    parser.add_argument("--data_path", default="data/public.jsonl")
    parser.add_argument("--output_path", default="results/qlora_results.jsonl")
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument("--max_input_length", type=int, default=8192)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=0.95)
    args = parser.parse_args()

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    print("Loading base model in 4-bit...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )

    base_model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        trust_remote_code=True,
        quantization_config=bnb_config,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    print(f"Loading adapter from {args.adapter_dir}...")
    model = PeftModel.from_pretrained(base_model, args.adapter_dir)
    model.eval()

    data = load_jsonl(args.data_path)
    if args.limit is not None:
        data = data[:args.limit]

    results = []

    for item in tqdm(data, desc="QLoRA generating"):
        is_mcq = is_mcq_item(item)
        messages = build_messages(item)

        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False
        )

        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=args.max_input_length,
        ).to(model.device)

        do_sample = args.temperature > 0

        gen_kwargs = {
            "max_new_tokens": args.max_new_tokens,
            "do_sample": do_sample,
            "pad_token_id": tokenizer.eos_token_id,
        }
        if do_sample:
            gen_kwargs["temperature"] = args.temperature
            gen_kwargs["top_p"] = args.top_p

        with torch.no_grad():
            output_ids = model.generate(**inputs, **gen_kwargs)

        generated_ids = output_ids[0][inputs["input_ids"].shape[1]:]
        response = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        response = force_boxed_if_needed(response, is_mcq)

        if args.eval:
            if "answer" not in item:
                raise KeyError("You passed --eval but data row has no 'answer'.")
            gold = item["answer"]
            correct = score_mcq(response, gold) if is_mcq else score_free_form(response, gold)
            record = {
                "id": item.get("id"),
                "is_mcq": is_mcq,
                "gold": gold,
                "response": response,
                "correct": correct,
            }
        else:
            record = {
                "id": item.get("id"),
                "is_mcq": is_mcq,
                "response": response,
            }

        results.append(record)
        save_jsonl(results, args.output_path)

    print(f"Saved {len(results)} records to {args.output_path}")

    if args.eval:
        mcq = [r for r in results if r["is_mcq"]]
        free = [r for r in results if not r["is_mcq"]]

        def acc(rows: list[dict[str, Any]]) -> float:
            return 100.0 * sum(bool(r["correct"]) for r in rows) / len(rows) if rows else 0.0

        print("=" * 60)
        print("QLORA EVALUATION RESULTS")
        print("=" * 60)
        print(f"MCQ      : {sum(r['correct'] for r in mcq):4d} / {len(mcq):4d} ({acc(mcq):.2f}%)")
        print(f"Free-form: {sum(r['correct'] for r in free):4d} / {len(free):4d} ({acc(free):.2f}%)")
        print(f"Overall  : {sum(r['correct'] for r in results):4d} / {len(results):4d} ({acc(results):.2f}%)")
        print("=" * 60)


if __name__ == "__main__":
    main()
