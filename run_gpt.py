#!/usr/bin/env python3
import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

from openai import OpenAI
from tqdm import tqdm


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


def build_prompt(item: dict[str, Any], explain: bool) -> str:
    question = item["question"]
    options = item.get("options")
    is_mcq = is_mcq_item(item)

    if is_mcq:
        if explain:
            instruction = (
                "Solve this multiple-choice math problem carefully. "
                "You may explain your reasoning briefly. "
                "Only the final answer should be inside \\boxed{}. "
                "The final line must be exactly one capital letter in a box, for example \\boxed{C}. "
                "Do not box intermediate results."
            )
        else:
            instruction = (
                "Solve this multiple-choice math problem. "
                "Return ONLY the final answer as exactly one capital letter inside \\boxed{}, "
                "for example \\boxed{C}. No explanation."
            )
        return f"{instruction}\n\nProblem:\n{question}\n\nOptions:\n{format_options(options)}"

    if explain:
        instruction = (
            "Solve this math problem carefully. "
            "You may explain your reasoning briefly. "
            "Only the final answer should be inside \\boxed{}. "
            "The final line must be exactly the final answer inside \\boxed{}. "
            "Do not box intermediate results."
        )
    else:
        instruction = (
            "Solve this math problem. "
            "Return ONLY the final answer inside \\boxed{}. No explanation."
        )
    return f"{instruction}\n\nProblem:\n{question}"


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


def force_boxed(response: str, is_mcq: bool) -> str:
    response = response.strip()
    if "\\boxed{" in response:
        return response
    if is_mcq:
        letter = extract_letter(response)
        if letter:
            return f"\\boxed{{{letter}}}"
        return response
    return f"\\boxed{{{response}}}"


def score_mcq(response: str, gold_letter: Any) -> bool:
    return extract_letter(response) == str(gold_letter).strip().upper()


def score_free_form(response: str, gold: Any) -> bool:
    gold_list = gold if isinstance(gold, list) else [gold]
    try:
        sys.path.insert(0, ".")
        from judger import Judger
        judger = Judger(strict_extract=False)
        return bool(judger.auto_judge(pred=response, gold=gold_list, options=[[]] * len(gold_list)))
    except Exception:
        pred = extract_last_boxed(response) or response.strip()
        return pred in [str(g).strip() for g in gold_list]


def call_gpt(client: OpenAI, model: str, prompt: str, max_output_tokens: int, retries: int, sleep: float) -> str:
    last_error = None
    for attempt in range(retries):
        try:
            response = client.responses.create(
                model=model,
                input=prompt,
                max_output_tokens=max_output_tokens,
            )
            return response.output_text.strip()
        except Exception as e:
            last_error = e
            wait_time = sleep * (2 ** attempt)
            print(f"\nAPI error attempt {attempt + 1}/{retries}: {e}")
            print(f"Waiting {wait_time:.1f}s then retrying...")
            time.sleep(wait_time)
    raise RuntimeError(f"Failed after {retries} attempts. Last error: {last_error}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", default="data/public.jsonl")
    parser.add_argument("--output_path", default="results/gpt_results.jsonl")
    parser.add_argument("--model", default="gpt-5.5")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--eval", action="store_true", help="Use for public data with answers. Do not use for private.")
    parser.add_argument("--explain", action="store_true", help="Allow explanation/reasoning. More expensive and slower.")
    parser.add_argument("--max_output_tokens", type=int, default=256)
    parser.add_argument("--sleep", type=float, default=0.2)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--resume", action="store_true", help="If output file exists, skip already completed ids.")
    args = parser.parse_args()

    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError(
            "OPENAI_API_KEY is not set. In terminal run: export OPENAI_API_KEY=\"your_key_here\""
        )

    client = OpenAI()

    data = load_jsonl(args.data_path)
    if args.limit is not None:
        data = data[:args.limit]

    results = []
    completed_ids = set()
    if args.resume and Path(args.output_path).exists():
        results = load_jsonl(args.output_path)
        completed_ids = {row.get("id") for row in results}
        print(f"Resume mode: found {len(results)} existing results.")

    for item in tqdm(data, desc="GPT generating"):
        qid = item.get("id")
        if args.resume and qid in completed_ids:
            continue

        is_mcq = is_mcq_item(item)
        prompt = build_prompt(item, explain=args.explain)

        response = call_gpt(
            client=client,
            model=args.model,
            prompt=prompt,
            max_output_tokens=args.max_output_tokens,
            retries=args.retries,
            sleep=args.sleep,
        )
        response = force_boxed(response, is_mcq=is_mcq)

        if args.eval:
            if "answer" not in item:
                raise KeyError("You used --eval, but this data has no answer field.")
            gold = item["answer"]
            correct = score_mcq(response, gold) if is_mcq else score_free_form(response, gold)
            record = {
                "id": qid,
                "is_mcq": is_mcq,
                "gold": gold,
                "response": response,
                "correct": correct,
            }
        else:
            record = {
                "id": qid,
                "is_mcq": is_mcq,
                "response": response,
            }

        results.append(record)
        save_jsonl(results, args.output_path)
        time.sleep(args.sleep)

    save_jsonl(results, args.output_path)
    print(f"\nSaved {len(results)} records to {args.output_path}")

    if args.eval:
        mcq_rows = [r for r in results if r["is_mcq"]]
        free_rows = [r for r in results if not r["is_mcq"]]

        def acc(rows: list[dict[str, Any]]) -> float:
            if not rows:
                return 0.0
            return 100.0 * sum(bool(r["correct"]) for r in rows) / len(rows)

        print("=" * 60)
        print("GPT EVALUATION RESULTS")
        print("=" * 60)
        print(f"MCQ      : {sum(r['correct'] for r in mcq_rows):4d} / {len(mcq_rows):4d} ({acc(mcq_rows):.2f}%)")
        print(f"Free-form: {sum(r['correct'] for r in free_rows):4d} / {len(free_rows):4d} ({acc(free_rows):.2f}%)")
        print(f"Overall  : {sum(r['correct'] for r in results):4d} / {len(results):4d} ({acc(results):.2f}%)")
        print("=" * 60)


if __name__ == "__main__":
    main()
