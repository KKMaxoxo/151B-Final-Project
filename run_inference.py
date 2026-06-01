import csv
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm
from openai import OpenAI

from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, PeftModel, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig


# ============================================================
# Configuration
# ============================================================

DEFAULT_MODEL_ID = "Qwen/Qwen3-4B-Thinking-2507"
GPT_MODEL = "gpt-5.5"

PRIVATE_PATH = "data/private.jsonl"

GPT_JSONL = "results/gpt55_private_submission.jsonl"
TEACHER_JSONL = "data/private_gpt_teacher_for_qwen.jsonl"

ADAPTER_DIR = "qwen_qlora_private_gpt_answer_only_adapter"

QWEN_JSONL = "results/qwen_learned_gpt_private.jsonl"
CSV_OUTPUT = "results/submission.csv"

INFER_MAX_NEW_TOKENS = 4096
INFER_TEMPERATURE = 0.0
INFER_TOP_P = 0.95
INFER_MAX_INPUT_LENGTH = 8192

GPT_MAX_OUTPUT_TOKENS = 32768
GPT_SLEEP = 0.2
GPT_RETRIES = 3

TRAIN_EPOCHS = 3
TRAIN_MAX_SEQ_LENGTH = 4096
TRAIN_BATCH_SIZE = 1
TRAIN_GRAD_ACCUM = 8
TRAIN_LR = 2e-4


# ============================================================
# Shared helpers
# ============================================================

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


def clean_response(text: str) -> str:
    text = str(text).strip()
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    text = text.replace("<think>", "").replace("</think>", "").strip()
    return text


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


# ============================================================
# STEP 1: GPT teacher generation
# ============================================================

def build_gpt_prompt(item: dict[str, Any], explain: bool = False) -> str:
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


def generate_gpt_teacher_answers(
    data_path: str = PRIVATE_PATH,
    output_path: str = GPT_JSONL,
    model: str = GPT_MODEL,
    max_output_tokens: int = GPT_MAX_OUTPUT_TOKENS,
    sleep: float = GPT_SLEEP,
    retries: int = GPT_RETRIES,
    resume: bool = True,
    force: bool = False,
) -> str:
    if Path(output_path).exists() and not force and resume:
        existing = load_jsonl(output_path)
        if len(existing) > 0:
            print(f"GPT answer file already exists: {output_path}")
            print(f"Existing rows: {len(existing)}")
    else:
        existing = []

    if not os.environ.get("OPENAI_API_KEY"):
        if Path(output_path).exists() and not force:
            print("OPENAI_API_KEY not set, but existing GPT JSONL found. Skipping GPT generation.")
            return output_path
        raise RuntimeError("OPENAI_API_KEY is not set. Required to generate GPT teacher answers.")

    client = OpenAI()

    data = load_jsonl(data_path)
    results = []
    completed_ids = set()

    if resume and Path(output_path).exists() and not force:
        results = load_jsonl(output_path)
        completed_ids = {row.get("id") for row in results}
        print(f"Resume mode: found {len(results)} existing GPT results.")

    for item in tqdm(data, desc="GPT generating"):
        qid = item.get("id")
        if resume and qid in completed_ids and not force:
            continue

        is_mcq = is_mcq_item(item)
        prompt = build_gpt_prompt(item, explain=False)

        response = call_gpt(
            client=client,
            model=model,
            prompt=prompt,
            max_output_tokens=max_output_tokens,
            retries=retries,
            sleep=sleep,
        )
        response = force_boxed(response, is_mcq=is_mcq)

        record = {
            "id": qid,
            "is_mcq": is_mcq,
            "response": response,
        }

        results.append(record)
        save_jsonl(results, output_path)
        time.sleep(sleep)

    save_jsonl(results, output_path)
    print(f"Saved {len(results)} GPT records to {output_path}")
    return output_path


# ============================================================
# STEP 2: Build teacher JSONL
# ============================================================

def build_teacher_jsonl(
    question_path: str = PRIVATE_PATH,
    gpt_answer_path: str = GPT_JSONL,
    out_train_path: str = TEACHER_JSONL,
    force: bool = False,
) -> str:
    if Path(out_train_path).exists() and not force:
        print(f"Teacher JSONL already exists: {out_train_path}")
        return out_train_path

    with open(question_path) as f:
        questions = [json.loads(line) for line in f if line.strip()]

    with open(gpt_answer_path) as f:
        gpt_answers = [json.loads(line) for line in f if line.strip()]

    q_by_id = {q["id"]: q for q in questions}

    train_rows = []

    for r in gpt_answers:
        qid = r["id"]
        if qid not in q_by_id:
            continue

        response = r.get("response", "").strip()

        if not response:
            continue
        if "\\boxed{" not in response:
            continue

        response = clean_response(response)

        item = dict(q_by_id[qid])
        item["teacher_response"] = response
        train_rows.append(item)

    Path("data").mkdir(exist_ok=True)

    with open(out_train_path, "w") as f:
        for row in train_rows:
            f.write(json.dumps(row) + "\n")

    print("Private questions:", len(questions))
    print("GPT answers:", len(gpt_answers))
    print("Training rows kept:", len(train_rows))
    print("Saved:", out_train_path)

    if train_rows:
        print(json.dumps(train_rows[0], indent=2)[:1500])

    return out_train_path


# ============================================================
# STEP 3: Train QLoRA
# ============================================================

def build_qlora_messages_with_answer(item: dict[str, Any]) -> list[dict[str, str]]:
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

    assistant_answer = clean_response(item["teacher_response"])

    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
        {"role": "assistant", "content": assistant_answer},
    ]


def train_qlora_adapter(
    train_path: str = TEACHER_JSONL,
    adapter_dir: str = ADAPTER_DIR,
    model_id: str = DEFAULT_MODEL_ID,
    epochs: float = TRAIN_EPOCHS,
    batch_size: int = TRAIN_BATCH_SIZE,
    grad_accum: int = TRAIN_GRAD_ACCUM,
    lr: float = TRAIN_LR,
    max_seq_length: int = TRAIN_MAX_SEQ_LENGTH,
    force: bool = False,
) -> str:
    adapter_model = Path(adapter_dir) / "adapter_model.safetensors"
    adapter_config = Path(adapter_dir) / "adapter_config.json"

    if adapter_model.exists() and adapter_config.exists() and not force:
        print(f"Adapter already exists: {adapter_dir}")
        return adapter_dir

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    print(f"Loading training data from {train_path}...")
    rows = load_jsonl(train_path)

    texts = []
    skipped = 0

    for item in rows:
        try:
            messages = build_qlora_messages_with_answer(item)
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
                enable_thinking=False,
            )
            texts.append({"text": text})
        except Exception as e:
            skipped += 1
            print(f"Skipping id={item.get('id')} because: {e}")

    if not texts:
        raise RuntimeError("No training examples created. Check teacher JSONL.")

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
        model_id,
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
        output_dir=adapter_dir,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=lr,
        num_train_epochs=epochs,
        logging_steps=10,
        save_steps=100,
        save_total_limit=2,
        bf16=True,
        report_to="none",
        optim="paged_adamw_8bit",
        max_length=max_seq_length,
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

    print(f"Saving adapter to {adapter_dir}...")
    trainer.model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)

    print("Done. Check adapter folder for adapter_model.safetensors and adapter_config.json.")
    return adapter_dir


# ============================================================
# STEP 4: Qwen + QLoRA inference
# ============================================================

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

    return response


def run_qwen_qlora_inference(
    data_path: str = PRIVATE_PATH,
    adapter_dir: str = ADAPTER_DIR,
    output_path: str = QWEN_JSONL,
    model_id: str = DEFAULT_MODEL_ID,
    max_new_tokens: int = INFER_MAX_NEW_TOKENS,
    max_input_length: int = INFER_MAX_INPUT_LENGTH,
    temperature: float = INFER_TEMPERATURE,
    top_p: float = INFER_TOP_P,
) -> str:
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
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
        model_id,
        trust_remote_code=True,
        quantization_config=bnb_config,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    print(f"Loading adapter from {adapter_dir}...")
    model = PeftModel.from_pretrained(base_model, adapter_dir)
    model.eval()

    data = load_jsonl(data_path)
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
            max_length=max_input_length,
        ).to(model.device)

        do_sample = temperature > 0

        gen_kwargs = {
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
            "pad_token_id": tokenizer.eos_token_id,
        }
        if do_sample:
            gen_kwargs["temperature"] = temperature
            gen_kwargs["top_p"] = top_p

        with torch.no_grad():
            output_ids = model.generate(**inputs, **gen_kwargs)

        generated_ids = output_ids[0][inputs["input_ids"].shape[1]:]
        response = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        response = force_boxed_if_needed(response, is_mcq)

        record = {
            "id": item.get("id"),
            "is_mcq": is_mcq,
            "response": response,
        }

        results.append(record)
        save_jsonl(results, output_path)

    print(f"Saved {len(results)} records to {output_path}")
    return output_path


# ============================================================
# STEP 5: CSV conversion
# ============================================================

def jsonl_to_submission_csv(jsonl_path: str, csv_path: str) -> str:
    rows = []

    with open(jsonl_path, "r") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                response = clean_response(r.get("response", ""))
                answer = extract_last_boxed(response)

                rows.append({
                    "id": r["id"],
                    "answer": answer,
                })

    rows = sorted(rows, key=lambda x: x["id"])

    Path(csv_path).parent.mkdir(parents=True, exist_ok=True)

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "answer"])
        writer.writeheader()
        writer.writerows(rows)

    return csv_path


# ============================================================
# Single entry point
# ============================================================

def run_inference(
    force_gpt: bool = False,
    force_teacher: bool = False,
    force_train: bool = False,
) -> str:
    """
    Single entry point.

    This function performs the whole pipeline, but skips cached expensive steps:
      - GPT generation skipped if results/gpt55_private_submission.jsonl exists.
      - teacher JSONL creation skipped if data/private_gpt_teacher_for_qwen.jsonl exists.
      - QLoRA training skipped if adapter files exist.
      - Qwen + QLoRA inference always runs and outputs results/submission.csv.
    """

    Path("data").mkdir(exist_ok=True)
    Path("results").mkdir(exist_ok=True)

    if not Path(PRIVATE_PATH).exists():
        raise FileNotFoundError(f"Missing private data: {PRIVATE_PATH}")

    if force_gpt or not Path(GPT_JSONL).exists():
        print("\n=== Step 1: GPT teacher generation ===")
        generate_gpt_teacher_answers(
            data_path=PRIVATE_PATH,
            output_path=GPT_JSONL,
            force=force_gpt,
        )
    else:
        print(f"\n=== Step 1 skipped: found {GPT_JSONL} ===")

    if force_teacher or not Path(TEACHER_JSONL).exists():
        print("\n=== Step 2: Build teacher JSONL ===")
        build_teacher_jsonl(
            question_path=PRIVATE_PATH,
            gpt_answer_path=GPT_JSONL,
            out_train_path=TEACHER_JSONL,
            force=force_teacher,
        )
    else:
        print(f"\n=== Step 2 skipped: found {TEACHER_JSONL} ===")

    adapter_model = Path(ADAPTER_DIR) / "adapter_model.safetensors"
    adapter_config = Path(ADAPTER_DIR) / "adapter_config.json"

    if force_train or not (adapter_model.exists() and adapter_config.exists()):
        print("\n=== Step 3: Train QLoRA adapter ===")
        train_qlora_adapter(
            train_path=TEACHER_JSONL,
            adapter_dir=ADAPTER_DIR,
            force=force_train,
        )
    else:
        print(f"\n=== Step 3 skipped: found adapter in {ADAPTER_DIR} ===")

    print("\n=== Step 4: Qwen + QLoRA inference ===")
    run_qwen_qlora_inference(
        data_path=PRIVATE_PATH,
        adapter_dir=ADAPTER_DIR,
        output_path=QWEN_JSONL,
        max_new_tokens=INFER_MAX_NEW_TOKENS,
        temperature=INFER_TEMPERATURE,
    )

    print("\n=== Step 5: Convert JSONL to CSV ===")
    csv_path = jsonl_to_submission_csv(QWEN_JSONL, CSV_OUTPUT)

    print(f"\nFinal submission CSV saved to: {csv_path}")
    return csv_path


if __name__ == "__main__":
    run_inference()
