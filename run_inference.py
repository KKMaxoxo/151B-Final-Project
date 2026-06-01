import json
from pathlib import Path

QUESTION_PATH = "data/private.jsonl"
GPT_ANSWER_PATH = "results/gpt_private.jsonl"
OUT_TRAIN_PATH = "data/answer_for_qwen.jsonl"

with open(QUESTION_PATH) as f:
    questions = [json.loads(line) for line in f if line.strip()]

with open(GPT_ANSWER_PATH) as f:
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

    item = dict(q_by_id[qid])
    item["teacher_response"] = response
    train_rows.append(item)

Path("data").mkdir(exist_ok=True)

with open(OUT_TRAIN_PATH, "w") as f:
    for row in train_rows:
        f.write(json.dumps(row) + "\n")
