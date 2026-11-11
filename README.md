# Qwen3 Math QA: Evaluation and QLoRA

This UC San Diego CSE 151B project explores how to evaluate and adapt **Qwen3-4B-Thinking-2507** for multiple-choice and free-response math questions. The final notebook calls a GPT script to generate teacher answers, turns those answers into training examples, and trains a QLoRA adapter. It then runs Qwen with the adapter and exports predictions. The standalone scripts also support scoring on labeled questions.

## Repository files

| File | Purpose |
| --- | --- |
| [`final_project.ipynb`](final_project.ipynb) | Runs the teacher-label workflow. Calls `run_gpt.py`, joins teacher answers to questions by ID, trains and tests the Qwen adapter, and exports a CSV. |
| [`run_gpt.py`](run_gpt.py) | Generates GPT answers through the OpenAI API. Saves each answer to JSONL, retries failed requests, and can resume by skipping completed IDs. |
| [`train_qlora.py`](train_qlora.py) | Trains a Qwen adapter from labeled JSONL. Use `--answer_key answer` for reference labels or `--answer_key teacher_response` for teacher labels. |
| [`test_qlora.py`](test_qlora.py) | Loads a trained adapter and writes JSONL predictions. With `--eval`, scores labeled inputs and prints accuracy by question type. |
| [`run_inference.py`](run_inference.py) | Wraps Qwen adapter inference and converts JSONL predictions into `results/submission.csv`. See the flag compatibility note below. |

## Setup

Run commands from the project root. The documented QLoRA configuration uses CUDA, bitsandbytes, and bfloat16. Use a CUDA-enabled PyTorch installation that matches your system and a GPU that supports bfloat16.

Install the adapter workflow's Python packages after setting up PyTorch. These package names describe the dependencies used by the scripts. Use the original environment's package versions when reproducing a run.

```bash
python -m pip install transformers peft trl datasets accelerate bitsandbytes tqdm
```

For the notebook's GPT stage, install `openai` and set `OPENAI_API_KEY` in the environment that launches Jupyter or the terminal. Keep the key out of the notebook and the repository. The standalone Qwen training and evaluation scripts run without an API key.

## Data format

Inputs use JSON Lines with one question per line. Each question has a unique `id` and a `question` string. Use `options` for multiple-choice choices. The scripts also support an explicit `is_mcq` flag. Labeled examples include an `answer` field. The following are synthetic examples.

```jsonl
{"id":1,"question":"What is 9 squared?","options":["18","72","81","90"],"answer":"C"}
{"id":2,"question":"Find the positive solution to x^2 = 49.","answer":"7"}
```

The commands below use `data/train.jsonl` and `data/eval.jsonl` as example paths for separate labeled splits. The notebook expects `data/private.jsonl`. Use authorized data for the teacher experiment because its prompts are sent to the API.

## Train and evaluate an adapter

### 1. Train on labeled questions

```bash
python train_qlora.py \
  --train_path data/train.jsonl \
  --answer_key answer \
  --adapter_dir qwen_qlora_adapter \
  --epochs 1 \
  --max_seq_length 2048
```

The training script uses a 4-bit NF4 base model with double quantization. By default, the LoRA adapter uses rank 16, alpha 32, and dropout 0.05. The adapter directory is used together with the original Qwen checkpoint during inference.

### 2. Run a small held-out evaluation

```bash
python test_qlora.py \
  --data_path data/eval.jsonl \
  --adapter_dir qwen_qlora_adapter \
  --output_path results/qlora_eval_smoke.jsonl \
  --max_new_tokens 512 \
  --eval \
  --limit 10
```

Inspect the first run for empty, truncated, or incorrectly formatted outputs. Remove `--limit 10` to evaluate the full split, and change the output path to keep the smoke-test results. For unlabeled inference, omit `--eval`.

### 3. Read the results

The evaluator prints correct counts, total counts, and accuracy separately for multiple-choice and free-response questions, then prints overall accuracy. With `--eval`, each JSONL row includes `id`, `is_mcq`, `gold`, `response`, and `correct`.

- **Multiple choice.** The evaluator extracts an option letter and compares it with the normalized reference letter.
- **Free response.** The evaluator uses the course's `Judger.auto_judge` when the separately supplied course grader is available. If importing or running the grader raises an exception, it falls back to comparing the extracted answer with the reference strings. The two scoring methods can treat equivalent math expressions differently.
- **Answer formatting.** The prompts ask for a LaTeX-style `\boxed{...}` answer. Post-processing attempts to put the response in the expected format. A well-formatted response can still be mathematically incorrect.

When comparing runs, record the split, model and adapter, generation limit, and grading method. Use the same split and grader for each variant. A measured QLoRA gain requires baseline and adapter scores from a shared held-out benchmark.

## Reproduce the notebook's teacher-label workflow

Open `final_project.ipynb` from the repository root. It runs GPT generation, joins the answers to questions, trains an adapter, runs Qwen inference, and converts the predictions to CSV.

### 1. Generate GPT teacher answers

The notebook calls `run_gpt.py` for this step. Use the output filename below so the notebook's preparation cell can find it.

```bash
python -m pip install openai
python run_gpt.py \
  --data_path data/private.jsonl \
  --output_path results/gpt55_private_submission.jsonl \
  --model gpt-5.5 \
  --max_output_tokens 32768 \
  --resume
```

The GPT script writes progress after each response. Use the same input and output paths when resuming a run. For a small scored GPT run, point the script at a labeled split and add `--eval --limit 5`, using a separate output path.

### 2. Build teacher labels and train

Run the notebook cell that joins GPT responses to the original questions by ID. It keeps nonempty responses containing a boxed answer, adds them as `teacher_response`, and writes `data/private_gpt_teacher_for_qwen.jsonl`. The next stage calls `train_qlora.py` with `--answer_key teacher_response`, three epochs, and a 4,096-token training sequence limit. It saves the adapter in `qwen_qlora_private_gpt_answer_only_adapter/`.

### 3. Generate Qwen predictions and export

```bash
python test_qlora.py \
  --data_path data/private.jsonl \
  --adapter_dir qwen_qlora_private_gpt_answer_only_adapter \
  --output_path results/qwen_learned_gpt_private.jsonl \
  --max_new_tokens 4096
```

Run the notebook's CSV export cell after inference. It extracts the answer from each response and writes `results/qwen_learned_gpt_private_submission.csv` with the columns `id` and `answer`. GPT generated the teacher labels and Qwen generated the final predictions.

**Evaluation boundary.** The notebook trains on GPT answers to the same questions it later uses for Qwen inference. This run measures reproduction of teacher labels if predictions are compared against those labels. To measure accuracy on unseen questions, create training and evaluation splits before generating teacher labels and score only on the held-out split.

## Troubleshooting

| Symptom | What to check |
| --- | --- |
| `unrecognized arguments: --temperature 0` | The notebook's inference cell and `run_inference.py` pass this flag, but the available `test_qlora.py` uses greedy decoding internally and has no temperature flag. Run the standalone command above or remove the flag and its value from the caller. |
| The notebook cannot find GPT answers | Match the GPT cell’s `--output_path` to `GPT_ANSWER_PATH` in the preparation cell. The saved notebook uses different names for these paths. |
| CUDA out-of-memory or dtype errors | Check GPU support for bfloat16 and available VRAM. For training, check sequence length and batch size. |
| Correct-looking free-response answers score as wrong | Check which grader was used and inspect the boxed answer. The string fallback may mark two equivalent expressions differently. |
| Missing `answer` during scoring | Use a labeled JSONL with `--eval`. Omit the flag for unlabeled inference. |
