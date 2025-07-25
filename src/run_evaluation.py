import os
import json
import argparse
import tarfile
from tqdm import tqdm
from pathlib import Path
import subprocess
import sys

# === Install dependencies ===
try:
    REQUIREMENTS_PATH = "/opt/ml/processing/code/requirements.txt"
    if os.path.exists(REQUIREMENTS_PATH):
        print(f"Installing dependencies from {REQUIREMENTS_PATH}")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", REQUIREMENTS_PATH])
    # Upgrade bitsandbytes if present (safe for 8-bit envs too)
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "bitsandbytes"])
except Exception as e:
    print(f"Dependency installation failed: {e}")

# === Heavy imports after pip install ===
import nltk
import evaluate
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

nltk.download("punkt")

def load_test_prompts(file_path: str):
    samples = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            if "prompt" in obj and "expected_output" in obj:
                samples.append(obj)
    return samples

def extract_tar_if_exists(tar_path, extract_dir):
    if os.path.isfile(tar_path):
        print(f"Extracting {tar_path} to {extract_dir}")
        with tarfile.open(tar_path, "r:gz") as tar:
            tar.extractall(path=extract_dir)
        print("Extraction completed.")

def evaluate_model(model_path, test_input_path, output_file):
    # Unpack if tar.gz present
    extract_tar_if_exists(f"{model_path}/model.tar.gz", model_path)

    print("Model directory contents:")
    for root, dirs, files in os.walk(model_path):
        for f in files:
            print(os.path.join(root, f))

    # Load tokenizer and model (merged FP16 version)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    model.eval()

    test_samples = load_test_prompts(test_input_path)
    preds, refs = [], []

    for sample in tqdm(test_samples, desc="Evaluating"):
        input_ids = tokenizer(sample["prompt"], return_tensors="pt").input_ids.to(model.device)
        with torch.no_grad():
            output_ids = model.generate(
                input_ids,
                max_new_tokens=256,
                pad_token_id=tokenizer.eos_token_id
            )[0]
        output_text = tokenizer.decode(output_ids[input_ids.shape[-1]:], skip_special_tokens=True).strip()
        preds.append(output_text)
        refs.append(sample["expected_output"])

    # Compute metrics
    rouge = evaluate.load("rouge")
    bleu = evaluate.load("bleu")
    meteor = evaluate.load("meteor")

    rouge_result = rouge.compute(predictions=preds, references=refs)
    bleu_result = bleu.compute(predictions=preds, references=refs)
    meteor_result = meteor.compute(predictions=preds, references=refs)

    metrics = {
                "custom_metrics":{
                    "rougeL": round(rouge_result["rougeL"], 4),
                    "bleu": round(bleu_result["bleu"], 4),
                    "meteor": round(meteor_result["meteor"], 4)
                 }
              }

    # Write to file
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(f"Evaluation metrics written to {output_path}")
    print(json.dumps(metrics, indent=2))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate merged Hermes model.")
    parser.add_argument("--model_dir", required=True, help="Path to merged model directory")
    parser.add_argument("--test_input", required=True, help="Path to test_prompts.jsonl downloaded from S3")
    parser.add_argument("--output_file", required=True, help="Path to write evaluation metrics JSON")
    args = parser.parse_args()

    evaluate_model(args.model_dir, args.test_input, args.output_file)
