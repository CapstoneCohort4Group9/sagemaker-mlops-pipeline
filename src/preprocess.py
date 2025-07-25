import argparse
import json
from tqdm import tqdm
from typing import List, Dict


def format_prompt_only_assistant(example: Dict) -> List[Dict]:
    convo = example.get("messages", [])
    prompts = []
    for i, turn in enumerate(convo):
        if turn["role"] == "assistant":
            context = ""
            for j in range(i):
                role = convo[j]["role"]
                value = convo[j]["content"]
                context += f"<|im_start|>{role}\n{value}<|im_end|>\n"
            context += "<|im_start|>assistant\n"
            label = turn["content"] + "<|im_end|>"
            prompts.append({
                "text": context.strip(),
                "labels": label.strip()
            })
    return prompts


def preprocess_dataset(input_path: str, output_path: str):
    processed = []
    with open(input_path, "r", encoding="utf-8") as infile:
        for line in tqdm(infile, desc=f"Processing {input_path}"):
            if not line.strip():
                continue
            try:
                example = json.loads(line)
                formatted = format_prompt_only_assistant(example)
                processed.extend(formatted)
            except Exception as e:
                print(f"Error processing line: {e}")
    with open(output_path, "w", encoding="utf-8") as outfile:
        for item in processed:
            json.dump(item, outfile, ensure_ascii=False)
            outfile.write("\n")
    print(f"Processed {len(processed)} examples saved to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess chat dataset for assistant-only training.")
    parser.add_argument("--train_input", required=True)
    parser.add_argument("--test_input", required=True)
    parser.add_argument("--train_output", required=True)
    parser.add_argument("--test_output", required=True)
    args = parser.parse_args()

    preprocess_dataset(args.train_input, args.train_output)
    preprocess_dataset(args.test_input, args.test_output)
