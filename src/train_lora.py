import os
import torch
import glob
import mlflow
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel
from trl import SFTTrainer
from datasets import load_dataset

# === Environment Variables ===
MODEL_NAME = os.environ.get("HF_MODEL_NAME", "NousResearch/Hermes-2-Pro-Mistral-7B")
DATA_PATH = os.environ.get("DATA_PATH", "/opt/ml/input/data/train/train.jsonl")
EVAL_PATH = os.environ.get("EVAL_PATH", "/opt/ml/input/data/eval/test.jsonl")
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "/opt/ml/model")
MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "")

# === MLflow Setup ===
if MLFLOW_TRACKING_URI:
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment("Hermes2Pro-Mistral-Finetuning")

with mlflow.start_run():
    # === Tokenizer ===
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.add_special_tokens({
        "additional_special_tokens": [
            "<|im_start|>", "<|im_end|>",
            "user", "assistant", "system",
            "<tool_call>", "</tool_call>",
            "<tool_response>", "</tool_response>",
            "tool"
        ]
    })

    # === Load and Preprocess Dataset ===
    IGNORE_INDEX = -100
    MAX_LENGTH = 4096

    def tokenize(example):
        prompt_ids = tokenizer(example["text"], add_special_tokens=False)["input_ids"]
        response_ids = tokenizer(example["labels"], add_special_tokens=False)["input_ids"]
        input_ids = prompt_ids + response_ids
        labels = [IGNORE_INDEX] * len(prompt_ids) + response_ids
        input_ids = input_ids[:MAX_LENGTH]
        labels = labels[:MAX_LENGTH]
        padding = MAX_LENGTH - len(input_ids)
        input_ids += [tokenizer.pad_token_id] * padding
        labels += [IGNORE_INDEX] * padding
        attention_mask = [1] * len(input_ids) + [0] * padding
        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask[:MAX_LENGTH]
        }

    train_dataset = load_dataset("json", data_files=DATA_PATH)["train"]
    eval_dataset = load_dataset("json", data_files=EVAL_PATH)["train"]
    train_dataset = train_dataset.map(tokenize, remove_columns=train_dataset.column_names)
    eval_dataset = eval_dataset.map(tokenize, remove_columns=eval_dataset.column_names)

    # === 8-bit Quant Config ===
    bf16 = torch.cuda.is_bf16_supported()
    bnb_config = BitsAndBytesConfig(
        load_in_8bit=True,
        bnb_8bit_compute_dtype=torch.bfloat16 if bf16 else torch.float16,
        bnb_8bit_use_double_quant=True,
        bnb_8bit_quant_type="nf4",
    )

    torch.cuda.empty_cache()

    # === Load Base Model ===
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        device_map="auto",
        low_cpu_mem_usage=True,
        quantization_config=bnb_config,
        torch_dtype=torch.bfloat16 if bf16 else torch.float16,
        trust_remote_code=True
    )
    base_model.resize_token_embeddings(len(tokenizer))

    # === Apply LoRA ===
    base_model = prepare_model_for_kbit_training(base_model)
    peft_config = LoraConfig(
        r=32,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "lm_head", "embed_tokens"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(base_model, peft_config)

    # === MLflow Logging ===
    mlflow.log_params({
        "model_name": MODEL_NAME,
        "learning_rate": 2e-4,
        "epochs": 1,
        "batch_size": 1,
        "gradient_accumulation": 4,
        "max_seq_length": MAX_LENGTH,
        "lora_r": 32,
        "lora_alpha": 16,
        "lora_dropout": 0.05,
        "dataset_train": DATA_PATH,
        "dataset_eval": EVAL_PATH
    })

    # Resume from checkpoint if one exists
    checkpoint_dir = "/opt/ml/checkpoints"
    last_checkpoint = None

    if os.path.isdir(checkpoint_dir):
        candidates = glob.glob(os.path.join(checkpoint_dir, "checkpoint-*"))
        if candidates:
            last_checkpoint = sorted(candidates)[-1]
            print(f"Found checkpoint to resume from: {last_checkpoint}")
        else:
            print("Checkpoint directory exists but no valid checkpoint found.")
    else:
        print("No checkpoint directory found — starting from scratch.")

    # === Training ===
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        save_total_limit=2,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=4,
        dataloader_num_workers=8,
        num_train_epochs=1,
        learning_rate=2e-4,
        logging_steps=1,
        save_strategy="epoch",
        eval_strategy="epoch",
        lr_scheduler_type="cosine",
        warmup_steps=10,
        bf16=bf16,
        gradient_checkpointing=True,
        optim="adamw_bnb_8bit",
        weight_decay=0.0,
        push_to_hub=False,
        report_to=["mlflow"],
        logging_dir="/opt/ml/output/logs"
    )

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator
    )

    if last_checkpoint:
        trainer.train(resume_from_checkpoint=last_checkpoint)
    else:
        trainer.train()

    metrics = trainer.evaluate()
    for k, v in metrics.items():
        mlflow.log_metric(k, v)

    # === Merge Adapter into Base Model ===
    print("Merging LoRA adapter into base model")
    merged_model = model.merge_and_unload()

    # === Save Final Merged Model ===
    print(f"Saving merged model to {OUTPUT_DIR}")
    merged_model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    mlflow.log_artifacts(OUTPUT_DIR, artifact_path="model")
