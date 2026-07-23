from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

MODEL_NAME = "distilgpt2"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

dataset = load_dataset(
    "json",
    data_files="data/recipes.json",
    split="train"
)


def preprocess(example):

    text = f"""Instruction:
{example['instruction']}

Recipe:
{example['output']}
"""

    tokens = tokenizer(
        text,
        truncation=True,
        padding="max_length",
        max_length=256,
    )

    tokens["labels"] = tokens["input_ids"].copy()

    return tokens


tokenized_dataset = dataset.map(preprocess)

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,
)

training_args = TrainingArguments(
    output_dir="models",
    num_train_epochs=5,
    per_device_train_batch_size=2,
    logging_steps=1,
    save_strategy="epoch",
    report_to="none"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=data_collator,
)

trainer.train()

trainer.save_model("models")
tokenizer.save_pretrained("models")

print("\nTraining Completed Successfully!")