# -*- coding: utf-8 -*-
"""
@Time ： 2024/10/12 下午7:24
@Auth ： simonzfei
@File ：finetune.py
@IDE ：PyCharm
@Motto：thinking coding
"""
import torch
from datasets import load_dataset
from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments
from torch.nn import CrossEntropyLoss

# Load the dataset
dataset = load_dataset("wikitext", "wikitext-103-v1")

# Load the GPT-2 tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

# Add a pad token to the tokenizer
tokenizer.pad_token = tokenizer.eos_token  # Use eos_token as pad_token

# Tokenize the dataset with padding and truncation
def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, max_length=1024, padding="max_length")

tokenized_datasets = dataset.map(tokenize_function, batched=True)

# Load pre-trained GPT-2 model
model = GPT2LMHeadModel.from_pretrained("gpt2")

# Define training arguments
training_args = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=4,  # Adjust batch size according to your GPU
    num_train_epochs=2,
    save_steps=10_000,
    save_total_limit=2,
    logging_dir="./logs",
)

# Subclass Trainer and override compute_loss
class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        outputs = model(**inputs)
        logits = outputs.logits
        labels = inputs["input_ids"]

        # Shift the logits and labels for language modeling
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        # Flatten the tensors for CrossEntropyLoss
        loss_fct = CrossEntropyLoss()
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        return (loss, outputs) if return_outputs else loss

# Initialize the custom Trainer
trainer = CustomTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
)

# Start fine-tuning
trainer.train()

# Save the model
trainer.save_model("./fine_tuned_gpt2")

# Evaluate the model on the validation set
eval_results = trainer.evaluate()

# Calculate perplexity
eval_loss = eval_results["eval_loss"]
perplexity = torch.exp(torch.tensor(eval_loss))

print(f"Perplexity: {perplexity.item()}")

# Generate text from a prompt
prompt = "Artificial intelligence is"
inputs = tokenizer(prompt, return_tensors="pt")

# Generate text
output = model.generate(**inputs, max_length=100)
generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
print(generated_text)
