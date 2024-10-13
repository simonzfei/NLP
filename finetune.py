# -*- coding: utf-8 -*-
"""
@Time ： 2024/10/12 下午7:24
@Auth ： simonzfei
@File ：finetune.py
@IDE ：PyCharm
@Motto：thinking coding 
"""
from datasets import load_dataset
from transformers import GPT2Tokenizer


from transformers import GPT2LMHeadModel, Trainer, TrainingArguments


# Load the dataset
dataset = load_dataset("wikitext", "wikitext-103-v1")

# Check the structure of the dataset
# print(dataset)

#Tokenize the text for model input using GPT-2’s tokenizer:
# Load the GPT-2 tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

# Tokenize the dataset
def tokenize_function(examples):
    return tokenizer(examples["text"], return_special_tokens_mask=True)

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

# Initialize Trainer
trainer = Trainer(
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
print(f"Perplexity: {eval_results['perplexity']}")


# Generate text from a prompt
prompt = "Artificial intelligence is"
inputs = tokenizer(prompt, return_tensors="pt")

# Generate text
output = model.generate(**inputs, max_length=100)
generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
print(generated_text)


