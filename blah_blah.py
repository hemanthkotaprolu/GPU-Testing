
import numpy as np
from tqdm import tqdm
# prompt: import xnli dataset from huggingface

from datasets import load_dataset

dataset = load_dataset("Divyanshu/indicxnli", 'bn')
#from transformers import MBartForConditionalGeneration, MBartTokenizer, Trainer, TrainingArguments

# Load pretrained mBART model and tokenizer
#model_name = "facebook/mbart-large-cc25"
#tokenizer = MBartTokenizer.from_pretrained(model_name)
#model = MBartForConditionalGeneration.from_pretrained(model_name)

from transformers import T5ForConditionalGeneration, T5Tokenizer, Trainer, TrainingArguments
import torch
import accelerate

tokenizer = T5Tokenizer.from_pretrained("t5-small")
model = T5ForConditionalGeneration.from_pretrained("t5-small")




def tokenize_function(examples):
    premise_encodings = tokenizer(examples["premise"], padding="max_length", truncation=True)
    hypothesis_encodings = tokenizer(examples["hypothesis"], padding="max_length", truncation=True)
    labels = hypothesis_encodings.input_ids.copy()
    labels = [[-100 if token == tokenizer.pad_token_id else token for token in example] for example in labels]
    return {
        "input_ids": premise_encodings.input_ids,
        "attention_mask": premise_encodings.attention_mask,
        "decoder_input_ids": hypothesis_encodings.input_ids,
        "decoder_attention_mask": hypothesis_encodings.attention_mask,
        "labels": labels
    }


train_tokenized_dataset = dataset["train"].map(tokenize_function, batched=True, batch_size=8)
train_tokenized_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'decoder_input_ids', 'decoder_attention_mask', 'labels'])


val_tokenized_dataset = dataset["validation"].map(tokenize_function, batched=True, batch_size=8)
val_tokenized_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'decoder_input_ids', 'decoder_attention_mask', 'labels'])


training_args = TrainingArguments(
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=5,
    output_dir='./logs',
)


train_dataset = dataset['train']
val_dataset = dataset['test']


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_tokenized_dataset,
    eval_dataset=val_tokenized_dataset,
    tokenizer=tokenizer,
)


trainer.train()

results = trainer.evaluate()
output_dir = "./fine_tuned_model"
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)
print(results)
