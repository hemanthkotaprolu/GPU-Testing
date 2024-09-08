import os
import numpy as np
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

from datasets import Dataset, load_dataset
from transformers import (
    T5ForConditionalGeneration, 
    T5Tokenizer, 
    AutoModelForCausalLM, 
    AutoTokenizer, 
    BitsAndBytesConfig,
    HfArgumentParser,
    TrainingArguments, 
    Trainer, 
    pipeline
)
from peft import (
    LoraConfig, 
    PeftModel, 
    prepare_model_for_kbit_training, 
    get_peft_model
)
from trl import SFTTrainer
import accelerate

from optimum.habana import GaudiConfig, GaudiTrainer, GaudiTrainingArguments

def convert_to_alpaca(item):
    return "Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n" + f"### Instruction:\n{instruction}\n\n### Input:\n{item['input']}\n\n### Response:\n{item['output']}"


dataset = load_dataset('hemanthkotaprolu/goemotions-plutchiks')
train_dataset = dataset['train']
train_dataset = train_dataset.map(lambda item: {'text': convert_to_alpaca(item)})

base_model = "meta-llama/Meta-Llama-3-8B-Instruct"
new_model = "./models/Llama3_8b_instruct_finetuned_e5_lr_5_bs_4"

# Load pretrained mBART model and tokenizer
#model_name = "facebook/mbart-large-cc25"
#tokenizer = MBartTokenizer.from_pretrained(model_name)
#model = MBartForConditionalGeneration.from_pretrained(model_name)

model = AutoModelForCausalLM.from_pretrained(
    base_model,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True
)
model.config.use_cache = False # silence the warnings
model.config.pretraining_tp = 1
model.gradient_checkpointing_enable()


tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
tokenizer.padding_side = 'right'
tokenizer.pad_token = tokenizer.eos_token
tokenizer = T5Tokenizer.from_pretrained("t5-small")

model.resize_token_embeddings(len(tokenizer))
model.config.pad_token_id = tokenizer.pad_token_id


peft_config = LoraConfig(
    lora_alpha=8,
    r=16,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj","gate_proj","down_proj","up_proj"]
)

model = get_peft_model(model, peft_config)


training_args = GaudiTrainingArguments(
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=5,
    learning_rate=1e-5,
    weight_decay=0.01,
    save_strategy="epoch",
    output_dir='./logs/llama3_8b_lr5_e5_bs8',
    use_habana=True,
    use_lazy_mode=True,
)


trainer = GaudiTrainer(
    model=model,
    train_dataset=train_dataset,
    tokenizer=tokenizer,
    args=training_args,
    packing=False,
    dataset_text_config="text",
    peft_config=peft_config
)


trainer.train()

results = trainer.evaluate()
output_dir = "./fine_tuned_model"
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)
print(results)
