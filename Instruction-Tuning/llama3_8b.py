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
    TaskType,
    prepare_model_for_kbit_training, 
    get_peft_model
)
import accelerate
from optimum.habana import GaudiConfig, GaudiTrainer, GaudiTrainingArguments

instruction = """
Read the input text carefully and identify the emotion of the text. After identifying the emotion output the serial number of the emotion. The emotions that you need to choose from along with defenitions are:
0) Anger - Feeling upset, mad, or annoyed.
1) Anticipation - Looking forward to something, eager or excited about what's to come.
2) Joy - Feeling happy, cheerful, or pleased.
3) Trust - Feeling safe, confident, or positive towards someone or something.
4) Fear - Feeling scared, anxious, or worried.
5) Surprise - Feeling shocked or startled by something unexpected.
6) Sadness - Feeling unhappy, sorrowful, or down.
7) Disgust - Feeling disgusted, repulsed, or strongly disapproving.
8) Neutral - Feeling nothing in particular, indifferent or without strong emotions.

For example, if the identified emotion is Joy, the output has to be 2, since the serial number of joy is 2.
"""

def convert_to_alpaca(item):
    return "Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n" + f"### Instruction:\n{instruction}\n\n### Input:\n{item['input']}\n\n### Response:\n{item['output']}"

def template(item, pipeline):
    chat = [{"role": "user", "content": instruction + item["input"]},{"role": "assistant", "content": item["output"]}]

    prompt = pipeline.tokenizer.apply_chat_template(chat, tokenize=False)
    return prompt

pipeline = transformers.pipeline(
    "text-generation",
    model=base_model,
    model_kwargs={"torch_dtype": torch.bfloat16},
    device_map="auto",
)

dataset = load_dataset('hemanthkotaprolu/goemotions-plutchiks')
train_dataset = dataset['train']
train_dataset = train_dataset.map(lambda item: {'text': template(item, pipeline)})

base_model = "meta-llama/Meta-Llama-3-8B-Instruct"
new_model = "./models/Llama3_8b_instruct_finetuned_e5_lr_5_bs_4"

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

model.resize_token_embeddings(len(tokenizer))
model.config.pad_token_id = tokenizer.pad_token_id

peft_config = LoraConfig(
                r=16,
                lora_alpha=8,
                lora_dropout=0.05,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj","gate_proj","down_proj","up_proj"],
                bias="none",
                task_type=TaskType.CAUSAL_LM,
                )

lora_model = get_peft_model(model, peft_config)

lora_model.print_trainable_parameters()
gaudi_config = GaudiConfig()
gaudi_config.use_fused_adam = True
gaudi_config.use_fused_clip_norm = True

training_arguments = TrainingArguments(
        save_strategy="epoch",
        output_dir="./logs/Llama3_8b_instruct_finetuned_e5_lr_5_bs_4_cp/",
        num_train_epochs=3,
        gradient_accumulation_steps = 4,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        lr_scheduler_type ='cosine',
        # report_to="wanb",
        learning_rate=1e-5,
        weight_decay=0.01,
        warmup_steps = 10,
        use_habana=True,
        use_lazy_mode=true,
    )

trainer = GaudiTrainer(
            model=lora_model,
            gaudi_config=gaudi_config,
            args=training_args,
            train_dataset=train_dataset if training_args.do_train else None,
            eval_dataset=eval_dataset if training_args.do_eval else None,
            tokenizer=tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_metrics if training_args.do_eval else None,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics if training_args.do_eval else None,
        )

# training_args = GaudiTrainingArguments(
#     per_device_train_batch_size=8,
#     per_device_eval_batch_size=8,
#     num_train_epochs=5,
#     learning_rate=1e-5,
#     weight_decay=0.01,
#     save_strategy="epoch",
#     output_dir='./logs/llama3_8b_lr5_e5_bs8',
#     use_habana=True,
#     use_lazy_mode=True,
#     GaudiConfig=""
# )


# trainer = GaudiTrainer(
#     model=model,
#     train_dataset=train_dataset,
#     tokenizer=tokenizer,
#     args=training_args,
#     packing=False,
#     dataset_text_config="text",
#     peft_config=peft_config
# )

trainer.train()

results = trainer.evaluate()
output_dir = "./fine_tuned_model"
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)
print(results)
