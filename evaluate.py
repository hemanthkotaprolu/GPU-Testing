
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from datasets import load_dataset

import tqdm
import json
from datetime import datetime
from zoneinfo import ZoneInfo

import re
import os
import shutil


device=torch.device("hpu")
model_name = "Qwen/Qwen2.5-7B-Instruct"
new_model = "/home/fine_tuned_model/models/Qwen25_7b_Instruct_finetuned_e1"

tokenizer = AutoTokenizer.from_pretrained(model_name, return_tensors="pt", return_token_type_ids=False)
model = AutoModelForCausalLM.from_pretrained(new_model, device_map="auto").to(device)


dataset = load_dataset("hemanthkotaprolu/goemotions-plutchiks")
testset = dataset["test"]


NUM_SAMPLES = testset.num_rows

prompt_template = """
Read the input text carefully and identify the emotion of the text. After identifying the emotion output the serial number of the emotion within the <emotion></emotion> tags. The emotions that you need to choose from along with definitions are:
0) Anger - Feeling upset, mad, or annoyed.
1) Anticipation - Looking forward to something, eager or excited about what's to come.
2) Joy - Feeling happy, cheerful, or pleased.
3) Trust - Feeling safe, confident, or positive towards someone or something.
4) Fear - Feeling scared, anxious, or worried.
5) Surprise - Feeling shocked or startled by something unexpected.
6) Sadness - Feeling unhappy, sorrowful, or down.
7) Disgust - Feeling grossed out, repulsed, or strongly disapproving.
8) Neutral - Feeling nothing in particular, indifferent or without strong emotions.

For example, if the identified emotion is Joy, then the output has to be <emotion>2</emotion>, since the serial number of joy is 2.

Input - {text}
"""


texts = testset[:NUM_SAMPLES]["input"]
labels = testset[:NUM_SAMPLES]["output"]


data = []
for i in tqdm.tqdm(range(len(texts))):
    text = texts[i]
    label = labels[i]

    prompt = prompt_template.format(text=text)
    
    chat = [{"role": "user", "content": prompt_template.format(text=text)+"\n"}]
    prompt = tokenizer.apply_chat_template(chat, tokenize=False)
    
    # print(prompt)
    
    encodings = tokenizer(prompt,return_tensors="pt").to(model.device)

    if 'token_type_ids' in encodings:
        print("++++++++++++++++++Yes++++++++++++++++++")
    outputs = model.generate(**encodings, 
                            do_sample=True,
                            temperature=0.2, 
                            max_new_tokens=10,
                            # lazy_mode=True, 
                            pad_token_id=tokenizer.eos_token_id)
    output_text = tokenizer.batch_decode(outputs)
    response = output_text[0][len(prompt)+17:]
    # print("+"*50)
    # print(response)
    temp_dict = {}
    temp_dict = {
        "text": text,
        "label": label,
        "response": response
    }
    
    data.append(temp_dict)

ist_time = datetime.now(ZoneInfo("Asia/Kolkata"))
timestamp = ist_time.strftime("%Y%m%d_%H%M")
json_name = new_model.split("/")[-1]
filename = f"./data/{json_name}_output_{timestamp}.json"


backup_dir = './data/backup/'

if os.path.exists(filename):
    shutil.move(filename, os.path.join(backup_dir, filename))
    print(f"Moved existing file to {backup_dir}/{filename}")

final_data = {
    "prompt_template": prompt_template,
    "data": data
}

with open(filename, "w") as json_file:
    json.dump(final_data, json_file, indent=4)


pattern = r'\b[0-8]\b'
def getInt(s):
    match = re.search(pattern, s)
    if match:
        return int(match.group())
    else:
        return None


with open("./data/output_20240923_2101.json", "r") as f:
    output = json.load(f)

noneCount = 0

for i in range(300):
    response = getInt(output["data"][i]["response"])
    if response == None:
        noneCount+=1
    # print(response)
    # print("="*50)
    
print(f"Number of Nones from {new_model} is {noneCount}")