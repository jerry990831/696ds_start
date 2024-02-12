import json

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import login

torch_device = "cuda" if torch.cuda.is_available() else "cpu"
login("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")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
data_path = "dataset/train_rand_split.jsonl"
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf").to(torch_device)
dataset = []
with open('dataset/train_rand_split.jsonl', 'r') as f:
    for line in f:
        data = json.loads(line)
        dataset.append(data)


def evaluate_question(question, decode_method="greedy"):
    prompt = "choose a choice in this question"
    prompt += question['stem'] + "\n"
    for choice in question['choices']:
        prompt += f"{choice['label']}: {choice['text']}\n"
    print(prompt)
    print("--------------------------")
    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(torch_device)
    if decode_method == "greedy":
        output = model.generate(input_ids, max_length=200)
    print(tokenizer.decode(output[0], skip_special_tokens=True)[len(input_ids):])


num_correct = 0
dataset = dataset[:100]
for i in range(100):
    question1 = dataset[i]['question']
    answer_key1 = dataset[i]['answerKey']
    evaluate_question(question1)

