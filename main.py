import json

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import login

torch_device = "cuda" if torch.cuda.is_available() else "cpu"
login("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")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf").to(torch_device)
data_path = "dataset/train_rand_split.jsonl"

dataset = []
with open('dataset/train_rand_split.jsonl', 'r') as f:
    for line in f:
        data = json.loads(line)
        dataset.append(data)


def evaluate_logic(question):
    prompt = question['stem'] + "\n"
    choices = []
    for choice in question['choices']:
        choices.append(choice['text'])
    encoded_inputs = [tokenizer.encode(prompt + " " + choice, return_tensors='pt') for choice in choices]
    logits_list = []
    for encoded_input in encoded_inputs:
        with torch.no_grad():
            outputs = model(encoded_input)
            logits = outputs.logits
            logits_list.append(logits)
    scores = [logit[:, -1, :].max(1).values.item() for logit in logits_list]
    best_choice_index = scores.index(max(scores))
    best_choice = chr(65 + best_choice_index)

    print(best_choice)
    return best_choice


def evaluate_logic_1_shot(question1, question2, answer1):
    prompt = "here is an example \n" + question1['stem'] + "\n"
    for choice in question1['choices']:
        prompt += f"{choice['label']}: {choice['text']}\n"
    prompt += "Correct answer is " + answer1
    prompt += question2['stem'] + "\n"
    choices = []
    for choice in question2['choices']:
        choices.append(choice['text'])
    encoded_inputs = [tokenizer.encode(prompt + " " + choice, return_tensors='pt') for choice in choices]
    logits_list = []
    for encoded_input in encoded_inputs:
        with torch.no_grad():
            outputs = model(encoded_input)
            logits = outputs.logits
            logits_list.append(logits)
    scores = [logit[:, -1, :].max(1).values.item() for logit in logits_list]
    best_choice_index = scores.index(max(scores))
    best_choice = chr(65 + best_choice_index)

    print(best_choice)
    return best_choice


def evaluate_question(question):
    prompt = "choose a choice in this question"
    prompt += question['stem'] + "\n"
    for choice in question['choices']:
        prompt += f"{choice['label']}: {choice['text']}\n"
    input_ids = tokenizer.encode(prompt, return_tensors='pt')
    greedy_output = model.generate(input_ids, max_length=200)
    print(tokenizer.decode(greedy_output[0], skip_special_tokens=True)[len(input_ids):])


num_correct = 0
dataset = dataset[:100]
for i in range(100):
    question1 = dataset[i]['question']
    answer_key1 = dataset[i]['answerKey']
    print("Question" + str(i))
    print("----------------------------------------------------")
    evaluate_question(question1)
    print("----------------------------------------------------")
    print(answer_key1)
