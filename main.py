import json

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import login

login("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")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
data_path = "dataset/train_rand_split.jsonl"
dataset = []
with open('dataset/train_rand_split.jsonl', 'r') as f:
    for line in f:
        data = json.loads(line)
        dataset.append(data)


# def evaluate_question(question):
#     prompt = "choose an answer of this question with any reason \n" + question['stem'] + "\n"
#     for choice in question['choices']:
#         prompt += f"{choice['label']}: {choice['text']}\n"
#     inputs = tokenizer.encode(prompt, return_tensors='pt')
#     output = model.generate(inputs, max_length=300, no_repeat_ngram_size=3,
#                             temperature=1)
#     generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
#     print("________________________________________")
#     print(generated_text)
#     return generated_text
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


num_correct = 0
for item in dataset[:100]:
    question = item['question']
    answer_key = item['answerKey']
    output = evaluate_logic(question)
    if output == answer_key:
        num_correct += 1
print(num_correct / 100)
