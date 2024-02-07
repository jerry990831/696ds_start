import json

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
        break


def evaluate_question(question):
    prompt = "choose an answer of this question with any reason \n"+question['stem'] + "\n"
    for choice in question['choices']:
        prompt += f"{choice['label']}: {choice['text']}\n"
    inputs = tokenizer.encode(prompt, return_tensors='pt')
    output = model.generate(inputs, max_length=300, no_repeat_ngram_size=3,
                            temperature=0.5)
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    print(generated_text)
    return generated_text


for item in dataset:
    question = item['question']
    answer_key = item['answerKey']

    selected_answer = evaluate_question(question)
