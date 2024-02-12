import json
import random
import time

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import login

torch_device = "cuda" if torch.cuda.is_available() else "cpu"
login("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")



def generate_output_file_name(model_name, num_shots, dataset, decode_method, num_iter, num_beams, top_p):
    model_name = model_name.split('/')[-1]

    if decode_method == 'greedy':
        return f'result_{model_name}_{num_shots}_{dataset}_{decode_method}_{num_iter}.txt'
    elif decode_method == 'beam':
        return f'result_{model_name}_{num_shots}_{dataset}_{decode_method}_{num_iter}_{num_beams}.txt'
    elif decode_method == 'nucleus':
        return f'result_{model_name}_{num_shots}_{dataset}_{decode_method}_{num_iter}_{top_p}.txt'


def load_commonsenseQA():
    file_path = "dataset/train_rand_split.jsonl"

    data = []  # List to store the JSON objects
    # Open the file and read line by line
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            # Parse the JSON object in each line
            json_object = json.loads(line)

            # Do something with the JSON object (here we're just appending it to a list)
            data.append(json_object)

    return data


def experiment_query_text(num_shots: int, data):
    """

    :param n: Number of shots.
    :param data: The whole dataset.
    :return: Complete query and the correct answer of the last question.
    """

    random_elements = random.sample(data, num_shots + 1)
    query = 'Choose an answer of these questions with any reason.\n'
    for question in random_elements[:-1]:
        q = question['question']
        query += q['stem'] + '\n'
        for choice in q['choices']:
            query += choice['label'] + ': ' + choice['text'] + '\n'
        query += 'The correct answer is ' + question['answerKey'] + '\n\n'

    q = random_elements[-1]['question']
    query += q['stem'] + '\n'
    for choice in q['choices']:
        query += choice['label'] + ': ' + choice['text'] + '\n'

    return query, random_elements[-1]['answerKey']


def make_query(model_name, num_shots, dataset, decode_method, num_iter, num_beams=0, top_p=0.0):
    if decode_method == 'greedy':
        print(
            f'Generating queries with {model_name}, {num_shots}-shot with dataset {dataset}, using {decode_method} method to decode.\n')
    elif decode_method == 'beam':
        print(
            f'Generating queries with {model_name}, {num_shots}-shot with dataset {dataset}, using {decode_method} method to decode, beam width is {num_beams}.\n')
    elif decode_method == 'nucleus':
        print(
            f'Generating queries with {model_name}, {num_shots}-shot with dataset {dataset}, using {decode_method} method to decode, top p is {top_p}.\n')
    print(f'Query number: {num_iter} \n')
    # Load the model - make sure to use the correct model class for text generation (e.g., AutoModelForCausalLM)
    model = AutoModelForCausalLM.from_pretrained(model_name).to(torch_device)
    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    max_new_tokens = 30 * num_shots + 20
    seconds = 0
    data = []
    if dataset == 'commonsenseQA':
        data = load_commonsenseQA()

    output_file_name = generate_output_file_name(model_name, num_shots, dataset, decode_method, num_iter, num_beams,
                                                 top_p)
    with open('./results/' + output_file_name, 'w', encoding='utf-8') as f:
        for i in range(num_iter):
            input_text, answer = experiment_query_text(num_shots, data)
            # Encode input text and generate output
            model_inputs = tokenizer(input_text, return_tensors='pt').to(torch_device)

            start_time = time.time()
            # Generate text - adjust parameters like max_length as needed
            if decode_method == 'greedy':
                output = model.generate(
                    **model_inputs,
                    temperature=1.2,
                    max_new_tokens=max_new_tokens,
                )
            elif decode_method == 'beam':
                output = model.generate(
                    **model_inputs,
                    temperature=1.2,
                    max_new_tokens=max_new_tokens,
                    num_beams=num_beams,
                    early_stopping=True
                )
            elif decode_method == 'nucleus':
                output = model.generate(
                    **model_inputs,
                    temperature=1.2,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    top_p=top_p,
                    top_k=0
                )

            generated_text = tokenizer.decode(output[0], skip_special_tokens=True)

            end_time = time.time()
            seconds += end_time - start_time

            f.write('Next:\n')
            f.write('Generation: \n')
            f.write(generated_text[len(input_text):] + '\n')
            f.write('Ground Truth Answer: \n')
            f.write(answer + '\n')

        f.write(f'Average decoding time is {seconds / num_iter} seconds')


model_name = 'meta-llama/Llama-2-7b-hf'
torch_device = "cuda" if torch.cuda.is_available() else "cpu"

num_shots = 0
commonsenseQA = load_commonsenseQA()
make_query(model_name, num_shots, 'commonsenseQA', 'greedy', 300)
