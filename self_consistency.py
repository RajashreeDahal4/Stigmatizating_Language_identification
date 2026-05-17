
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from typing import List, Dict
import os
import json
import ast
from collections import Counter
import pickle

hf_token=os.getenv("HF_TOKEN")

with open("config.json",'rb') as f:
        data=json.load(f)
bias_identifier_prompt=data['bias_identifier']
model_path="/home/rdaha1@AD/bias/Multi-Agents-Debate/models/medgemma-27b-text-it"


tokenizer = AutoTokenizer.from_pretrained(model_path,use_auth_token=hf_token)

# Load model
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map="auto",
    use_auth_token=hf_token
)


def self__consistency(prompt: str) -> Dict:
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}  #new addition
    input_length = inputs["input_ids"].shape[1] 

    votes=[]
    for i in range(0,8):
        # Generate output while ensuring no repetition of input
        output_tokens = model.generate(
            **inputs,
            do_sample=True,  # Sampling for diversity, optional
            temperature=0.5,  # Adjust for creativity
            return_dict_in_generate=True,
            max_length=4096,
            output_scores=True
        )
        new_tokens = output_tokens.sequences[:, input_length:]  

        response = tokenizer.decode(new_tokens[0], skip_special_tokens=True)
        try:
            list_start = response.find('[')
            list_end = response.rfind(']') + 1
            list_part = response[list_start:list_end]
            bias_list=ast.literal_eval(list_part)

            votes.extend(bias_list)
        except Exception as e:
            continue
    if not votes:
        return [{"bias_type": "", "quote": ""}]
    else:
        count_dict = {}
        for item in votes:
            try:
                key = (item["bias_type"], item["quote"])
                if key in count_dict:
                    count_dict[key] += 1
                else:
                    count_dict[key] = 1
            except:
                continue

        vote_items = [
            {"bias_type": bias_type, "quote": quote}
            for (bias_type, quote), count in count_dict.items()
            if count > 2
        ]
        return vote_items
    
def process_chunks_and_dump(
    chunks: List[Dict],
    bias_identifier_prompt: str,
    output_file: str
):
    results = []
    for chunk_data in chunks:
        chunk_paragraph = chunk_data["chunk_paragraph"]
        chunk_start_word_pos = chunk_data["chunk_start_word_pos"]
        filled_prompt = bias_identifier_prompt.replace("##medical_note##", chunk_paragraph)

        consistent_list = self__consistency(filled_prompt)
        for bias_result in consistent_list:
            try: 
                if bias_result.get("bias_type") and bias_result.get("quote"):
                    quote_start_pos = chunk_paragraph.find(bias_result["quote"])
                    if quote_start_pos == -1:
                        quote_start_pos = None

                    results.append({
                        "bias_type": bias_result["bias_type"],
                        "quote": bias_result["quote"],
                        "chunk_paragraph": chunk_paragraph,
                        "quote_start_pos_in_chunk": quote_start_pos,
                        "chunk_start_word_pos_in_doc": chunk_start_word_pos
                    })
            except: 
                print("Faiiled",bias_result)
                continue
    with open(output_file, "w") as f:
        json.dump(results, f, indent=4)
    print(f"Dumped results to: {output_file}")

if __name__ == "__main__":
    with open("raw_data/pmc_patientsall.pkl",'rb') as file:
        patient_data=pickle.load(file)

    patient_data=patient_data[0:50]

    with open("code/utils/config.json",'rb') as f:
        data=json.load(f)
    bias_identifier_prompt=data['bias_identifier']

    window_size=300
    overlap=10
    step_size=window_size-overlap
    for enum,each_data in enumerate(patient_data):
        words = each_data.split()
        num_words = len(words)
        
        chunks = []
        for start_idx in range(0, num_words, step_size):
            end_idx = start_idx + window_size
            window = words[start_idx:end_idx]
            chunks.append({"chunk_paragraph":" ".join(window),"chunk_start_word_pos":start_idx})
        dataset="medgemma_pmc_self_consistency"
        process_chunks_and_dump(
            chunks=chunks,
            bias_identifier_prompt=bias_identifier_prompt,
            output_file=f"dumps/{dataset}/{enum}.json")


