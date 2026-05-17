import json
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from typing import List, Dict
import pickle 
import ast
import os
# Set visible GPUs (1,2,3,4)
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
model_path="/home/rdaha1@AD/bias/Multi-Agents-Debate/models/Llama-3.3-70B-Instruct"

HF_TOKEN=os.getenv("HF_TOKEN")

from transformers import AutoModelForCausalLM, AutoTokenizer

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_path,use_auth_token=HF_token)
model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto")

generator = pipeline("text-generation", model=model, tokenizer=tokenizer)

def get_bias_from_chunk_llama(prompt: str) -> Dict:
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}  
    input_length = inputs["input_ids"].shape[1]
    # Generate output while ensuring no repetition of input
    output_tokens = model.generate(
        **inputs,
        do_sample=True,
        temperature=0.1,
        return_dict_in_generate=True,
        max_length=4096,
        output_scores=True
    )

    new_tokens = output_tokens.sequences[:, input_length:]
    response = tokenizer.decode(new_tokens[0], skip_special_tokens=True)

    # Try extracting JSON
    try:
        list_start = response.find('[')
        list_end = response.rfind(']') + 1
        list_part = response[list_start:list_end]
        bias_list=ast.literal_eval(list_part)
        return bias_list
    except Exception as e:
        print("Failed to parse model output:", e)
        return [{"bias_type": "", "quote": ""}]

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

        bias_list_result = get_bias_from_chunk_llama(filled_prompt)
        for bias_result in bias_list_result:
            try: 
                print("Bias result",bias_result,type(bias_result))
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
    with open("config.json",'rb') as f:
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
        dataset="llama70_pmc"
        os.makedirs(f"dumps/{dataset}", exist_ok=True)

        process_chunks_and_dump(
            chunks=chunks,
            bias_identifier_prompt=bias_identifier_prompt,
            output_file=f"dumps/{dataset}/{enum}.json")
