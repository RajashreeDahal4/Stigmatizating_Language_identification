
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from typing import List, Dict
import os
import json

bias_types={"Explicit Bias": "Explicit forms of bias include preferences, beliefs, and attitudes of which people are generally consciously aware, endorsed, and can be identified and communicated (e.g., labeling a patient as 'non-compliant' or 'irresponsible').",
            "Implicit Bias": "Implicit biases are unconscious mental processes that lead to associations and reactions that are automatic and without intention and actors have no awareness of the associations with a stimulus.  (e.g., noting that a patient is 'well-spoken for their background').",
            "Stigmatizing Language": "Stigmatizing language is any language in clinical documentation that portrays the patient in a negative or judgmental way—by expressing doubt about their credibility, implying they are difficult, noncompliant, or deceptive, or by using language that casts their behavior, attitude, or emotional state in a negative light. This includes both overt and subtle wording that reflects blame, shame, or disrespect. Example: 'claims to see a therapist'",
            "Racial Bias": "Racism is an organized social system in which the dominant racial group, based on an ideology of inferiority, categorizes and ranks people into social groups called ‘races’ and uses its power to devalue, disempower, and differentially allocate valued society resources and opportunities to groups defined as inferior. Example: Black male, non-compliant with medications.",
            "Gender Bias": "Gender bias refers to both assuming sameness when differences exist and assuming differences when none exist, leading to unequal treatment or stereotyping. Example: 'patient is a middle-aged woman with a demanding family life"
}


prompt="""You are a medical language model trained to identify bias in clinical notes. You previously predicted bias type in the input paragraph and the corresponding quote that highlights the bias.
Now reflect on that decision. Based on the definition for bias type and input paragraph, was your prediction for bias type correct?
Use the information to determine whether the quote, as it appears in the paragraph, matches the bias_type.
You are supposed to answer a single digit 1 if there exists bias mentioned above, else answer single digit 0.
Your answer should explicitly be a single digit 1 or 0. Do not explain or justify your answer.
###Note####: Ensure that you do not give any justification and reasoning for your answer.

###Example1:
**quote**: "The patient is non-compliant with their medication regimen."
**Paragraph**: "Despite multiple counseling sessions, the patient shows little interest in improving their condition."
**bias type**: Explicit Bias
** The definition of Explicit Bias is **: Overtly negative or judgmental language used in a medical note (e.g., labeling a patient as 'non-compliant' or 'irresponsible') 
Answer: 1

### Example2:
**quote**: "feeling tired during the day."
**Paragraph**: "The patient reports feeling tired during the day but denies any issues with sleep or recent lifestyle changes. Vitals are within normal range."
**bias type**: Explicit Bias
** The definition of Explicit Bias is **: Overtly negative or judgmental language used in a medical note (e.g., labeling a patient as 'non-compliant' or 'irresponsible') 
Answer: 0

### Question:

**quote**: ##quote##
**Paragraph**: ##paragraph## 
**bias type**: ##bias type##
** The definition of ##bias type## is **: ##definition##.
Answer: 
"""







# Set your token
hf_token=os.getenv("HF_TOKEN")
# model_path = "./models/Llama-3.1-8B-Instruct"
model_path="/home/rdaha1@AD/bias/Multi-Agents-Debate/models/medgemma-27b-text-it"


tokenizer = AutoTokenizer.from_pretrained(model_path,use_auth_token=hf_token)

# Load model
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map="auto",
    use_auth_token=hf_token
)


def self__reflection(prompt: str) -> Dict:
    # Tokenize and move to device
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}  
    input_length = inputs["input_ids"].shape[1] 
    output_tokens = model.generate(
        **inputs,
        do_sample=True, 
        temperature=0.1,
        return_dict_in_generate=True,
        max_new_tokens=1,
        output_scores=True
    )

    # Extract only the new generated tokens beyond the input length
    new_tokens = output_tokens.sequences[:, input_length:]  # Skip input tokens

    # Decode the newly generated tokens
    response = tokenizer.decode(new_tokens[0], skip_special_tokens=True)
    return int(response)

def vote_reflection(lst):
    return 1 if lst.count(1) >= lst.count(0) else 0


path="/home/rdaha1@AD/bias/Multi-Agents-Debate/dumps/rebuttal/medgemma_mimic_self_consistency"
files=os.listdir(path)
full_files=[path+"/"+i for i in files]
for each_file in full_files:
    dump_file=each_file.replace("medgemma_mimic_self_consistency","medgemma_mimic_consistent_reflection")
    with open(each_file,'rb') as file:
        sample_data=json.load(file)
    reflected_data=[]

    for each_data in sample_data:
        # print(each_data)
        reflections=[]
        for i in range(0,10):
            bias_type=each_data['bias_type']
            quote=each_data['quote']
            chunk_paragraph=each_data['chunk_paragraph']
            final_prompt=prompt.replace("##paragraph##",chunk_paragraph).replace("##bias type##",bias_type).replace("##quote##",quote).replace("##definition##",bias_types[bias_type])
            # print("The prompt is",final_prompt)
            single_reflection=self__reflection(final_prompt)
            reflections.append(single_reflection)
        reflection=vote_reflection(reflections)
        each_data['reflection']=reflection
        reflected_data.append(each_data)
    with open(dump_file, 'w') as outfile:
        json.dump(reflected_data, outfile, indent=4)

