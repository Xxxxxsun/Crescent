import json
from tqdm import tqdm

# Transform dataset to a Axoltol supported format
def convert_to_sharegpt(json_file, output_file, id_prefix, start_id=0):
    with open(json_file, 'r') as file:
        data = json.load(file)

    with open(output_file, 'w') as file:
        for entry in data:
            conversation_id = f"{id_prefix}_{entry['response_id']+start_id}"
            instruction = entry['response']
            response = entry['final_response']
            conversations = [
                {"from": "human", "value": instruction},
                {"from": "gpt", "value": response}
            ]
            gen_input_configs = entry['gen_config']
            # gen_input_configs['pre_query_template'] = entry['pre_query_template']

            # if entry['gen_input_configs']['input_generator'] != entry['gen_response_configs']['output_generator']:
            #     raise ValueError("Input and output generators must be the same")
            
            # if id_prefix not in entry['gen_input_configs']['input_generator']:
            #     raise ValueError(f"Input generator must contain {id_prefix}")

            sharegpt_entry = {
                "conversation_id": conversation_id,
                #"instruction": instruction,
                #"response": response,
                "conversations": conversations,
                "gen_input_config": gen_input_configs,
        
            }

            file.write(json.dumps(sharegpt_entry) + '\n')
    
    print(f"Converted {len(data)} entries to {output_file}")
    return len(data)


input_files = [
    "/data1/syt/syt3/syt/mathcorpus/questions/llama2_13b_creative_200k_thres025_answer.json" 
]

# Convert each file to Axolotl format
idx = 0
id_prefix = "Llama2-13B-Chat"
converted_files = []
for i in tqdm(range(len(input_files))):
    converted_file_name = f"{id_prefix}_sharegpt_creative_shard{i}_200k.jsonl"
    len_data = convert_to_sharegpt(input_files[i], converted_file_name, id_prefix, idx)
    idx += len_data
    converted_files.append(converted_file_name)

# Concatenate all files
output_file = f"{id_prefix}_llama2mark1_creative_sharegpt_200k_thres025.jsonl"
with open(output_file, 'w') as outfile:
    for fname in tqdm(converted_files):
        with open(fname) as infile:
            for line in infile:
                outfile.write(line)

print(f"Concatenated {len(converted_files)} files to {output_file}")


