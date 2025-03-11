import json
import os
import random

def convert_sharegpt_data(
    data_dir, output_dir, data_file="/data1/syt/syt3/syt/mathcorpus/SFT_dataset_process/Llama2-13B-Chat_llama2mark1_creative_sharegpt_200k_thres025.jsonl", num_examples=None
):
    os.makedirs(output_dir, exist_ok=True)
    examples = []
    with open(os.path.join(data_dir, data_file), "r") as fin:
        for line in fin:
            examples.append(json.loads(line.strip()))

    if num_examples:
        examples = random.sample(examples, k=num_examples)

    output_path = os.path.join(output_dir, "llama2-13B-Chat_llama2mark1_creative_sharegpt_200k_thres025_SFTdata.jsonl")
    with open(output_path, "w") as fout:
        invalid_cnt = 0
        for idx, example in enumerate(examples):
            messages = []
            valid = True
            for message in example["conversations"]:
                if message["from"] == "human" or message["from"] == "user":
                    messages.append({"role": "user", "content": message["value"]})
                elif message["from"] == "gpt" or message["from"] == "chatgpt":
                    messages.append({"role": "assistant", "content": message["value"]})
                # elif message["from"] == "gpt_reflect" or message["from"] == "chatgpt":
                #     messages.append({"role": "assistant", "content": message["value"]})
                elif message["from"] == "system":
                    valid = False
                    invalid_cnt += 1
                    break
                elif message["from"] == "bing":
                    valid = False
                    invalid_cnt += 1
                    break
                else:
                    continue
            if messages and valid:
                fout.write(
                    json.dumps({"dataset": "sharegpt", "id": f"sharegpt_{example['conversation_id']}", "messages": messages}) + "\n"
                )
        if invalid_cnt > 0:
            print(f"# of invalid examples in sharegpt data: {invalid_cnt}")

convert_sharegpt_data("/data1/syt/syt3/syt/mathcorpus/SFT_dataset_process", "/data1/syt/syt3/syt/mathcorpus/SFT_dataset_process")

            
