import os
import json
import argparse
from collections import Counter
from tqdm import tqdm
import time
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from fastchat.model import get_conversation_template
from fastchat.conversation import get_conv_template

def get_args():
    parser = argparse.ArgumentParser(description="Response Generation Manager.")
    parser.add_argument("--model_path", type=str, default="/data1/syt/models/llama2-7B-Chat", help="Default model path if not specified in gen_config.")
    parser.add_argument("--input_file", type=str, default="../questions/llama2_test.json", help="Input dataset file name")
    parser.add_argument("--batch_size", type=int, default=50, help="Number of samples per batch")
    parser.add_argument("--n_responses", type=int, default=3, help="Number of responses to generate per question")
    parser.add_argument("--temperature", type=float, default=0.95)
    parser.add_argument("--max_tokens", type=int, default=4096)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--repetition_penalty", type=float, default=1.0)
    parser.add_argument("--device", type=str, default="7", help="CUDA device index")
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["float16", "bfloat16"], help="Data type for model weights")
    return parser.parse_args()

def load_dataset_from_file(filename):
    with open(filename, 'r') as f:
        return json.load(f)

def save_dataset(dataset, filename):
    with open(filename, 'w') as f:
        json.dump(dataset, f, indent=2)

def extract_last_number(response):
    import re
    match = re.findall(r"-?\d*\.?\d+", response)  # Matches integers, decimals, and negative numbers
    return float(match[-1]) if match else None  # Return the last number as a float

def process_batch(batch, llm, tokenizer, args, n_responses):
    # Prepare prompts by duplicating each prompt n_responses times
    prompts = []
    for item in batch:
        #question = item['question']
        question = item['response'] 
        gen_config = item.get('gen_config', {})
        model_name = gen_config.get('model', args.model_path)

        # Generate prompt for each question
        conv = get_conv_template('llama-2')
        conv.append_message(conv.roles[0], question + " Please think step by step and provide a detailed explanation. Please provide the answer figure in the end of the response: \n\n#### Answer: ")
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        # Duplicate each prompt n_responses times
        prompts.extend([prompt] * n_responses)

    # Set sampling parameters for the entire batch
    sampling_params = SamplingParams(
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty
    )

    # Generate all responses in one batch
    responses_batch = llm.generate(prompts, sampling_params)
    

    # Prepare to collect inconsistent responses for later batch correction
    inconsistent_responses = []  # Store tuples of (item, index, incorrect_response)
    idx = 0  # Index to keep track of the current response set
    
    reflect_qurey = []

    for item in batch:
        # Get the n_responses for the current question
        responses = [responses_batch[idx + i].outputs[0].text.strip() for i in range(n_responses)]
        idx += n_responses  # Move to the next set of responses

        # Extract answers and determine the most frequent answer
        answers = [extract_last_number(resp) for resp in responses]
        answer_counter = Counter(answers)
        most_common_answer, count = answer_counter.most_common(1)[0]

        # Select a natural language response that matches the most common answer
        final_response = next(resp for resp in responses if extract_last_number(resp) == most_common_answer)

        # Handle inconsistent answers
        inconsistent_responses_for_item = [
            resp for resp in responses if extract_last_number(resp) != most_common_answer
        ]
        
        # Add inconsistent responses to the list for later batch generation
        for inc_resp in inconsistent_responses_for_item:
            inconsistent_responses.append((item, inc_resp, most_common_answer,final_response))

        # Save results (excluding corrected responses for now)
        item['responses'] = responses
        item['most_common_answer'] = most_common_answer
        item['final_response'] = final_response
        item['inconsistent_responses'] = inconsistent_responses_for_item
        
        text11 = """
            If you think the provided answer is correct, output the answer in the following format:
            "reflection": correct, 
            "the final answer": <Your Answer Here>
            
            If you think the provided answer is incorrect, rethink the question and output the new answer in the following format:
            "reflection": incorrect,
            "the new solution": <Your New Solution Here>
            "the final answer": <Your New Answer Here>
            
            """
        
        conv = get_conv_template('llama-2')
        conv.append_message(conv.roles[0], "Question:"+item['response']+"\nAnswers:"+final_response+text11)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt() 
        reflect_qurey.extend([prompt])
        
    reflect_response = llm.generate(reflect_qurey, sampling_params)
    idx = 0
    for item in batch:
        reflect = reflect_response[idx].outputs[0].text.strip()
        idx +=1
        item['reflect_response']=reflect
        
        
    
        

    # # Now, handle all inconsistent responses in one batch
    # if inconsistent_responses:
    #     # Create batch of correction prompts
    #     correction_prompts = []
    #     for item, inc_resp, most_common_answer,final_response in inconsistent_responses:
    #         model_name = item['gen_config'].get('model', args.model_path)
    #         conv = get_conv_template('llama-3')
    #         conv.append_message(conv.roles[0], item['response'] + inc_resp + f"\n\n Here is another solution: {final_response}, please reflect on the solution process above and show me why you are wrong or why you are right.")
    #         conv.append_message(conv.roles[1], None)  # No response yet
    #         #print("+++++++++++++++++++++++",conv.get_prompt())
    #         correction_prompts.append(conv.get_prompt())

    #     # Generate corrections in one batch
    #     correction_responses_batch = llm.generate(correction_prompts, sampling_params)

    #     # Assign the corrected responses back to the respective items in the batch
    #     correction_idx = 0
    #     for item, _, _,_ in inconsistent_responses:
    #         corrected_response = correction_responses_batch[correction_idx].outputs[0].text.strip()
    #         item.setdefault('corrected_responses', []).append(corrected_response)

    #         correction_idx += 1

    # # Save results (including corrected responses now)
    # for item in batch:
    #     item['corrected_responses'] = item.get('corrected_responses', [])
    #     item['inconsistent_responses'] = item.get('inconsistent_responses', [])

    return batch



def main():
    args = get_args()

    # Load dataset
    dataset = load_dataset_from_file(args.input_file)

    # Group items by model to avoid reloading models
    model_to_items = {}
    for item in dataset:
        model_name = item['gen_config'].get('model', args.model_path)
        model_to_items.setdefault(model_name, []).append(item)

    for model_name, items in model_to_items.items():
        print(f"\nProcessing items for model: {model_name}")
        os.environ["CUDA_VISIBLE_DEVICES"] = args.device
        print("Initializing VLLM model...")

        # Initialize the model and tokenizer
        llm = LLM(
            model=args.model_path,
            dtype=args.dtype,
            trust_remote_code=True
        )
        tokenizer = AutoTokenizer.from_pretrained(args.model_path)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Process items in batches
        start = time.time()
        batch_size = args.batch_size
        num_batches = (len(items) + batch_size - 1) // batch_size
        for i in tqdm(range(num_batches), desc=f"Processing batches for model {args.model_path}"):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, len(items))
            batch = items[start_idx:end_idx]
            batch = process_batch(batch, llm, tokenizer, args, args.n_responses)
            items[start_idx:end_idx] = batch
            
        end = time.time()
        print(f"\nProcessed {len(items)} items in {end - start:.2f} seconds.")

    # Save the updated dataset
    output_file = f"{os.path.splitext(args.input_file)[0]}_answers.json"
    save_dataset(dataset, output_file)
    print(f"\nFinal dataset saved to {output_file}.")

if __name__ == "__main__":
    main()
