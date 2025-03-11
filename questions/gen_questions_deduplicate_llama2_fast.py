import faiss
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from sentence_transformers import SentenceTransformer
from vllm import LLM, SamplingParams
import json
from transformers import AutoTokenizer
from tqdm import tqdm
import time
import os
import argparse
import torch

from datetime import timedelta

from preprocess.extract_only_question import clean_and_extract_question
from fastchat.model import get_conversation_template
from fastchat.conversation import get_conv_template


def parse_arguments():
    parser = argparse.ArgumentParser(description="Generate responses using VLLM")
    parser.add_argument("--model", type=str, default="/data1/syt/syt3/syt/models/Llama-2-7b-chat-hf", help="Model to use")
    parser.add_argument("--num_prompts", type=int, default=500, help="Number of prompts")
    parser.add_argument("--n", type=int, default=50, help="Number of questions to generate per prompt, total number of questions = n x num_prompts")
    parser.add_argument("--output_file", type=str, default="llama2_test.json", help="Output file name")
    parser.add_argument("--max_tokens", type=int, default=2048, help="Maximum number of tokens to generate")
    parser.add_argument("--temperature", type=float, default=1, help="Sampling temperature")
    parser.add_argument("--top_p", type=float, default=1, help="Top-p sampling parameter")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use (cuda or cpu)")
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["float16", "bfloat16"])
    parser.add_argument("--use_tokenizer_template", action="store_true", help="Use tokenizer template for generating the response")
    parser.add_argument("--swap_space", type=float, default=2.0)
    parser.add_argument("--max_model_len", type=int, default=4096)
    parser.add_argument("--gpu_id", type=int, default=0, help="GPU ID to use if device is cuda")
    parser.add_argument("--skip_special_tokens", action="store_true", help="Skip special tokens in output")
    return parser.parse_args()

args = parse_arguments()

with open("../configs/model_configs.json", "r") as f:
    model_configs = json.load(f)
    model_config = model_configs[args.model]
    stop_tokens = model_config["stop_tokens"]
    stop_token_ids = model_config["stop_token_ids"]
for i in range(torch.cuda.device_count()):
    print(torch.cuda.get_device_name(i))
# Initialize SentenceTransformer model
sentence_model = SentenceTransformer("/data1/syt/syt3/syt/models/all-MiniLM-L6-v2", device='cuda:2', trust_remote_code=True)

sample_text = "This is a sample text."
sample_vector = sentence_model.encode(sample_text)
embedding_dim = sample_vector.shape[0]

# Initialize FAISS index for similarity checking
faiss_index = None
faiss_vectors = np.empty((0, embedding_dim), dtype=np.float32)  


# Global list to store all previously generated prompts
all_generated_prompts = []

def add_to_faiss(new_vectors):
    global faiss_index, faiss_vectors
    new_vectors = np.vstack(new_vectors)
    if faiss_index is None:
        faiss_index = faiss.IndexFlatL2(embedding_dim)  
        print("FAISS index created.")
    faiss_index.add(new_vectors.astype(np.float32))
    faiss_vectors = np.vstack((faiss_vectors, new_vectors))  
    print(f"Added {new_vectors.shape[0]} vectors to FAISS index. Total vectors: {faiss_vectors.shape[0]}")


    
    
    
def find_duplicates_within_batch(vectors, threshold=0.85):
    """Find duplicates within the batch based on cosine similarity."""
    duplicate_indices = []
    print("Checking for duplicates within the batch...")
    start = time.time()
    for i, vector in enumerate(vectors):
        # Calculate cosine similarity between current vector and all previous vectors
        for j in range(i + 1, len(vectors)):
            sim = cosine_similarity([vector], [vectors[j]])[0][0]
            if sim >= threshold:  # If similarity is above the threshold, mark it as a duplicate
                duplicate_indices.append(i)
                break
    end = time.time()
    print(f"Time taken: {end - start:.2f}s")
    return duplicate_indices

def find_duplicates_with_history(vectors, threshold=0.85):
    global faiss_index
    if faiss_index is None or faiss_index.ntotal == 0:
        print("FAISS index is empty. No historical duplicates.")
        return []
    vectors = np.array(vectors).astype(np.float32)  # Shape: (n, embedding_dim)
    D, I = faiss_index.search(vectors, k=1)
    duplicates = []
    for i, (dist, idx) in enumerate(zip(D, I)):
        if dist[0] < threshold:
            duplicates.append(idx[0])
            print(f"Historical duplicate found for vector {i} with index {idx[0]} and distance {dist[0]}")
    return duplicates


def main():
    args = parse_arguments()
    start_time = time.time()
    total_gpu_hours = 0.0

    # Set device
    if args.device == "cuda":
        if torch.cuda.is_available():
            device = f"cuda:{args.gpu_id}"
            torch.cuda.set_device(args.gpu_id)
        else:
            print("CUDA is not available. Falling back to CPU.")
            device = "cpu"
    else:
        device = "cpu"

    print(f"Using device: {device}")
    
    #os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)

    # Initialize VLLM
    llm = LLM(model=args.model, 
              trust_remote_code=True,
              dtype=args.dtype,
              tensor_parallel_size=2, 
              max_model_len=args.max_model_len,
              swap_space=args.swap_space,
              gpu_memory_utilization=0.9, 
              #device=device,
              enable_prefix_caching=True)

    # Set up sampling parameters
    sampling_params_batch = SamplingParams(
        n=args.n,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        stop=stop_tokens,
        skip_special_tokens=args.skip_special_tokens
    )
    model_name = args.model
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Prepare prompts
    
    global_response_counter = 0
    prompt_results = []
    for i in enumerate(tqdm(range(args.num_prompts), desc="Generating responses")):
        #raw_prompts = ["Create one unique math word problem. \n\nProblem: " for _ in range(1)]
        raw_prompts = ["Use your creativity, create one unique math word problem. Please provide only the problem, no any other words. \n\nProblem: " for _ in range(1)]
        #raw_prompts = ["Please make this following math question a bit more complex:  \nThere are 15 apples in a basket. If 4 apples are given as a gift to a friend, how many apples will be left in the basket?" for _ in range(1)]
        prompts=[]
        for item in raw_prompts:
            question = item
            
            # Construct prompt based on conversation template or tokenizer
            
            
            if not args.use_tokenizer_template:
                
                #conv = get_conversation_template(model_name)
                conv = get_conv_template('llama-2')
            
                conv.append_message(conv.roles[0], question)
                conv.append_message(conv.roles[1], None)
                prompt = conv.get_prompt()
                
            else:
                chat = [{"role": "user", "content": question}]
                prompt = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
            prompts.append(prompt)

        # Generate responses
        outputs = llm.generate(prompts, sampling_params_batch)
        
        
        
        prompt_texts = []
        for output in outputs:
            for generated_text in output.outputs:
                processed_text = clean_and_extract_question(generated_text.text)
                prompt_texts.append(processed_text)
                # result = {
                #     "response_id": len(prompt_results),
                #     "global_response_id": global_response_counter, 
                #     "prompt": prompt,
                #     "response": processed_text,
                #     "gen_config": {
                #         "temperature": args.temperature,
                #         "top_p": args.top_p,
                #         "max_tokens": args.max_tokens,
                #         "model": args.model,
                #         "device": device
                #     }
                # }
                # prompt_results.append(result)
                # global_response_counter += 1
        
        # Vectorize the generated responses
        generated_vectors = sentence_model.encode(prompt_texts)
        
        # Find duplicates within the current batch (self-duplication)
        duplicate_indices = find_duplicates_within_batch(generated_vectors, threshold=0.85)
        if duplicate_indices:
            print(f"Found duplicates in current batch: {duplicate_indices}")
            print("duplicates: ", [prompt_texts[i] for i in duplicate_indices])
            # Remove duplicates from the current batch
            unique_prompts = [prompt_texts[i] for i in range(len(prompt_texts)) if i not in duplicate_indices]
        else:
            print("No duplicates found in the current batch.")
            unique_prompts = prompt_texts

        # Add the current batch to FAISS index (for historical comparison)
        #add_to_faiss([sentence_model.encode([text]) for text in unique_prompts])

        # Now compare with the entire history (previously generated prompts)
            # Initialize lists to collect duplicates for batch processing
        duplicates_to_modify = []
        similar_prompts_list = []  # To store similar prompts for each duplicate

        # Iterate through all unique prompts to check for historical duplicates
        for text in unique_prompts:
            # Encode the current text
            vector = sentence_model.encode(text)  # Shape: (embedding_dim,)

            # Check for duplicates in the historical FAISS index
            historical_duplicates = find_duplicates_with_history([vector], threshold=0.25)  # Adjust threshold as needed
            print(f"Found historical duplicates: {historical_duplicates}")

            if historical_duplicates:
                # Collect similar prompts from the historical library
                similar_prompts = [all_generated_prompts[idx] for idx in historical_duplicates]
                print(f"Found duplicates in history: {similar_prompts}, current: {text}")



                modified_prompt = f"This following generated math question:" + text + " is very similar to this question: " + similar_prompts[0] + ". Please modify it to make it different." 
                
                
                conv = get_conv_template('llama-2')
            
                conv.append_message(conv.roles[0], modified_prompt)
                conv.append_message(conv.roles[1], None)
                modi_prompt = conv.get_prompt()
                
                # Create a modified prompt to request regeneration
                # modified_prompt = (
                #     f"This following generated math question: \"{text}\" is very similar to this question: "
                #     f"\"{', '.join(similar_prompts)}\". Please modify it to make it different."
                # )

                # Append the modified prompt and corresponding text to their respective lists
                duplicates_to_modify.append(modi_prompt)
                similar_prompts_list.append(similar_prompts)  # Optional: If you need to reference similar prompts later
            else:
                # If no historical duplicates, save the unique prompt and its response as usual
                result = {
                    "response_id": len(prompt_results),
                    "global_response_id": global_response_counter, 
                    "prompt": prompt,  # Ensure 'prompt' is correctly defined in your context
                    "response": text,  # Directly use the generated text as the response
                    "gen_config": {
                        "temperature": args.temperature,
                        "top_p": args.top_p,
                        "max_tokens": args.max_tokens,
                        "model": args.model,
                        "device": device
                    }
                }
                prompt_results.append(result)
                global_response_counter += 1

                # Update history and FAISS index
                all_generated_prompts.append(text)
                unique_vector = sentence_model.encode(text)
                add_to_faiss([unique_vector])

        # After processing all unique_prompts, handle duplicates in batch
        if duplicates_to_modify:
            print(f"Batch regenerating {len(duplicates_to_modify)} duplicate prompts.")

            # Set up sampling parameters for batch generation
            sampling_params_batch_modify = SamplingParams(
                n=1,  # Generate one modification per prompt
                temperature=args.temperature,
                top_p=args.top_p,
                max_tokens=args.max_tokens,
                stop=stop_tokens,
                skip_special_tokens=args.skip_special_tokens
            )

            # Generate modified responses in batch
            modified_outputs = llm.generate(duplicates_to_modify, sampling_params_batch_modify)

            # Iterate through the generated modified responses
            for i, output in enumerate(modified_outputs):
                # Extract and clean the generated text
                modified_response_text = output.outputs[0].text
                modified_response_text = clean_and_extract_question(modified_response_text)
                
                # Update history and FAISS index with the modified response
                all_generated_prompts.append(modified_response_text)
                modified_vector = sentence_model.encode(modified_response_text)
                add_to_faiss([modified_vector])

                # Create a result entry for the modified response
                result = {
                    "response_id": len(prompt_results),
                    "global_response_id": global_response_counter, 
                    "prompt": duplicates_to_modify[i],
                    "response": modified_response_text,
                    "gen_config": {
                        "temperature": args.temperature,
                        "top_p": args.top_p,
                        "max_tokens": args.max_tokens,
                        "model": args.model,
                        "device": device
                    }
                }
                prompt_results.append(result)
                global_response_counter += 1

            print(f"Successfully regenerated {len(duplicates_to_modify)} duplicate prompts.")

        # Continue with the rest of your code...
        
    print(f"\nTotal GPU hours consumed: {total_gpu_hours:.4f} GPU-hours")
    print(f"Breakdown:")
    print("llama2_25k_gpu\n")
    print(f"- Total wall time: {timedelta(seconds=int(total_time_seconds))}")
    #print(f"- GPU parallelism: {tensor_parallel_size} GPU(s)")
    print(f"- Compute efficiency: {total_gpu_hours:.2f} GPU-hours")

    # Save results to file
    with open(args.output_file, "w") as f:
        json.dump(prompt_results, f, indent=2)

    print(f"Generated {len(prompt_results)} responses for {args.num_prompts} prompts. Saved to {args.output_file}")

if __name__ == "__main__":
    main()
