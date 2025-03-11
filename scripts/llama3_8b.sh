#!/bin/bash

# ques parameters
model_path=""     
num_prompts=550                                
n=100                                           
max_tokens=2048                                 
temperature=1                                   
top_p=1                                        
device="cuda"                                   
dtype="bfloat16"                               
swap_space=2.0                                  
max_model_len=4096                             
gpu_id=0                                       
#skip_special_tokens=true                        
# ans parameters
model_path_ans=""    
batch_size=16                                   
max_tokens_ans=2048                              
temperature_ans=1                                  
top_p_ans=1                                                    
dtype_ans="bfloat16"                              
repetition_penalty=1.0



total_prompts=$((num_prompts * n))

timestamp=$(date +%s)

output_file="llama3_questions_num${total_prompts}_temp${temperature}_topp${top_p}_${timestamp}.json"

log_dir="../mathcorpus_data"
if [ ! -d "$log_dir" ]; then
    mkdir -p "$log_dir"
fi

job_name="${model_path##*/}_topp${top_p}_temp${temperature}_${timestamp}"
job_path="${log_dir}/${job_name}"
mkdir -p $job_path
exec > >(tee -a "$job_path/${job_name}.log") 2>&1

echo "job_path: $job_path"

echo "[run_mathcorpus.sh] Model Path: $model_path"
echo "[run_mathcorpus.sh] Output File: $output_file"
echo "[run_mathcorpus.sh] Total Prompts: $total_prompts"
echo "[run_mathcorpus.sh] Generation Config: n=$n, max_tokens=$max_tokens, temperature=$temperature, top_p=$top_p"
echo "[run_mathcorpus.sh] System Config: device=$device, dtype=$dtype, swap_space=$swap_space, gpu_id=$gpu_id"

echo "[run_mathcorpus.sh] Start Generating Questionss..."
CUDA_VISIBLE_DEVICES=$gpu_id python ../questions/gen_questions_deduplicate_llama3_fast.py \
    --model $model_path \
    --num_prompts $num_prompts \
    --n $n \
    --max_tokens $max_tokens \
    --temperature $temperature \
    --top_p $top_p \
    --device $device \
    --dtype $dtype \
    --swap_space $swap_space \
    --max_model_len $max_model_len \
    --gpu_id $gpu_id \
    --output_file "$job_path/$output_file"

echo "[run_mathcorpus.sh] Finished Generating Questionss!"


processed_output_file="processed_questions_num${total_prompts}_temp${temperature}_topp${top_p}_${timestamp}.json"


input_file="$job_path/$output_file"   
output_file="$job_path/$processed_output_file"  


echo "[run_mathcorpus.sh] Start Processing Questions..."
python ../questions/preprocess/preprocess_ques.py "$input_file" "$output_file"
echo "[run_mathcorpus.sh] Finished Processing Questions!"
             

echo "[run_mathcorpus.sh] Start Generating Answers!"
CUDA_VISIBLE_DEVICES=$gpu_id python ../answers/generate_two_way_answers_update3_llama3.py \
    --model_path $model_path_ans \
    --input_file $output_file \
    --batch_size $batch_size \
    --temperature $temperature_ans \
    --max_tokens $max_tokens_ans \
    --top_p $top_p_ans \
    --repetition_penalty $repetition_penalty \
    --device $gpu_id \
    --dtype $dtype_ans 

echo "[run_mathcorpus.sh] Finished Generating Answers!"