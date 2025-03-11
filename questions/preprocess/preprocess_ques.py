import json
import argparse
from extract_only_question import clean_and_extract_question  # Import the function

def process_file(filename,outputname):
    # 读取文件内容
    with open(filename, 'r', encoding='utf-8') as file:
        data = json.load(file)

    original_count = len(data)
    processed_data = []
    
    # 处理每个条目
    for entry in data:
        question = clean_and_extract_question(entry['response'])
        if question:  # 如果问题不为空
            new_entry = {
                #"prompt_id": entry["prompt_id"],
                "question_id": len(processed_data),  # 更新序号
                #"prompt": entry["prompt"],
                "question": question,  # 修改字段名称
                "gen_config": entry["gen_config"]
            }
            processed_data.append(new_entry)

    processed_count = len(processed_data)
    
    # 保存处理后的数据到新文件
    output_filename = outputname
    with open(output_filename, 'w', encoding='utf-8') as outfile:
        json.dump(processed_data, outfile, ensure_ascii=False, indent=2)

    print(f"Original count: {original_count}")
    print(f"Processed count: {processed_count}")

# 使用示例
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process a single JSON file to extract questions.")
    parser.add_argument("input_file", type=str, help="The input JSON file.")
    parser.add_argument("output_file", type=str, help="The output JSON file.")
    args = parser.parse_args()

    process_file(args.input_file, args.output_file)
