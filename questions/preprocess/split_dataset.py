import json

# 读取原始 JSON 文件
with open('/data1/syt/mathcorpus/questions/preprocess/processed_test_magpie_llama2_common_900k.json', 'r', encoding='utf-8') as infile:
    data = json.load(infile)

# 提取前 30,000 个对象
extracted_data = data[:30000]

# 将提取的数据写入新的 JSON 文件
with open('/data1/syt/mathcorpus/questions/preprocess/processed_test_magpie_llama2_common_30k.json', 'w', encoding='utf-8') as outfile:
    json.dump(extracted_data, outfile, ensure_ascii=False, indent=4)
