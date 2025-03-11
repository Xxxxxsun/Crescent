import re

def clean_and_extract_question(response):
    # 去掉特殊符号，仅保留字母、数字、空格和常见标点符号
    cleaned_response = re.sub(r'[^a-zA-Z0-9\s.,:;?!]', '', response)
    
    # 检测是否包含问号，保留问号前面的部分
    if '?' in cleaned_response:
        cleaned_response = cleaned_response.split('?')[0] + '?'
    else:
        return ""  # 如果没有问号，则直接返回空

    # 使用正则表达式分割句子，保留分隔符
    sentences = re.split(r'([.:;?!])', cleaned_response)
    
    # 合并分割符和句子
    combined_sentences = []
    for i in range(0, len(sentences) - 1, 2):
        combined_sentences.append(sentences[i] + sentences[i + 1])
    if len(sentences) % 2 != 0:
        combined_sentences.append(sentences[-1])

    # 定义要过滤的关键词列表
    keywords_to_remove = ["of course", "math word problem","for you to solve","i'd be happy to","here's one","happy to help","here it is","happy to","i need","word problem","help me","your task","answer","here is one","here is a problem","math problem","heres one","solve it","sure thing","modified","math questions","math question","how about this","heres your problem"]

    # 遍历每个句子，过滤掉包含特定关键词的句子
    result_sentences = []
    for sentence in combined_sentences:
        # 去掉前后空格
        sentence = sentence.strip()

        # 如果句子不包含特定关键词，则保留
        if not any(keyword in sentence.lower() for keyword in keywords_to_remove):
            result_sentences.append(sentence)

    # 将保留的句子合并为最终的问题
    return " ".join(result_sentences)


if __name__ == "__main__":
    # 示例数据
    response = "\ud83d\ude0a\n\nOf course, I'd be happy to create a math word problem for you! Here it is:\n\nSarah is planning a party for her birthday and wants to make sure she has enough food for all of her guests. She has a bowl of chips that she knows will feed 6 people, and she also has a bag of pizza that will feed 8 people. How many total servings of food does Sarah have if she wants to make sure everyone gets at least 2 servings of food?"
    # 清理和提取问题
    extracted_question = clean_and_extract_question(response)
    print("Extracted Math Problem:", extracted_question)
