def get_pairs(text):
    pairs = {}
    for i in range(len(text) - 1):
        pair = (text[i], text[i + 1])
        if pair in pairs:
            pairs[pair] += 1
        else:
            pairs[pair] = 1
    return pairs

def merge_pair(text, pair):
    new_token = ''.join(pair)
    new_text = []
    i = 0
    while i < len(text):
        if i < len(text) - 1 and (text[i], text[i + 1]) == pair:
            new_text.append(new_token)
            i += 2  # Skip the next character as it's part of the pair
        else:
            new_text.append(text[i])
            i += 1
    return new_text

# 示例数据
text = "hello hello hello"
text_list = list(text)

# 统计频率并找到最频繁的二元组
pairs = get_pairs(text_list)
most_frequent_pair = max(pairs, key=pairs.get)

# 替换原始数据
updated_text = merge_pair(text_list, most_frequent_pair)

print("原始数据:", text)
print("最频繁的二元组:", most_frequent_pair)
print("更新后的数据:", ''.join(updated_text))
