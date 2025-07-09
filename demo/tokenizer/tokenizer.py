from collections import OrderedDict 
from collections import Counter

data_file_name="./data.txt"

def read_data(file_name):
    """
    以二进制的方式读取数据
    """
    with open(file_name, 'rb') as file:
        data = file.read()
    return data

def take_gram(txt: bytes):
    """
    将字节序列转换为二元组
    """
    return [bytes([txt[i], txt[i+1]]) for i in range(0,len(txt)-1)]

def find_max_freq(txt: bytes, gram_list: list):
    """
    找当当前情况下的最高频率的词
    """
    max_freq = 0
    vocab = None
    for i in gram_list:
        c = txt.count(i)
        if c>=max_freq: # 如果出现更高的频率
            max_freq = c # 更新当前最高频率  
            vocab = i # 记录这个词
    return vocab 
    


txt = read_data(data_file_name)

txt_b = list(txt)
vocab = list(set(txt_b))
vocab.sort() 
vocab_dict = OrderedDict()
for i in vocab:
    vocab_dict[i] = bytes([i])

max_id = 257
# 构造二元组 
gran_2 = take_gram(txt=txt)

new_vocab = find_max_freq(txt=txt, gram_list=gran_2)
vocab_dict[max_id] = new_vocab 
print(vocab_dict)
# txt = txt.replace(new_vocab,"")  # 如何做提替换？ 




