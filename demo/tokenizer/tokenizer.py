from collections import OrderedDict 
from collections import Counter
from datetime import datetime
import pdb


data_file_name="./data.txt"

def kmp_search(main_list, sublist):
    def compute_lps(pattern):
        lps = [0] * len(pattern)
        length = 0
        i = 1
        while i < len(pattern):
            if pattern[i] == pattern[length]:
                length += 1
                lps[i] = length
                i += 1
            else:
                if length != 0:
                    length = lps[length - 1]
                else:
                    lps[i] = 0
                    i += 1
        return lps

    lps = compute_lps(sublist)
    count = 0
    i = j = 0

    while i < len(main_list):
        if sublist[j] == main_list[i]:
            i += 1
            j += 1

        if j == len(sublist):
            count += 1
            j = lps[j - 1]
        elif i < len(main_list) and sublist[j] != main_list[i]:
            if j != 0:
                j = lps[j - 1]
            else:
                i += 1

    return count




def update_vocab(d: dict=OrderedDict(), new_vocab: list=[]) -> tuple:
    if len(d) == 0: # 初始化
        for i in range(256):
            d[i] = [i] 
        return d, None 
    else:  # 存在则找到最大值后新增
        if new_vocab not in list(d.values()):
            max_id = len(d)
            d[max_id] = new_vocab
            return d, max_id 
        else:
            return d, len(d)
    

def read_data(file_name):
    """
    以二进制的方式读取数据
    """
    with open(file_name, 'rb') as file:
        data = file.read()
    return data

def take_gram(txt: list[int])-> list[list]: # 返回嵌套列表
    """
    将字节序列转换为二元组
    """
    return [[txt[i], txt[i+1]] for i in range(0,len(txt)-1)]

def find_max_freq(txt: list[int], gram_list: list[list]) -> list:
    """
    找当当前情况下的最高频率的词
    """
    max_freq = 0
    vocab: list[int] = []
    for gram in gram_list:
        c = 0
        for i in range(0, len(txt)-1):
            if [txt[i], txt[i+1]] == gram:
                c+=1
        # c = kmp_search(txt, gram)
        if c>=max_freq: # 如果出现更高的频率
            max_freq = c # 更新当前最高频率  
            vocab = gram # 记录这个词
    return vocab 
    
def get_next_id(d: dict):
    """为新找到的vocab分配一个id"""
    return max(list(d.keys())) + 1

def update_txt(txt_b: list, new_vocab: list[list], max_id: int):
    """用新找到的vocab来替换原始的数据"""
    new_txt_b = []
    sindex = 0
    while 1:
        if sindex < len(txt_b)-1:  # 还没到最后交界点
            if [txt_b[sindex], txt_b[sindex+1]] == new_vocab:  # 需要合并的词
                new_txt_b.append(max_id)
                sindex += 2
            else:
                new_txt_b.append(txt_b[sindex])
                sindex += 1
        elif sindex == len(txt_b)-1:  # 刚好是最后一个
            new_txt_b.append(txt_b[sindex])
            sindex += 1
        else: # 没了
            break
    return new_txt_b



txt = read_data(data_file_name)

txt_b = list(txt)
vocab = list(set(txt_b))
vocab.sort() 
vocab_dict, _ = update_vocab()
for i in vocab:
    vocab_dict, _ = update_vocab(d=vocab_dict, new_vocab=[i])

while len(vocab_dict) < 512:
    # pdb.set_trace()
    gran_2 = take_gram(txt=txt_b)
    new_vocab = find_max_freq(txt=txt_b, gram_list=gran_2)
    vocab_dict, max_id = update_vocab(d=vocab_dict, new_vocab=new_vocab)
    txt_b = update_txt(txt_b=txt_b, new_vocab=new_vocab, max_id=max_id)
    print(f"{datetime.now()} current vocab size: {len(vocab_dict)}, new id: {max_id}")

pdb.set_trace()
print(1)
print(1)
print(1)
print(1)
print(1)