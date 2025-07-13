from collections import OrderedDict 
from collections import Counter

data_file_name="./data.txt"

def update_vocab(d: dict=OrderedDict(), new_vocab: list=[]) -> tuple:
    if len(d) == 0: # 初始化
        for i in range(256):
            d[i] = [i] 
        return d, None 
    else:  # 存在则找到最大值后新增
        max_id = max(list(d.keys()))
        d[max_id] = new_vocab
        return d, max_id
    

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

def find_max_freq(txt: bytes, gram_list: list[list]) -> list:
    """
    找当当前情况下的最高频率的词
    """
    max_freq = 0
    vocab = []
    for i in gram_list:
        c = txt.count(bytes(i))
        if c>=max_freq: # 如果出现更高的频率
            max_freq = c # 更新当前最高频率  
            vocab = i # 记录这个词
    return vocab 
    
def get_next_id(d: dict):
    return max(list(d.keys())) + 1

txt = read_data(data_file_name)

txt_b = list(txt)
vocab = list(set(txt_b))
vocab.sort() 
vocab_dict, _ = update_vocab()
for i in vocab:
    vocab_dict, _ = update_vocab(d=vocab_dict, new_vocab=[i])

gran_2 = take_gram(txt=txt_b)

new_vocab = find_max_freq(txt=txt, gram_list=gran_2)
vocab_dict, max_id = update_vocab(d=vocab_dict, new_vocab=new_vocab)
print(vocab_dict)

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


# txt = txt.replace(new_vocab,"")  # 如何做提替换？ 
# 使用 int 进行记录， int 为字典的键， 值是原始的 byte 拼接， 这样可以保证反解的时候只遍历一次



