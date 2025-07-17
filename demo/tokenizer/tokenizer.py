from collections import OrderedDict
from datetime import datetime
import numpy as np
from typing import Optional, Union
import os, sys, time, json

class Token():

    def __init__(self, vocab: Union[list[int], list[list[int]]], idx: int):
        self.vocab = vocab
        self.idx = idx


class Tokenizer():

    def __init__(self, vocab_size: int=512):
        self.vocab_dict = OrderedDict()
        self.vocab_size = vocab_size
        self.txt: list = []
        self.tokenizer_file = "./tokenizer.json"

        # methods
        self._init_vocab_dict()

    def _init_vocab_dict(self):
        """初始化词表，添加基础词"""
        for i in range(256):
            self.vocab_dict[i] = [i]



    def _read_data(self, fname: str):
        if not os.path.exists(fname):
            print(f"{datetime.now()} File: {fname} Not Exists!")
            sys.exit(-1)
        if not fname.endswith(".txt"):
            print(f"{datetime.now()} Current Support .txt Only.")
            sys.exit(-1)
        with open(fname, "r", encoding="utf-8") as f:
            contents = f.read()
        return contents.encode("utf-8")

    def set_contents(self, databytes: bytes=b''):
        self.txt = np.array(databytes, dtype='uint8')

    def take_gram(self) -> list[tuple]:
        """将字节序列转换为二元列表"""
        return list(zip(self.txt[:-1], self.txt[1:]))

    def find_max_freq(self, gram_list: list[tuple[int]]) -> list:
        """找到当前内容中频率最高的词"""
        max_freq = 0
        vocab: list[int] = []
        np_txt = np.array(self.txt, dtype=int)  # 这里需要动态生成， 因为txt会被合并
        for gram in gram_list:
            first_matches = np_txt[:-1] == gram[0]
            last_matches = np_txt[1:] == gram[1]
            cnt = np.count_nonzero(first_matches & last_matches)
            if cnt >= max_freq:  # 如果出现更高的频率
                max_freq = cnt  # 更新当前最高频率
                vocab = list(gram)  # 记录这个词
        return vocab

    def update_txt(self, new_vocab: list[list], max_id: int):
        new_txt_b = []
        sindex = 0
        while 1:
            if sindex < len(self.txt) - 1:  # 还没到最后交界点
                if [self.txt[sindex], self.txt[sindex + 1]] == new_vocab:  # 需要合并的词
                    new_txt_b.append(max_id)
                    sindex += 2
                else:
                    new_txt_b.append(self.txt[sindex])
                    sindex += 1
            elif sindex == len(self.txt) - 1:  # 刚好是最后一个
                new_txt_b.append(self.txt[sindex])
                sindex += 1
            else:  # 没了
                break
        self.txt = new_txt_b # update

    def update_txt_new(self, new_vocab: list[list], max_id: int):
        new_txt_b = []
        sindex = 0
        first_matches = self.txt[:-1] == new_vocab[0]
        last_matches = self.txt[1:] == new_vocab[1]
        match_flag = (first_matches & last_matches)  # -> matched flag, is this index matched with the vocab
        match_index = list(range(len(match_flag))[match_flag])  # -> convent the match flag to start index
        if match_index[0] != 0:  # the first one not vocab
            new_txt_b.append(self.txt[:match_index[0]])
        for i in match_index:
            next_i = i + 2
            new_txt_b.append(np.array([max_id])) # inplace
            # from the very end to the start, in this way, replace not effects exists index for the value

        new_txt_b = np.hstack((self.txt[:i], [max_id], self.txt[i+2:]) for i in match_index)  # -> this data should be remove from txt list,
        self.txt = new_txt_b # update

    def update_vocab(self, new_vocab: []) -> int:
        max_id = len(self.vocab_dict)
        if new_vocab not in list(self.vocab_dict.values()):
            self.vocab_dict[max_id] = new_vocab
        return max_id

    def train(self):
        print(f"{datetime.now()} Start Train Tokenizer....")
        t0 = time.time()
        while len(self.vocab_dict) < self.vocab_size:
            gran_2 = self.take_gram()
            new_vocab = self.find_max_freq(gram_list=gran_2)
            max_id = self.update_vocab(new_vocab=new_vocab)
            self.update_txt(new_vocab=new_vocab, max_id=max_id)
            print(f"{datetime.now()} current vocab size: {len(self.vocab_dict)}, new id: {max_id}")
        t1 = time.time()
        print(f"{datetime.now()} Tokenizer Training Finished, Cost: {t1-t0:.2f}")
        with open(self.tokenizer_file,"w") as f:
            json.dump(self.vocab_dict, f, indent=4, ensure_ascii=False)
        print(f"{datetime.now()} Tokenizer File Save To: {self.tokenizer_file}")

    def encode(self, raw_txt: str):
        contents = list(raw_txt.encode())
        for idx, vocab in list(self.vocab_dict.items())[::-1]:
            ll = []
            start_index = 0
            while 1:
                if start_index < len(contents) - 1:
                    if [contents[start_index], contents[start_index + 1]] == vocab:
                        ll.append(idx)
                        start_index += 2
                    else:
                        ll.append(contents[start_index])
                        start_index += 1
                elif start_index == len(contents) - 1:
                    ll.append(contents[start_index])
                    start_index += 1
                else:
                    break
            contents = ll  # 更新内容
        return contents

    def decode(self, idx_list: list):
        ll = []
        for idx in idx_list:
            ll += self.vocab_dict[idx]
        return bytes(ll).decode()


if __name__ == "__main__":
    data = "./data.txt"
    tokenizer = Tokenizer()
    data_b = tokenizer._read_data(fname=data)
    tokenizer.set_contents(databytes=data_b)
    tokenizer.train()
    text_txt = "北京奥运会志愿者标志发布，志愿者项目正式启动。6月26日，北京奥组委宣布第29届奥运会主题口号：“同一个世界，同一个梦想”（One World，One Dream）。11月11日，第29届奥运会吉祥物在北京公布"
    idx_list = tokenizer.encode(raw_txt=text_txt)
    print(f"len raw_txt: {len(text_txt)}, len raw bytes: {len(tokenizer.txt)}, len encode idx: {len(idx_list)}")
    decoded_txt = tokenizer.decode(idx_list=idx_list)
    print(f"Decoded contents: {decoded_txt}")
