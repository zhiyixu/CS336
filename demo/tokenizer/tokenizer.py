from collections import OrderedDict
from datetime import datetime
import numpy as np
from typing import Optional, Union
import os, sys, time, json
import itertools


class Token():

    def __init__(self, vocab: Union[list[int], list[list[int]]], idx: int):
        self.vocab = vocab
        self.idx = idx


class Tokenizer():

    def __init__(self, vocab_size: int=512):
        self.vocab_dict = OrderedDict()
        self.vocab_size = vocab_size
        self.txt: np.ndarray
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
        self.txt = np.array(list(databytes), dtype='int')

    def take_gram(self) -> list[list]:
        """将字节序列转换为二元列表"""
        return list(map(list,zip(self.txt[:-1].tolist(), self.txt[1:].tolist())))

    def find_max_freq(self, gram_list: list[list[int]]) -> list:
        """找到当前内容中频率最高的词"""
        max_freq = 0
        vocab = []
        np_txt = np.array(self.txt, dtype=int)  # 这里需要动态生成， 因为txt会被合并
        for gram in gram_list:
            first_matches = np_txt[:-1] == gram[0]
            last_matches = np_txt[1:] == gram[1]
            cnt = np.count_nonzero(first_matches & last_matches)
            if cnt >= max_freq:  # 如果出现更高的频率
                max_freq = cnt  # 更新当前最高频率
                vocab = gram  # 记录这个词
        return vocab

    def update_txt(self, new_vocab: list[list], max_id: int):
        old_txt_b = self.txt.copy()
        new_txt_b = []
        first_matches = self.txt[:-1] == new_vocab[0]
        last_matches = self.txt[1:] == new_vocab[1]
        match_flag = np.array(first_matches & last_matches)  # -> matched flag, is this index matched with the vocab
        match_index = np.arange(len(match_flag))[match_flag][::-1]  # -> convent the match flag to start index
        for idx in match_index:
            new_txt_b.append(old_txt_b[idx+2:])
            new_txt_b.append(np.array([max_id]))
            old_txt_b = old_txt_b[:idx]
        new_txt_b.append(old_txt_b)
        self.txt = np.hstack(new_txt_b[::-1])


    def update_vocab(self, new_vocab: list) -> int:
        max_id = len(self.vocab_dict)
        if new_vocab not in list(self.vocab_dict.values()):
            self.vocab_dict[max_id] = list(itertools.chain.from_iterable([self.vocab_dict[i] for i in new_vocab]))
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
            json.dump(self.vocab_dict, f, ensure_ascii=True, indent=4)
        print(f"{datetime.now()} Tokenizer File Save To: {self.tokenizer_file}")


    def encode(self, raw_txt: str) -> list:
        contents = np.array(list(raw_txt.encode("utf-8")))
        print(f"encode contents: {contents}")
        for idx, vocab in list(self.vocab_dict.items()):
            if len(vocab) == 1:
                continue
            match_flag = contents==vocab[0]
            if sum(match_flag) == 0:
                continue
            ll = []
            # pdb.set_trace()
            start_index = np.arange(len(match_flag))[match_flag].tolist()[::-1]  # reverse order
            for index in start_index:
                if contents[index:index+len(vocab)].tolist() == vocab:  # match
                    ll.append(contents[index+len(vocab):].tolist())  # first add the rest of data
                    ll.append([idx])  # merge the matched vocab
                else:
                    ll.append(contents[index:].tolist())  # if not match , add them all
                contents = contents[:index]
            ll.append(contents.tolist())
            contents = np.array(list(itertools.chain.from_iterable(ll[::-1])))  # 更新内容
        encode_idx = list(contents)
        return encode_idx

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
    text_txt = """2008年上半年，奥运场馆测试赛陆续进行，包括手球国际邀请赛、布北京2008年奥运会火炬接力开始。4月2日，北京奥运会火炬接力第一站传递活动在哈萨克斯坦阿拉木图举行。5月4日，奥运圣火从中国"""
    idx_list = tokenizer.encode(raw_txt=text_txt)
    print(f"encode idx: {idx_list}")
    print(f"len raw_txt: {len(text_txt)}, len raw bytes: {len(tokenizer.txt)}, len encode idx: {len(idx_list)}")
    decoded_txt = tokenizer.decode(idx_list=idx_list)
    print(f"Decoded contents: {decoded_txt}")
