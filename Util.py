from os import close
from tkinter import Label
from xmlrpc.client import boolean
import torch
import torch.nn as nn
import jsonlines
from string import punctuation
from collections import Counter
from torch.utils.data import Dataset, DataLoader

class Dictionary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []
        self.counter = Counter()
        self.total = 0

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        token_id = self.word2idx[word]
        self.counter[token_id] += 1
        self.total += 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)




class DataGenerator(Dataset):
    def __init__(self, Iter, ReviewDict: Dictionary = None):
        self.samples = Iter
        if ReviewDict:
            self.ReviewDict = ReviewDict
        # self.DocDict = Dictionary()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        Review = self.ReviewTransformer(self.samples[idx]['Sample'])     #Size = [NodeNum, MaxNodeLen]
        Label = self.LabelTransformer(self.samples[idx]['Label'])
        return Review, Label
    
    def ReviewTransformer(self, String: str):
        Idx = []
        for word in String.split():
            Idx.append(self.ReviewDict.word2idx.get(word, self.ReviewDict.word2idx.get('<unk>')))
        return torch.tensor(Idx)
    def LabelTransformer(self, String: str):
        return torch.tensor(int(String))

def collate_batch(batch):
    TextList = []
    LabelList = []
    for i in batch:
        TextList.append(i[0])
        LabelList.append(i[1])
    TextList = torch.nn.utils.rnn.pad_sequence(TextList, batch_first = True, padding_value=0)
    LabelList = torch.stack(LabelList)
    if len(TextList) != len(LabelList):
        raise ValueError
    return {'Review': TextList, 'Label': LabelList}

def DictionaryBuilder(SamplesIter: list, DictObject: Dictionary):
        for i in SamplesIter:
            for word in i['Sample'].split():
                DictObject.add_word(word)
        return DictObject