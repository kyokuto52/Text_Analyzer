from os import close
import torch
import torch.nn as nn
import jsonlines

class EncoderRNN(nn.Module):
    def __init__(self, VocabSize, EmbeddingSize, hidden_size, BatchSize):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.BatchSize = BatchSize
        self.VocabSize = VocabSize
        self.EmbeddingSize = EmbeddingSize
        self.num_layers = 1
        self.embedding = nn.Embedding(VocabSize + 1 , hidden_size)
        self.gru = nn.GRU(input_size = hidden_size,hidden_size = hidden_size, batch_first = True,num_layers=self.num_layers, bidirectional = True)
        self.Dense = nn.Linear(hidden_size * 2, self.VocabSize)
    
    def forward(self, data, hidden):
        x = self.embedding(data)
        x, hidden = self.gru(x, hidden)
        print()
        x = self.Dense(x)
        return x, hidden
    def Init(self):
        return torch.zeros((2, self.BatchSize, self.hidden_size ))


class Corpus(object):
    def __init__(self, path):    #path requied dict path. 
        import pickle
        Fp = open('/Users/jthong/Desktop/GraphsGeneration/LowCodeVocab.pkl', 'rb')
        self.CodeDic = pickle.load(Fp)
        Fp = Fp.close()
        # print(self.CodeDic.word2idx.get('<eos>'))
    def DateGenerator(self, FilePath):
        JsonFP = jsonlines.open(FilePath)
        IdsSeq = []
        for i in JsonFP:
            Sample = i['RawCodes'].split() + ['<eos>']
            ids = []
            if Sample == '':
                continue
            for word in Sample:
                ids.append(self.CodeDic.word2idx.get(word, self.CodeDic.word2idx.get('<unk>')))
            # print(IdsSeq)
            IdsSeq.append(torch.tensor(ids).type(torch.int64))
        return torch.cat(IdsSeq)

# corpus = Corpus('da')
# a = corpus.DateGenerator('/Users/jthong/Desktop/GraphsGeneration/DataSet/Test/LowTest.jsonl')
# print(a.shape)