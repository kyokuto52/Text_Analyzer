from tensorflow.keras.preprocessing.text import text_to_word_sequence
import re
from collections import Counter
import pickle
from torch.utils.data import DataLoader, Dataset
import torch
from torch.nn.utils.rnn import pad_sequence
import random

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

def SaveDictClassFile(DictObject, FilePath):
    DictFp = open(FilePath, 'wb')
    pickle.dump(DictObject, DictFp)
    DictFp.close()

def isEnglish(s):
    try:
        s.encode(encoding='utf-8').decode('ascii')
    except UnicodeDecodeError:
        return False
    else:
        return True

def CamelSplit(Str):
    matches = re.finditer('.+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)', Str)
    return [m.group(0) for m in matches]

def ReplaceDigitAndSynx(MethodName: str):
    MethodName = re.sub(r"([!\"#$%&()*+,-./:;<=>?@[\]^_`{|}~])", r" \1 ", MethodName)
    MethodName = re.sub(r'\b\d+\b',' num_ ' , MethodName)
    return ' '.join(MethodName.split())

def CommentProcessor(CommentString: str):
    if len(CommentString) == 0:
        return False
    if CommentString[-1] != '.':
        CommentString = CommentString + '.'
    if not isEnglish(CommentString):
        return False
    if len(re.findall(r'(https?|ftp|file)://[-A-Za-z0-9+&@#/%?=~_|!:,.;]+[-A-Za-z0-9+&@#/%=~_|]',CommentString)) > 0:
        return False
    Comment = re.sub(r'</?[^>]+>', '', CommentString)
    Comment = re.sub(r"([!\"#$%&()*+,-./:;<=>?@[\]^_`{|}~])", r" \1 ", Comment)
    Comment = re.sub(r'@ (.+?) ', '',Comment)
    Comment = re.sub(r'\d+', 'NNUUMM', Comment)
    Comment = re.sub(r'".+"', ' SSTTRR ', Comment)

    Comment = text_to_word_sequence(Comment, filters='"#$%&()*+-/:;<=>@[\\]^_`{|}~\t\n', lower = False)
    Comment = ' '.join(Comment)
    Comment = Comment.replace('NNUUMM', ' num_ ')
    Comment = Comment.replace('SSTTRR', 'str_')
    Comment = Comment.lower().split()
    if len(Comment) > 60 or len(Comment) <= 3:
        # print(CommentString)
        return False
    else:
        return Comment

def CodeProcessor(NodeList:list):
    TargetList = []
    for NodeStr in NodeList:
        NodeStr = NodeStr.strip()
        if NodeStr in PlaceHolderList:
            TargetList.append(NodeStr.lower())
            continue
        NodeStr = re.sub(r"([!\"#$%&()*+,-./:;<=>?@[\]^_`{|}~])", r" \1 ", NodeStr)
        SplitedNode = ' '.join(CamelSplit(NodeStr)).lower()
        TargetList.append(SplitedNode)
        if len(TargetList) > 1000:
            return False
    # if TrainPart == True:
    #     CodeDict = Dictionary()
    #     CodeDict.add_word('<UNK>')
    #     for i in TargetList:
    #         for word in i.split():
    #             CodeDict.add_word(word)
    #     print(CodeDict.word2idx)    #保存单词表对象。
    return TargetList

def EdgeProcessor(EdgeList: list):
    SourceList = []
    TargetList = []
    for i in EdgeList:
        SourceList.append(i[0])
        TargetList.append(i[-1])
    return ' '.join(SourceList) + ' <SPL> ' + ' '.join(TargetList)
    

class DataGenerator(Dataset):
    def __init__(self, Iter, CodeDict: dict, DocDict: dict, Multiple: float, IsKG = False):
        self.samples = random.sample(Iter, int(len(Iter) * Multiple))
        self.CodeDict = CodeDict
        self.DocDict = DocDict
        self.IsKG = IsKG
        print('Sum Samples: ', len(self.samples))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        if self.IsKG:
            Nodes, NodeLengths = self.NodesTransformer(self.samples[idx]['KGNodes'])
            SelfEdge = self.SelfEdge(len(Nodes))
            Edges = self.EdgeTransformer(self.samples[idx]['KGEdges'], SelfEdge)
        else:
            Nodes, NodeLengths = self.NodesTransformer(self.samples[idx]['Nodes'])     #Size = [NodeNum, MaxNodeLen]
            SelfEdge = self.SelfEdge(len(Nodes))
            Edges = self.EdgeTransformer(self.samples[idx]['Edges'], SelfEdge)
        Doc = self.DocTransformer(self.samples[idx]['Doc'])
        return {'Nodes': Nodes, 'Edges': Edges, 'Doc': Doc, 'NodeLengths': NodeLengths}

    def NodesTransformer(self, NodeList):
        TargetList = []
        Length = []
        for i in NodeList.split(' <SPL> '):
            Buffer = []
            Words = i.split()
            Length.append(len(Words))
            for word in Words:
                Buffer.append(self.CodeDict.get(word, self.CodeDict.get('<unk>')))
            TargetList.append(torch.tensor(Buffer))
        return pad_sequence(TargetList, batch_first=True, padding_value=self.CodeDict.get('<pad>')), torch.tensor(Length)
    def EdgeTransformer(self, EdgeList, SelfEdge):
        Source, Target = EdgeList.split(' <SPL> ')
        # print(Source)
        # print(Target)
        return torch.tensor([[int(i) for i in Source.split()+ SelfEdge], [int(j) for j in Target.split()+ SelfEdge]])
    def DocTransformer(self, Doc):
        TargetTensor = []
        Doc = '<sos> ' + Doc + ' <eos>'
        for word in Doc.split():
            TargetTensor.append(self.DocDict.get(word, self.DocDict.get('<unk>')))
        return torch.tensor(TargetTensor)
    
    def SelfEdge(self, NodesNum):
        selfEdge = [i for i in range(NodesNum)]
        return selfEdge


class GRUDataGenerator(Dataset):
    def __init__(self, Iter, CodeDict: dict, DocDict: dict):
        self.samples = Iter
        self.CodeDict = CodeDict
        self.DocDict = DocDict

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]['RawCodes'].split()     #Size = [NodeNum, MaxNodeLen]
        # return {'Nodes': Nodes, 'Edges': Edges, 'Doc': Doc, 'NodeLengths': NodeLengths}



class DataPreprocess():
    def __init__(self, DataPath, LabelPath) -> None:
        self.DataPath = DataPath
        self.LabelPath = LabelPath
        self.ReviewDict = Dictionary()
        pass

# def DataToJSONL(SampleString: str, Label: str):
#     Fp = open(SampleString)
#     SampleHandle = Fp.read().split('\n')
#     Fp2 = open(Label)
#     LabelHandle = Fp2.read().split('\n')

