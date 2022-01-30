from pickletools import optimize
from typing import Iterator
import torch
import torch.nn as nn
import Util
import torch.optim as optim
import jsonlines
import Model

device = 'cuda' if torch.cuda.is_available() else 'cpu'

print('Run on ', device)

TrainSourceData = jsonlines.open('MRTrain.jsonl')
TrainSourceData = list(TrainSourceData)

TestSourceData = jsonlines.open('MRTest.jsonl')
TestSourceData = list(TestSourceData)

ReviewDict = Util.Dictionary()
ReviewDict.add_word('<pad>')
ReviewDict.add_word('<unk>')

Util.DictionaryBuilder(TrainSourceData, ReviewDict)

TrainIter = Util.DataGenerator(TrainSourceData, ReviewDict=ReviewDict)
TrainIter = Util.DataLoader(TrainIter, batch_size = 64, shuffle = True, collate_fn = Util.collate_batch)

TestIter = Util.DataGenerator(TestSourceData, ReviewDict=ReviewDict)
TestIter = Util.DataLoader(TestIter, batch_size = 64, shuffle = True, collate_fn = Util.collate_batch)

VocabSize = len(ReviewDict.idx2word)
EmbSize = 256
HiddenSize = 256
Layers = 1
BatchSize = 64
LinearOutputSize = 256

model = Model.GRUEncoder(VocabSize, EmbSize, HiddenSize, LinearOutputSize, Layers, BatchSize).to(device)
criterion = nn.BCELoss().to(device)
optimizer = optim.Adam(model.parameters())


def train_epoch(model: Model, optimizer, iterator, criterion, clip):
    model.train()
    epoch_loss = 0
    for data in iterator:
        optimizer.zero_grad()
        Review = data['Review'].to(device)
        Label = data['Label'].to(device)
        output = model(Review)
        loss = criterion(torch.reshape(output, (-1, )), Label.float())
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    return epoch_loss / len(iterator)


def ValidEvaluate(model: nn.Module,iterator, criterion: nn.Module):
    model.eval()
    epoch_loss = 0

    with torch.no_grad():
        for batch, data in enumerate(iterator):
            Review = data['Review'].to(device)
            Label = data['Label'].to(device)
            output = model(Review)
            loss = criterion(torch.reshape(output, (-1, )), Label.float())
            epoch_loss += loss.item()

    return epoch_loss / len(iterator)

def TestFunction(model: nn.Module,iterator):
    model.eval()
    CorrectPredict = 0
    IncorrectPredict = 0
    with torch.no_grad():
        for batch, data in enumerate(iterator):
            Review = data['Review'].to(device)
            Label = data['Label'].to(device)
            output = model(Review)
            output = torch.round(output)
            for i in range(len(output)):
                if int(output[i]) == Label[i]:
                    CorrectPredict += 1
                else:
                    IncorrectPredict += 1
        return CorrectPredict/len(iterator), IncorrectPredict/len(iterator)
train_epoch(model, optimizer, TrainIter, criterion, clip = 5)
ValidEvaluate(model, TestIter, criterion)
print(TestFunction(model, TestIter))