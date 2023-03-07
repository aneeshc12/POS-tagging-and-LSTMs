import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from copy import copy
from datetime import datetime
import matplotlib.pyplot as plt
import pickle

class LSTM(nn.Module):
    def __init__(self, embeddingDim, hiddenDim, wordVocab, tagVocab, device):
        super(LSTM, self).__init__()


        self.hiddenDim = hiddenDim
        self.padIdx = 1
        self.wordEmbeddings = nn.Embedding(len(wordVocab), embeddingDim, 
                                        padding_idx=self.padIdx)         # encode each word as an embedding of size embeddingDim

        with torch.no_grad():
            self.wordEmbeddings.weight[self.padIdx,:] = 1.0  

            self.lstm = nn.LSTM(embeddingDim, hiddenDim, batch_first=True)      # main lstm 
            self.fc1 = nn.Linear(hiddenDim, hiddenDim)
            self.fc2 = nn.Linear(hiddenDim, len(tagVocab))                       # output a tag based on the hidden dim (using the internal state)

            self.logSoftMax = F.log_softmax

            self.device = device

    def forward(self, sentences):
        s = sentences.to(self.device)         # move to cuda
        embeddings = self.wordEmbeddings(s)
        out, _ = self.lstm(embeddings)
        out = self.fc1(out)
        out = self.fc2(out)

        tagScores = self.logSoftMax(out, dim=1)
        return tagScores



with open("./weights/word2idx.pkl", 'rb') as f:
    word2idx = pickle.load(f)
with open("./weights/tag2idx.pkl", 'rb') as f:
    tag2idx = pickle.load(f)
with open("./weights/idx2tag.pkl", 'rb') as f:
    idx2tag = pickle.load(f)

def encodeSequence(seq, toIdx):
  encoded = []
  for word in seq:
    if word not in toIdx:
      encoded.append(toIdx["_unk"])
    else:
      encoded.append(toIdx[word])
  encoded = torch.FloatTensor(encoded)
  return encoded

def inference(model, sentence):
  """
  Given a model and a sentence, print predicted POS tags

  Return a tensor of tag indices
  """
  with torch.no_grad():
    processedSentence = sentence.lower().split(' ')
    encodedSentence = encodeSequence(processedSentence, word2idx).long()
    results = model(encodedSentence).to('cpu')

    predTags = []
    for r in results:
      predTags.append(int(np.argmax(r)))

    words = sentence.split(' ')
    for w, t in zip(words, predTags):
      print("%s\t%s" % (w, idx2tag[t]))

  return predTags


myLSTM = LSTM(100,150, word2idx, tag2idx, 'cpu')
myLSTM.load_state_dict(torch.load("./weights/final_weights.pt"))

inpSentence = input()
inference(myLSTM, inpSentence)
