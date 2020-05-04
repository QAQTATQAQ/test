# -*- coding: utf-8 -*-
"""
Created on Mon Feb 24 21:10:32 2020

@author: M
"""

import torch
# from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as tud

from collections import Counter
import numpy as np
import random
import math

import pandas as pd
import scipy
import sklearn
from sklearn.metrics.pairwise import cosine_similarity

USE_CUDA = torch.cuda.is_available()

random.seed(1)
np.random.seed(1)
torch.manual_seed(1)
if USE_CUDA:
    torch.cuda.manual_seed(1)

C = 3
K = 100
NUM_EPOCHS = 2
MAX_VOCAB_SIZE = 30000
BATCH_SIZE = 128
LEARNING_RATE = 0.2
EMBEDDING_SIZE = 100

def word_tokensize(text):
    return text.split()

with open('','r') as fin:
    text = fin.read()
    
text = text.split()
vocab = dict(Counter(text).most_common(MAX_VOCAB_SIZE - 1))
vocab["<unk>"] = len(text) - np.sum(list(vocab.values()))
    
idx_to_word =  [word for word in vocab.keys()]
word_to_idx = {word:i for word in enumerate(idx_to_word)}

word_counts = np.array([count for count in vocab.values()],dtype=np.float32)
word_freqs = word_counts / np.sum(word_counts)
word_freqs = word_freqs ** (3./4.)
word_freqs = word_freqs / np.sum(word_freqs)
VOCAB_SIZE = len(idx_to_word)

class WordEmbeddingDataset(tud.Dataset):
    def __init__(self,text,word_to_idx,idx_to_word,word_freqs,word_counts):
        super(WordEmbeddingDataset,self).__init__()
        self.text_encoded = [word_to_idx.get(word,word_to_idx["<unk>"]) for word in text ]
        self.text_encoded = torch.LongTensor(self.text_encoded)
        self.word_to_idx = word_to_idx
        self.idx_to_word = idx_to_word
        self.word_freqs = torch.Tensor(word_freqs)
        self.word_counts = torch.Tensor(word_counts)
        
    def __len__():
        return len(self.text_encoded)
       
    def __getitem__(self,idx):
        center_word = self.text_encoded[idx]
        pos_indices = list(range(idx-C,idx)) + list(range(idx+1,idx+C+1))
        pos_indices = [i % len(self.text_encoded) for i in pos_indices]
        pos_words = self.text_encoded[pos_indices]
        neg_words = torch.multinomial(self.word_freqs, K*pos_words.shape[0],True)
        return center_word,pos_words,neg_words
        
dataset = WordEmbeddingDataset(text,word_to_idx,idx_to_word,word_freqs,word_counts)
dataloader = tud.DataLoader(dataset,batch_size = BATCH_SIZE,shuffle=True,num_workers=4)
      
class EmbeddingModel(nn.Module):
    def __init__(self,vocab_size,embed_size):
        super( EmbeddingModel, self).__init__()
        
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        
        self.in_embed = nn.Embedding(self.vocab_size,self.embed_size)
        self.out_embed = nn.Embedding(self.vocab_size,self.embed_size)
        
    def forward(self,input_labels,pos_labels,neg_labels):
        input_embedding = self.in_embed(input_labels)
        pos_embedding = self.in_embed(pos_labels)
        neg_embedding = self.in_embed(neg_labels)
        
        input_embedding = input_embedding.unsqueeze(2)
        pos_dot = torch.bmm(pos_embedding,input_embedding).squeeze(2)
        neg_dot = torch.bmm(neg_embedding,-input_embedding).squeeze(2)
        
        log_pos = F.logsigmoid(pos_dot)
        log_neg = F.logsigmoid(neg_dot).sum(1)
        
        loss = log_neg + log_pos
        return -loss

    def input_        
         