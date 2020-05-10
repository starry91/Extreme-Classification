import torch
from torch import nn
from torchsummary import summary
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import pandas as pd
from torch import optim
import numpy as np
from torchviz import make_dot
from scipy.io.arff import loadarff
import math
import os
import sys
import matplotlib.pyplot as plt
torch.manual_seed(1)
np.random.seed(1)


class FeatureEmbedding(nn.Module):
    def __init__(self, input_size, embedding_size, attention_layer_size, hidden_layer_size, embedding_weight_matrix):
        super(FeatureEmbedding, self).__init__()
        self.embedding = nn.Embedding(input_size, embedding_size)
        if embedding_weight_matrix is not None:
            self.embedding.load_state_dict({'weight': embedding_weight_matrix})
        # Attention Module
        self.attentionfc1 = nn.Linear(embedding_size,  attention_layer_size)
        nn.init.kaiming_normal_(self.attentionfc1.weight, mode='fan_in')
        self.bn1 = nn.BatchNorm1d(input_size)
        # self.d1 = nn.Dropout(p=0.2)
        self.attentionfc2 = nn.Linear(attention_layer_size, embedding_size)
        nn.init.kaiming_normal_(self.attentionfc2.weight, mode='fan_in')
        self.bn2 = nn.BatchNorm1d(input_size)
        # self.d2 = nn.Dropout(p=0.2)
        self.featurefc1 = nn.Linear(embedding_size, hidden_layer_size)
        nn.init.kaiming_normal_(self.featurefc1.weight, mode='fan_in')

    def forward(self, bow, tfidf):
        word_embedding = self.embedding(bow)
        word_embedding = tfidf * word_embedding
        attention_embedding = word_embedding
        attention_embedding = self.attentionfc1(attention_embedding)
        attention_embedding = self.bn1(attention_embedding)
        attention_embedding = torch.relu(attention_embedding)
        # attention_embedding = self.d1(attention_embedding)
        attention_embedding = self.attentionfc2(attention_embedding)
        attention_embedding = torch.sigmoid(attention_embedding)
        # attention_embedding = self.d2(attention_embedding)
        word_embedding = attention_embedding * word_embedding
        word_embedding = word_embedding.mean(1)
        x_hidden = torch.relu(self.featurefc1(word_embedding))
        return x_hidden


class Encoder(nn.Module):
    def __init__(self, output_size, encoder_layer_size, hidden_layer_size):
        super(Encoder, self).__init__()
        self.encoderfc1 = nn.Linear(output_size, encoder_layer_size)
        nn.init.kaiming_normal_(self.encoderfc1.weight, mode='fan_in')
        self.bn1 = nn.BatchNorm1d(encoder_layer_size)
        # self.d1 = nn.Dropout(p=0.2)
        self.encoderfc2 = nn.Linear(encoder_layer_size, hidden_layer_size)
        nn.init.kaiming_normal_(self.encoderfc2.weight, mode='fan_in')
        # self.bn2 = nn.BatchNorm1d(hidden_layer_size)
        # self.d2 = nn.Dropout(p=0.2)

    def forward(self, labels):
        # print("Encoder labels size: {0}".format(labels.size()))
        y_hidden = self.encoderfc1(labels)
        y_hidden = self.bn1(y_hidden)
        y_hidden = torch.relu(y_hidden)
        # y_hidden = self.d1(y_hidden)
        y_hidden = self.encoderfc2(y_hidden)
        y_hidden = torch.sigmoid(y_hidden)
        # y_hidden = self.d2(y_hidden)
        return y_hidden


class Decoder(nn.Module):
    def __init__(self, output_size, encoder_layer_size, hidden_layer_size):
        super(Decoder, self).__init__()
        self.decoderfc1 = nn.Linear(hidden_layer_size, encoder_layer_size)
        nn.init.kaiming_normal_(self.decoderfc1.weight, mode='fan_in')
        self.bn1 = nn.BatchNorm1d(encoder_layer_size)
        # self.d1 = nn.Dropout(p=0.2)
        self.decoderfc2 = nn.Linear(encoder_layer_size, output_size)
        nn.init.kaiming_normal_(self.decoderfc2.weight, mode='fan_in')
        # self.bn2 = nn.BatchNorm1d(output_size)

    def forward(self, y_hidden):
        y_predicted = self.decoderfc1(y_hidden)
        y_predicted = self.bn1(y_predicted)
        y_predicted = torch.relu(y_predicted)
        # y_predicted = self.d1(y_predicted)
        y_predicted = self.decoderfc2(y_predicted)
        # y_predicted = torch.sigmoid(y_predicted)
        # y_predicted = torch.relu(y_predicted)
        return y_predicted


class AttentionModel(nn.Module):
    def __init__(self, input_size, embedding_size, attention_layer_size, encoder_layer_size, hidden_layer_size, output_size, embedding_weight_matrix=None):
        super(AttentionModel, self).__init__()
        self.featureEmbedding = FeatureEmbedding(
            input_size, embedding_size, attention_layer_size, hidden_layer_size, embedding_weight_matrix)
        self.encoder = Encoder(
            output_size, encoder_layer_size, hidden_layer_size)
        self.decoder = Decoder(
            output_size, encoder_layer_size, hidden_layer_size)

    def forward(self, bow, tfidf, labels):
        x_hidden = self.featureEmbedding(bow, tfidf)
        y_hidden = self.encoder(labels)
        y_predicted = self.decoder(y_hidden)
        return x_hidden, y_hidden, y_predicted

    def predict(self, bow, tfidf):
        x_hidden = self.featureEmbedding(bow, tfidf)
        y_predicted = self.decoder(x_hidden)
        return y_predicted

    def get_values(self, bow, tfidf):
        x_hidden = self.featureEmbedding(bow, tfidf)
        y_predicted = self.decoder(x_hidden)
        return x_hidden, y_predicted
