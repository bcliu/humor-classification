import argparse
import csv
import torch
import torch.nn as nn
from torch import optim
import bcolz
import pickle
from models.TextCNN import TextCNN
import numpy as np

HUMOR_TYPES = {
    'INCONG': 0,
    'AMBIG': 1,
    'INTERPERS': 2,
    'PHONETIC': 3,
    'OTHER': 4,
}


def sentence2vec(vocabulary, token_list):
    """
    :param vocabulary: Dictionary that maps word to id
    :param token_list: List of token strings
    :return: List of ids
    """
    vec = []
    for token in token_list:
        if token not in vocabulary:
            vec.append(vocabulary['UNK'])
        else:
            vec.append(vocabulary[token])
    return vec


def load_labels(labels_path, humor_types):
    labels = {}
    with open(labels_path) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            labels[int(row[0])] = humor_types[row[1]]
    return labels


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser()
parser.add_argument('--train', type=str, help='Path to training file')
parser.add_argument('--val', type=str, help='Path to validation file')
parser.add_argument('--labels', type=str, help='Path to labels file')
parser.add_argument('--embedding-vectors', type=str, help='Path to embeddings vector bcolz directory (e.g. GloVe)')
parser.add_argument('--embedding-word2idx', type=str, help='Path to embeddings word2idx file')
parser.add_argument('--batch-size', type=int)
args = parser.parse_args()

embedding_vectors = bcolz.open(args.embedding_vectors)[:]
embedding_dim = len(embedding_vectors[0])
embedding_word2idx = pickle.load(open(args.embedding_word2idx, 'rb'))
# Maps words to embedding vectors
embeddings = {w: embedding_vectors[embedding_word2idx[w]] for w in embedding_word2idx}

vocab = {'PAD': 0, 'UNK': 1}

# Build vocabulary using training set
with open(args.train) as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    for row in csv_reader:
        # row[0] is index of the sentence in the dataset
        for token in row[1].split():
            if token not in vocab:
                vocab[token] = len(vocab)

vocab_size = len(vocab)
print(f'Vocabulary size: {vocab_size}')
# Stores indexes of sentences provided in the original dataset
train_idx = []
train_data_unpadded = []
val_idx = []
val_data_unpadded = []

longest_sentence_length = 0

with open(args.train) as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    for row in csv_reader:
        train_idx.append(int(row[0]))
        sentence_split = row[1].split()
        if len(sentence_split) > longest_sentence_length:
            longest_sentence_length = len(sentence_split)
        train_data_unpadded.append(sentence2vec(vocab, sentence_split))

with open(args.val) as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    for row in csv_reader:
        val_idx.append(int(row[0]))
        sentence_split = row[1].split()
        val_data_unpadded.append(sentence2vec(vocab, sentence_split))


train_data = np.zeros([len(train_data_unpadded), longest_sentence_length])
val_data = np.zeros([len(val_data_unpadded), longest_sentence_length])

for i in range(len(train_data_unpadded)):
    unpadded_row = train_data_unpadded[i]
    train_data[i, :len(unpadded_row)] = unpadded_row

for i in range(len(val_data_unpadded)):
    unpadded_row = val_data_unpadded[i]
    val_data[i, :len(unpadded_row)] = unpadded_row[:min(len(unpadded_row), longest_sentence_length)]

word_weight_matrix = torch.zeros(vocab_size, embedding_dim, device=device)
for word in vocab:
    if word in embeddings:
        word_weight_matrix[vocab[word]] = torch.tensor(embeddings[word])
    else:
        word_weight_matrix[vocab[word]] = torch.rand(embedding_dim)

labels = load_labels(args.labels, HUMOR_TYPES)

textCNN = TextCNN(word_weight_matrix)
