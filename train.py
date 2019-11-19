import argparse
import csv
import torch
import torch.nn as nn
from torch import optim
import bcolz
import pickle
from models.TextCNN import TextCNN

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
embedding_word2idx = pickle.load(open(args.embedding_word2idx, 'rb'))
# Maps words to embedding vectors
embeddings = {w: embedding_vectors[embedding_word2idx[w]] for w in embedding_word2idx}

vocab = {'PAD': 0, 'UNK': 1}
word_weight_matrix = []

for file_path in [args.train, args.val]:
    with open(file_path) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            # row[0] is index of the sentence in the dataset
            for token in row[1].split():
                if token not in vocab:
                    vocab[token] = len(vocab)

print(len(vocab))
num = 0
for token in vocab:
    print(f'token: {token}, idx: {vocab[token]}')
    num += 1
    if num == 100:
        break

textCNN = TextCNN(word_weight_matrix)