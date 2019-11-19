import csv
import numpy as np
import torch


def create_vocabulary(path):
    """
    :param path: Path to training CSV file containing index and space-separated token list
    :return: Dictionary that maps vocabulary words to indices
    """
    vocab = {'PAD': 0, 'UNK': 1}
    with open(path) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            # row[0] is index of the sentence in the dataset
            for token in row[1].split():
                if token not in vocab:
                    vocab[token.lower()] = len(vocab)
    return vocab


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


def load_unpadded_train_val_data(path, vocabulary):
    indices = []
    unpadded_data = []
    longest_sentence_length = 0

    with open(path) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            indices.append(int(row[0]))
            sentence_split = row[1].split()
            if len(sentence_split) > longest_sentence_length:
                longest_sentence_length = len(sentence_split)
            unpadded_data.append(sentence2vec(vocabulary, sentence_split))
    return indices, unpadded_data, longest_sentence_length


def create_padded_data(unpadded_data, longest_length):
    padded_data = np.zeros([len(unpadded_data), longest_length])
    for i in range(len(unpadded_data)):
        unpadded_row = unpadded_data[i]
        padded_data[i, :len(unpadded_row)] = unpadded_row[:min(len(unpadded_row), longest_length)]
    return padded_data


def create_weight_matrix(vocabulary, embeddings, device):
    embedding_dim = len(list(embeddings.items())[0][1])
    matrix = torch.zeros(len(vocabulary), embedding_dim, device=device)
    for word in vocabulary:
        if word in embeddings:
            matrix[vocabulary[word]] = torch.tensor(embeddings[word])
        else:
            matrix[vocabulary[word]] = torch.rand(embedding_dim)
    return matrix
