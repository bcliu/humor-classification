import csv
from typing import Dict, List

import numpy as np
import torch


def create_vocabulary(path: str) -> Dict[str, int]:
    """
    :param path: Path to training CSV file containing index and space-separated token list
    :return: Dictionary that maps vocabulary words to ids in the vocabulary. All words are converted to lowercase.
             PAD and UNK are mapped to 0 and 1 respectively.
    """
    vocab = {'PAD': 0, 'UNK': 1}
    with open(path) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            # row[0] is index of the sentence in the dataset
            for token in row[1].lower().split():
                if token not in vocab:
                    vocab[token] = len(vocab)
    return vocab


def sentence2vec(vocabulary: Dict[str, int], token_list: List[str]) -> List[int]:
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


def load_unpadded_train_val_data(path: str, vocabulary: Dict[str, int], all_labels: Dict[int, int]):
    """
    Load train or validation data into unpadded lists divided into those with labels and those without labels
    :param path: Path to input file
    :param vocabulary: Word to id map
    :param all_labels: All existing labels. Maps id of sentence to label
    :return: Labeled and unlabeled sentence ids and data, labels, and longest sentence length
    """
    labeled_indices = []
    labeled_unpadded_data = []
    data_labels = []
    unlabeled_indices = []
    unlabeled_unpadded_data = []
    longest_sentence_length = 0

    with open(path) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            sentence_split = row[1].lower().split()
            if len(sentence_split) > longest_sentence_length:
                longest_sentence_length = len(sentence_split)
            idx = int(row[0])
            sentence_vec = sentence2vec(vocabulary, sentence_split)
            if idx in all_labels:
                labeled_indices.append(idx)
                labeled_unpadded_data.append(sentence_vec)
                data_labels.append(all_labels[idx])
            else:
                unlabeled_indices.append(idx)
                unlabeled_unpadded_data.append(sentence_vec)
    return labeled_indices, labeled_unpadded_data, data_labels, unlabeled_indices, unlabeled_unpadded_data,\
           longest_sentence_length


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


def create_batch_iterable(data, labels, batch_size, device):
    num_batches = (len(data) - 1) // batch_size + 1
    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = (i + 1) * batch_size
        yield(torch.tensor(data[start_idx:end_idx], dtype=torch.long, device=device),
              torch.tensor(labels[start_idx:end_idx], dtype=torch.long, device=device))
