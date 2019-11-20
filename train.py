import pickle

import bcolz
import click
import torch
import torch.nn.functional as F

from annotator import load_sentences_or_categories, load_existing_annotations
from models.TextCNN import TextCNN
from utils import create_vocabulary, load_unpadded_train_val_data, create_padded_data, create_weight_matrix, \
    create_batch_iterable

NUM_FILTERS = 2
WINDOW_SIZES = [2, 3, 4, 5]
LR = 1e-4
OPTIM_EPS = 1e-9
NUM_EPOCHS = 1000
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_one_epoch(model, data_batches, optimizer):
    model.train()
    for batch_idx, (data, labels) in enumerate(data_batches):
        optimizer.zero_grad()
        pred = model(data)
        loss = F.cross_entropy(pred, labels)
        loss.backward()
        optimizer.step()


def evaluate(model, test_batches):
    model.eval()
    for _, (data, labels) in enumerate(test_batches):
        pred = model(data)
        print(torch.sum(torch.argmax(pred, dim=1) != labels) * 1.0 / len(labels))


def rank_unlabeled_train(model, data, indices):
    """
    Rank unlabeled training data using a model by uncertainty
    :return: Indices of data ranked by uncertainty
    """
    pass


@click.command()
@click.option('--train-path', help='Path to training file', required=True)
@click.option('--val-path', help='Path to validation file', required=True)
@click.option('--labels-path', help='Path to labels file', required=True)
@click.option('--embedding-vectors-path', help='Path to embeddings vector bcolz directory (e.g. GloVe)', required=True)
@click.option('--embedding-word2idx-path', help='Path to embeddings word2idx file', required=True)
@click.option('--categories-def-path',
              help='Path to categories definition CSV file, in the format of ID,NAME',
              required=True)
@click.option('--batch-size', type=int, default=64)
def main(train_path, val_path, labels_path, embedding_vectors_path, embedding_word2idx_path,
         categories_def_path, batch_size):
    embedding_vectors = bcolz.open(embedding_vectors_path)[:]
    embedding_dim = len(embedding_vectors[0])
    embedding_word2idx = pickle.load(open(embedding_word2idx_path, 'rb'))
    # Maps words to embedding vectors. These are all embeddings available to us
    embeddings = {w: embedding_vectors[embedding_word2idx[w]] for w in embedding_word2idx}

    # Build vocabulary using training set. Maps words to indices
    vocab = create_vocabulary(train_path)
    vocab_size = len(vocab)
    print(f'Vocabulary size: {vocab_size}')
    labels = load_existing_annotations(labels_path)

    # Stores indexes of sentences provided in the original dataset
    train_labeled_idx, train_labeled_data_unpadded, train_labels, train_unlabeled_idx, train_unlabeled_data_unpadded,\
        longest_sentence_length = load_unpadded_train_val_data(train_path, vocab, labels)
    val_labeled_idx, val_labeled_data_unpadded, val_labels, val_unlabeled_idx, val_unlabeled_data_unpadded,\
        _ = load_unpadded_train_val_data(val_path, vocab, labels)

    # Create padded train and val dataset
    # TODO: Do not use longest length to pad input. Find mean and std
    train_labeled_data = create_padded_data(train_labeled_data_unpadded, longest_sentence_length)
    val_labeled_data = create_padded_data(val_labeled_data_unpadded, longest_sentence_length)

    humor_types = load_sentences_or_categories(categories_def_path)
    word_weight_matrix = create_weight_matrix(vocab, embeddings, device)

    textCNN = TextCNN(word_weight_matrix, NUM_FILTERS, WINDOW_SIZES, len(humor_types))
    optimizer = torch.optim.Adam(textCNN.parameters(), lr=LR, eps=OPTIM_EPS)

    for i in range(NUM_EPOCHS):
        train_one_epoch(textCNN, create_batch_iterable(train_labeled_data, train_labels, batch_size, device), optimizer)
        evaluate(textCNN, create_batch_iterable(train_labeled_data, train_labels, len(val_labeled_data), device))


if __name__ == '__main__':
    main()
