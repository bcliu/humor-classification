import pickle

import bcolz
import click
import torch
from torch.nn.parallel import DataParallel
import torch.nn.functional as F
from torch.distributions import Categorical
from sklearn.metrics import f1_score, precision_score, recall_score

from annotator import load_sentences_or_categories, load_existing_annotations
from models.TextCNN import TextCNN
from utils import create_vocabulary, load_unpadded_train_val_data, create_padded_data, create_weight_matrix, \
    create_batch_iterable

NUM_FILTERS = 64
WINDOW_SIZES = [1, 2, 3, 4, 5, 7, 9]
LR = 1e-3
OPTIM_EPS = 1e-9
NUM_EPOCHS = 100
VAL_SAMPLE_SIZE = 256
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_one_epoch(model, data_batches, optimizer, val_labeled_data, val_labels, num_iterations):
    model.train()
    for batch_idx, (data, labels) in enumerate(data_batches):
        optimizer.zero_grad()
        pred = model(data)
        loss = F.cross_entropy(pred, labels)
        if batch_idx % 20 == 0:
            f1, precision, recall = compute_f1_score(labels, pred)
            print(f'{batch_idx}/{num_iterations}, loss {loss.item()}, f1 score: {f1}, precision: {precision}, recall: {recall}')
        loss.backward()
        optimizer.step()

        # Free GPU memory associated with these two otherwise evaluation will result in OOM
        del pred
        del loss

        if batch_idx % 500 == 0:
            print('Validation error rate:')
            evaluate(batch_idx, model,
                     torch.tensor(val_labeled_data[:VAL_SAMPLE_SIZE], dtype=torch.long, device=device),
                     torch.tensor(val_labels[:VAL_SAMPLE_SIZE], dtype=torch.long, device=device))

def compute_f1_score(labels, pred):
    labels_numpy = labels.cpu().detach().numpy()
    pred_numpy = torch.argmax(pred, dim=1).cpu().detach().numpy()
    f1 = f1_score(labels_numpy, pred_numpy, average='macro')
    precision = precision_score(labels_numpy, pred_numpy, average='macro')
    recall = recall_score(labels_numpy, pred_numpy, average='macro')
    return f1, precision, recall

def evaluate(epoch, model, data, labels):
    model.eval()
    pred = model(data, train=False)
    error = torch.sum(torch.argmax(pred, dim=1) != labels) * 1.0 / labels.shape[0]
    f1, precision, recall = compute_f1_score(labels, pred)
    print(f'Error: {error}, f1: {f1}, precision: {precision}, recall: {recall}')

def rank_unlabeled_train(model, data, indices, uncertainty_output):
    """
    Rank unlabeled training data using a model by uncertainty
    :return: Indices of data ranked by uncertainty
    """
    model.eval()
    limit = 30000
    # TODO: batch here as well, memory constraint
    pred = model(data[:limit], train=False)
    entropy_list = Categorical(probs=pred).entropy().cpu().detach().numpy()
    entropy_idx_combined = [(entropy_list[idx], indices[idx]) for idx in range(limit)]
    entropy_idx_combined.sort(key=lambda x: x[0], reverse=True)
    with open(uncertainty_output, 'w') as output:
        for i in range(len(entropy_idx_combined)):
            output.write(f'{entropy_idx_combined[i][1]}\n')

@click.command()
@click.option('--train-path', help='Path to training file', required=True)
@click.option('--val-path', help='Path to validation file', required=True)
@click.option('--labels-path', help='Path to labels file', required=True)
@click.option('--embedding-vectors-path', help='Path to embeddings vector bcolz directory (e.g. GloVe)', required=True)
@click.option('--embedding-word2idx-path', help='Path to embeddings word2idx file', required=True)
@click.option('--categories-def-path',
              help='Path to categories definition CSV file, in the format of ID,NAME',
              required=True)
@click.option('--uncertainty-output-path', required=False)
@click.option('--batch-size', type=int, default=64)
@click.option('--model-snapshot-prefix', type=str, required=False)
@click.option('--model-snapshot-interval', type=int, required=False)
@click.option('--pretrained-model-path', type=str, required=False)
def main(train_path, val_path, labels_path, embedding_vectors_path, embedding_word2idx_path,
         categories_def_path, uncertainty_output_path, batch_size, model_snapshot_prefix, pretrained_model_path,
         model_snapshot_interval):
    embedding_vectors = bcolz.open(embedding_vectors_path)[:]
    embedding_dim = len(embedding_vectors[0])
    embedding_word2idx = pickle.load(open(embedding_word2idx_path, 'rb'))
    # Maps words to embedding vectors. These are all embeddings available to us
    embeddings = {w: embedding_vectors[embedding_word2idx[w]] for w in embedding_word2idx}

    # Build vocabulary using training set. Maps words to indices
    vocab = create_vocabulary(train_path)
    vocab_size = len(vocab)
    print(f'Vocabulary size: {vocab_size}\nBatch size: {batch_size}')
    # TODO: take advantage of the multiple annotations
    labels = load_existing_annotations(labels_path, load_first_annotation_only=True)

    if model_snapshot_interval:
        print(f'Taking model snapshot every {model_snapshot_interval} epochs')
    else:
        print(f'Taking model snapshot ONLY at the end of training')

    humor_types = load_sentences_or_categories(categories_def_path)
    # Map label IDs to indices so that when computing cross entropy we don't operate on raw label IDs
    label_id_to_idx = {label_id: idx for idx, label_id in enumerate(humor_types)}
    word_weight_matrix = create_weight_matrix(vocab, embeddings, device)

    # Stores indexes of sentences provided in the original dataset
    train_labeled_idx, train_labeled_data_unpadded, train_labels, train_unlabeled_idx, train_unlabeled_data_unpadded,\
        longest_sentence_length = load_unpadded_train_val_data(train_path, vocab, labels, label_id_to_idx)
    val_labeled_idx, val_labeled_data_unpadded, val_labels, val_unlabeled_idx, val_unlabeled_data_unpadded,\
        _ = load_unpadded_train_val_data(val_path, vocab, labels, label_id_to_idx)

    # Create padded train and val dataset
    # TODO: Do not use longest length to pad input. Find mean and std
    train_labeled_data = create_padded_data(train_labeled_data_unpadded, longest_sentence_length)
    val_labeled_data = create_padded_data(val_labeled_data_unpadded, longest_sentence_length)

    print(f'Num of labeled training data: {train_labeled_data.shape[0]}, labeled val: {val_labeled_data.shape[0]}')

    num_iterations = train_labeled_data.shape[0] // batch_size

    textCNN = DataParallel(TextCNN(word_weight_matrix, NUM_FILTERS, WINDOW_SIZES, len(humor_types))).to(device)
    if pretrained_model_path:
        textCNN.module.initialize_from_pretrained(pretrained_model_path)
    optimizer = torch.optim.Adam(textCNN.parameters(), lr=LR, eps=OPTIM_EPS)

    for i in range(NUM_EPOCHS):
        print(f'Epoch {i}')
        train_one_epoch(textCNN, create_batch_iterable(train_labeled_data, train_labels, batch_size, device),
                        optimizer, val_labeled_data, val_labels, num_iterations)
        if model_snapshot_prefix:
            if (not model_snapshot_interval and i + 1 == NUM_EPOCHS) or \
                    (model_snapshot_interval and (i + 1) % model_snapshot_interval == 0):
                print('\nSaving model snapshot...')
                torch.save(textCNN.state_dict(), f'{model_snapshot_prefix}_epoch{i}.mdl')
                print('Saved\n')

    if uncertainty_output_path:
        train_unlabeled_data = create_padded_data(train_unlabeled_data_unpadded, longest_sentence_length)
        rank_unlabeled_train(textCNN,
                             torch.tensor(train_unlabeled_data, dtype=torch.long, device=device),
                             train_unlabeled_idx,
                             uncertainty_output_path)

if __name__ == '__main__':
    main()
