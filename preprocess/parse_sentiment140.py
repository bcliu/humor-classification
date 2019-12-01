import os
import csv
import codecs
import random
from nltk.tokenize import word_tokenize

DIR = 'data/sentiment140'
# Separating sentences file from labels file to follow the input format of the training script. Labels are
# separated to perform active learning on the shortjokes dataset
TRAIN_PATH = os.path.join(DIR, 'training.1600000.processed.noemoticon.csv')
TRAIN_OUT_PATH = os.path.join(DIR, 'train.parsed.csv')
TRAIN_LABELS_OUT_PATH = os.path.join(DIR, 'train.labels.csv')

VAL_OUT_PATH = os.path.join(DIR, 'val.parsed.csv')
VAL_LABELS_OUT_PATH = os.path.join(DIR, 'val.labels.parsed.csv')

TEST_PATH = os.path.join(DIR, 'testdata.manual.2009.06.14.csv')
TEST_OUT_PATH = os.path.join(DIR, 'test.parsed.csv')
TEST_LABELS_OUT_PATH = os.path.join(DIR, 'test.labels.csv')

VAL_PERCENTAGE = 0.1

all_train = []

with codecs.open(TRAIN_PATH, 'r', encoding='utf-8', errors='ignore') as train_in:
    csv_reader = csv.reader(train_in, delimiter=',')

    for row in csv_reader:
        label = int(row[0])
        id = int(row[1])
        text = ' '.join(word_tokenize(row[-1]))
        all_train.append((id, text, label))

random.shuffle(all_train)

num_val = int(len(all_train) * VAL_PERCENTAGE)
train_split = all_train[:-num_val]
val_split = all_train[num_val:]

with open(TRAIN_OUT_PATH, 'w') as train_out, \
    open(TRAIN_LABELS_OUT_PATH, 'w') as train_labels_out:
    sentence_writer = csv.writer(train_out, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    labels_writer = csv.writer(train_labels_out, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

    for id, text, label in train_split:
        sentence_writer.writerow([id, text])
        labels_writer.writerow([id, label])

with open(VAL_OUT_PATH, 'w') as val_out, \
    open(VAL_LABELS_OUT_PATH, 'w') as val_labels_out:
    sentence_writer = csv.writer(val_out, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    labels_writer = csv.writer(val_labels_out, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

    for id, text, label in val_split:
        sentence_writer.writerow([id, text])
        labels_writer.writerow([id, label])

with codecs.open(TEST_PATH, 'r', encoding='utf-8', errors='ignore') as test_in, \
    open(TEST_OUT_PATH, 'w') as test_out, \
    open(TEST_LABELS_OUT_PATH, 'w') as test_labels_out:
    csv_reader = csv.reader(test_in, delimiter=',')
    sentence_writer = csv.writer(test_out, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    labels_writer = csv.writer(test_labels_out, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

    for row in csv_reader:
        label = int(row[0])
        # Since training data do not have neutral (label=2) sentences, remove those from test data too
        if label == 2:
            continue
        id = int(row[1])
        tokenized_sentence = ' '.join(word_tokenize(row[-1]))
        sentence_writer.writerow([id, tokenized_sentence])
        labels_writer.writerow([id, label])
