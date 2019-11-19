import csv
from nltk.tokenize import word_tokenize
import random

TRAIN_RATIO = 0.9

shortjokes_processed = []
with open('data/shortjokes.csv') as shortjokes:
    column_names = shortjokes.readline()
    csv_reader = csv.reader(shortjokes, delimiter=',')

    for row in csv_reader:
        tokens = word_tokenize(row[1].strip())
        shortjokes_processed.append([row[0], ' '.join(tokens)])

print('Shortjokes loaded: %d' % len(shortjokes_processed))

random.shuffle(shortjokes_processed)
num_train = int(TRAIN_RATIO * len(shortjokes_processed))

with open('data/shortjokes_train.csv', 'w') as train_file:
    csv_writer = csv.writer(train_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    for i in range(num_train):
        csv_writer.writerow(shortjokes_processed[i])

with open('data/shortjokes_val.csv', 'w') as val_file:
    csv_writer = csv.writer(val_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    for i in range(num_train, len(shortjokes_processed)):
        csv_writer.writerow(shortjokes_processed[i])
