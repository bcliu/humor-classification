import argparse
import bcolz
import numpy as np
import pickle

parser = argparse.ArgumentParser()
parser.add_argument('--glove-input', type=str, help='Path to GloVe input file')
parser.add_argument('--vector-output', type=str, help='File to save the vector pickle dump to')
parser.add_argument('--word2idx-output', type=str, help='File to save the word2idx pickle dump to')
args = parser.parse_args()

num_loaded = 0
word2idx = {}
vectors = bcolz.carray(np.zeros(1), rootdir=args.vector_output, mode='w')
dimensions = None

with open(args.glove_input, 'rb') as f:
    for l in f:
        line = l.decode().split()
        word = line[0]
        if num_loaded == 0:
            dimensions = len(line) - 1
        word2idx[word] = num_loaded
        num_loaded += 1
        vectors.append(np.array(line[1:]).astype(np.float))

print(f'Number of vectors loaded: {num_loaded}, dimension of vectors: {dimensions}')
vectors = bcolz.carray(vectors[1:].reshape((num_loaded, dimensions)), rootdir=args.vector_output, mode='w')
vectors.flush()
pickle.dump(word2idx, open(args.word2idx_output, 'wb'))
