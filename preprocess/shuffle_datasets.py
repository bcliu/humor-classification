import random

train_path = 'data/shortjokes_train.csv'
val_path = 'data/shortjokes_val.csv'

train_data = []
val_data = []

for path, var in [(train_path, train_data), (val_path, val_data)]:
    with open(path) as f:
        for l in f.readlines():
            var.append(l)

all_data = train_data + val_data
random.shuffle(all_data)

train_data = all_data[:len(train_data)]
val_data = all_data[len(train_data):]

for path, var in [(train_path, train_data), (val_path, val_data)]:
    with open(path, 'w') as f:
        for l in var:
            stripped = l.strip()
            if stripped:
                f.write(stripped + '\n')
