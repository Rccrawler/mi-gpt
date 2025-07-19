import os
import pickle
import numpy as np
from tqdm import tqdm

# Configura tus rutas
input_file_path = 'data/midatos/input.txt'
output_dir = 'data/midatos'
dataset_name = 'midatos'

# Lee todo el texto
with open(input_file_path, 'r', encoding='utf-8') as f:
    data = f.read()

# Divide entre entrenamiento (90%) y validación (10%)
n = len(data)
train_data = data[:int(n * 0.9)]
val_data = data[int(n * 0.9):]

# Construye vocabulario simple de caracteres
chars = sorted(list(set(data)))
vocab_size = len(chars)
print("Vocab size:", vocab_size)
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for ch, i in stoi.items()}

# Guarda el vocabulario como meta
meta = {
    'vocab_size': vocab_size,
    'itos': itos,
    'stoi': stoi,
}
with open(os.path.join(output_dir, 'meta.pkl'), 'wb') as f:
    pickle.dump(meta, f)

# Codifica los datos a enteros
def encode(s): return [stoi[c] for c in s]
train_ids = encode(train_data)
val_ids = encode(val_data)

# Guarda como binarios
train_ids = np.array(train_ids, dtype=np.uint16)
val_ids = np.array(val_ids, dtype=np.uint16)
train_ids.tofile(os.path.join(output_dir, 'train.bin'))
val_ids.tofile(os.path.join(output_dir, 'val.bin'))

print(f"{dataset_name} preparado: {len(train_ids)} tokens train, {len(val_ids)} tokens val")
