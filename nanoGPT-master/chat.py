import torch
from model import GPT  # o como se llame el modelo en nanoGPT-master
from tokenizers import Tokenizer  # si usas tokenizador externo
import sys

# --- Configuración básica ---
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Ruta al checkpoint guardado
checkpoint_path = 'out-tiny-midatos/checkpoint.pt'

# Aquí pon tu vocab_size y block_size usados en entrenamiento
vocab_size = 65  # Ejemplo, cambia según tu config
block_size = 128

# Crear modelo (usa la clase GPT de nanoGPT)
model = GPT(
    vocab_size=vocab_size,
    block_size=block_size,
    n_layer=4,
    n_head=4,
    n_embd=128
)

model.load_state_dict(torch.load(checkpoint_path, map_location=device))
model.to(device)
model.eval()

# --- Tokenizador simple (ejemplo si usas char-level) ---
# NanoGPT original usa un tokenizador simple por caracteres, puedes adaptar según tu caso

# Mapea caracter a índice (hazlo según tu vocab)
itos = [chr(i) for i in range(32, 127)]  # ejemplo ASCII imprimibles
stoi = {ch: i for i, ch in enumerate(itos)}

def encode(text):
    return [stoi.get(c, 0) for c in text]

def decode(indices):
    return ''.join([itos[i] for i in indices])

# --- Función para generar texto ---
@torch.no_grad()
def generate(model, start_text, max_new_tokens=100):
    model.eval()
    context = torch.tensor([encode(start_text)], dtype=torch.long, device=device)
    generated = context

    for _ in range(max_new_tokens):
        if generated.size(1) > block_size:
            x_cond = generated[:, -block_size:]
        else:
            x_cond = generated

        logits, _ = model(x_cond)
        logits = logits[:, -1, :]
        probs = torch.nn.functional.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        generated = torch.cat((generated, next_token), dim=1)

    output = generated[0].tolist()
    return decode(output)

# --- Loop de chat ---
print("NanoGPT Chat. Escribe 'salir' para terminar.")
while True:
    prompt = input("Tú: ")
    if prompt.lower() == 'salir':
        break
    output = generate(model, prompt, max_new_tokens=100)
    # Para solo mostrar el texto generado después del prompt
    respuesta = output[len(prompt):]
    print("GPT:", respuesta)
