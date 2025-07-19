out_dir = 'out-tiny-midatos'
eval_interval = 200
eval_iters = 40
log_interval = 10

always_save_checkpoint = True

wandb_log = False
dataset = 'midatos'
gradient_accumulation_steps = 2
batch_size = 64
block_size = 128

n_layer = 6
n_head = 6
n_embd = 384
dropout = 0.1

learning_rate = 1e-3
max_iters = 5000
lr_decay_iters = 5000
min_lr = 1e-4
warmup_iters = 100

device = 'cuda'  # Cambia a 'cpu' si no tienes GPU
