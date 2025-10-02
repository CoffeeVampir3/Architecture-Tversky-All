import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset
from transformers import AutoTokenizer
from tqdm import tqdm
from torch.amp import autocast
from safetensors.torch import save_model, load_model

from modeling.model import CoolLanguageModelWowExclamationMark, ModelConfig
from modeling.zRMSNorm import ZeroCenteredRMSNorm
from modeling.TverskyLayer import TverskyLayer

#torch.autograd.set_detect_anomaly(True)
torch.set_float32_matmul_precision('high')

class TextDataset(Dataset):
    def __init__(self, sequences):
        self.sequences = sequences

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx]

def varlen_collate_fn(batch):
    """
    input_ids -> [batch_size] list of [seq_len_i] -> [total_seq_len]
    cu_seqlens -> flash_attn_varlen_func: [0, seq_len_0, seq_len_0+seq_len_1, ...]
    max_seqlen -> token length of longest sequence
    """
    sequence_lengths = [len(seq) for seq in batch]
    concatenated = torch.cat(batch, dim=0)
    cu_seqlens = torch.zeros(len(batch) + 1, dtype=torch.int32)
    cu_seqlens[1:] = torch.cumsum(torch.tensor(sequence_lengths, dtype=torch.int32), dim=0)

    return {
        'input_ids': concatenated,
        'cu_seqlens': cu_seqlens,
        'max_seqlen': max(sequence_lengths),
        'sequence_lengths': sequence_lengths,
        'batch_size': len(batch)
    }

def load_and_preprocess_data(max_length=256):
    dataset = load_dataset("skeskinen/TinyStories-hf", split="train[:10%]")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    def tokenize_and_filter(examples):
        texts = [text.strip() for text in examples["text"]]

        encoded = tokenizer(
            texts,
            add_special_tokens=True,
            truncation=False,
            padding=False,
            return_attention_mask=False,
            return_token_type_ids=False,
        )

        valid_sequences = []
        for input_ids in encoded["input_ids"]:
            if len(input_ids) > 0 and len(input_ids) < max_length:
                valid_sequences.append(input_ids)

        return {"input_ids": valid_sequences}

    tokenized_dataset = dataset.map(
        tokenize_and_filter,
        batched=True,
        batch_size=1000,
        num_proc=10,
        remove_columns=dataset.column_names,
    )

    sequences = [torch.tensor(item["input_ids"], dtype=torch.long)
                 for item in tokenized_dataset]

    return TextDataset(sequences), tokenizer

def compute_varlen_loss_with_ce(model, concatenated_input_ids, cu_seqlens, sequence_lengths):
    inputs = []
    targets = []
    batch_size = len(cu_seqlens) - 1
    for i in range(batch_size):
        start_idx = cu_seqlens[i]
        end_idx = cu_seqlens[i + 1]
        seq_length = end_idx - start_idx
        if seq_length > 1:
            inputs.append(concatenated_input_ids[start_idx:end_idx-1])
            targets.append(concatenated_input_ids[start_idx+1:end_idx])

    model_inputs = torch.cat(inputs, dim=0)
    model_targets = torch.cat(targets, dim=0)

    input_lengths = [seq_length - 1 for seq_length in sequence_lengths if seq_length > 1]
    cu_seqlens_shifted = torch.zeros(len(input_lengths) + 1, dtype=torch.int32, device=concatenated_input_ids.device)
    cu_seqlens_shifted[1:] = torch.cumsum(torch.tensor(input_lengths, dtype=torch.int32), dim=0)
    max_seqlen_shifted = max(input_lengths)

    logits = model(model_inputs, cu_seqlens=cu_seqlens_shifted, max_seqlen=max_seqlen_shifted)

    loss = F.cross_entropy(logits, model_targets)
    return loss

def build_weight_decay_optm(model, learning_rate):
    zero_centered_rmsnorm_params = []
    tversky_params = []
    decay_params = []
    no_decay_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        module = model.get_submodule('.'.join(name.split('.')[:-1]))

        if isinstance(module, ZeroCenteredRMSNorm):
            zero_centered_rmsnorm_params.append(param)
        elif isinstance(module, TverskyLayer):
            if 'features' in name or 'prototypes' in name:
                tversky_params.append(param)
        elif any(exclude in name for exclude in [
            'bias', 'embedding', 'output_layer', 'norm.weight',
            'layernorm', 'dt_bias', 'A_log', 'expert_biases'
        ]):
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    # This comes out of nowhere apparently so let me explain the rationale for why each of these appears:

    # Zero centered RMS norm is proposed by qwen3next, we decay the single weight parameter so this converges
    # towards regular RMS norm over time, but we still get the nice distribution early on.

    # We're doing a somewhat approximate tversky with a sharpness that has issues with similarity near zero.
    # However, we also run into problems where the gradients are bad at "large" values
    # So weight decay is important to regulate away from the areas of bad gradients so the neurons don't die
    # But it's equally important to avoid regularizing all of our weights towards zero.

    # The undecayed params are for our large embeddings mostly that don't seem to benefit from weight decay.
    return torch.optim.Adam([
        {'params': zero_centered_rmsnorm_params, 'weight_decay': 1e-4},
        {'params': tversky_params, 'weight_decay': 2e-4},
        {'params': decay_params, 'weight_decay': 1e-5},
        {'params': no_decay_params, 'weight_decay': 0.0}
    ], lr=learning_rate, eps=1e-16)


class CosineWarmupScheduler:
    def __init__(self, optimizer, warmup_steps=25, total_steps=None, peak_lr=1e-4, min_lr=1e-6):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.peak_lr = peak_lr
        self.min_lr = min_lr
        self.step_count = 0

    def step(self):
        if self.step_count < self.warmup_steps:
            lr = self.peak_lr * (self.step_count / self.warmup_steps)
        else:
            progress = (self.step_count - self.warmup_steps) / (self.total_steps - self.warmup_steps)
            lr = self.min_lr + 0.5 * (self.peak_lr - self.min_lr) * (1 + math.cos(math.pi * progress))

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        self.step_count += 1
        return lr

def train(model, train_dataset, tokenizer, num_epochs=1, batch_size=16, learning_rate=3e-4):
    device = torch.device("cuda")
    model.to(device)
    optimizer =  build_weight_decay_optm(model, learning_rate)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        collate_fn=varlen_collate_fn
    )

    total_steps = len(train_loader) * num_epochs
    scheduler = CosineWarmupScheduler(optimizer, warmup_steps=25, total_steps=total_steps, peak_lr=learning_rate)

    global_step = 0
    total_tokens = 0
    epoch_tokens = 0

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        total_tokens += epoch_tokens
        epoch_tokens = 0

        for batch_idx, batch in enumerate(tqdm(train_loader)):
            input_ids = batch["input_ids"].to(device)
            cu_seqlens = batch["cu_seqlens"].to(device)
            max_seqlen = batch["max_seqlen"]
            sequence_lengths = batch["sequence_lengths"]

            optimizer.zero_grad()

            with autocast(device_type='cuda', dtype=torch.bfloat16):
                loss = compute_varlen_loss_with_ce(model, input_ids, cu_seqlens, sequence_lengths)

            loss.backward()
            optimizer.step()

            scheduler.step()

            total_loss += loss.item()
            global_step += 1

            epoch_tokens += input_ids.size(0)

            if global_step % 20 == 0:
                avg_seq_len = sum(sequence_lengths) / len(sequence_lengths)
                print(f"Step {global_step}, Loss: {loss.item():.4f}, Avg Seq Len: {avg_seq_len:.1f}, Total Tokens: {input_ids.size(0)}")

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{num_epochs}, Average Loss: {avg_loss:.4f}, Epoch Total Tokens: {epoch_tokens}")

def main():
    config = ModelConfig()
    train_dataset, tokenizer = load_and_preprocess_data()

    model = CoolLanguageModelWowExclamationMark(config)

    # model.compile(
    #     mode="reduce-overhead",
    # )
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    train(model, train_dataset, tokenizer)
    save_model(model, "model.safetensors")

if __name__ == "__main__":
    main()
