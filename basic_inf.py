import torch
from safetensors.torch import load_model
from transformers import AutoTokenizer
from modeling.model import CoolLanguageModelWowExclamationMark, ModelConfig
from torch.amp import autocast

def generate(model, tokenizer, prompt, max_new_tokens=50, device="cuda"):
    model.eval()

    input_ids = torch.tensor(
        tokenizer.encode(prompt, add_special_tokens=True),
        dtype=torch.long,
        device=device
    )

    with torch.no_grad():
        for _ in range(max_new_tokens):
            seq_len = input_ids.size(0)
            cu_seqlens = torch.tensor([0, seq_len], dtype=torch.int32, device=device)

            with autocast(device_type='cuda', dtype=torch.bfloat16):
                logits = model(input_ids, cu_seqlens=cu_seqlens, max_seqlen=seq_len)

            next_token_logits = logits[-1, :]
            next_token = torch.argmax(next_token_logits, dim=-1)

            input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=0)

    return tokenizer.decode(input_ids.tolist())

def main():
    device = torch.device("cuda")

    config = ModelConfig()
    model = CoolLanguageModelWowExclamationMark(config)
    load_model(model, "model.safetensors")
    model.to(device)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    prompt = "The happy dog wagged"
    output = generate(model, tokenizer, prompt, max_new_tokens=50, device=device)

    print(output)

if __name__ == "__main__":
    main()
