import torch
import torch.nn as nn
from dataclasses import dataclass

from .decoder import DecoderLayer
from .TverskyLayer import TverskyLayer

@dataclass
class ModelConfig:
    vocab_size: int = 50257
    hidden_size: int = 512
    intermediate_size: int = 1024
    num_decoders: int = 24
    n_attn_heads: int = 16
    tversky_num_features: int = 4096

class CoolLanguageModelWowExclamationMark(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embedding = nn.Embedding(config.vocab_size, config.hidden_size)

        self.shared_tversky_features = nn.Parameter(
            torch.zeros(config.tversky_num_features, config.hidden_size)
        )

        self.layers = nn.ModuleList([
            DecoderLayer(config, layer_idx=i,
                        shared_tversky_features=self.shared_tversky_features)
            for i in range(config.num_decoders)
        ])

        self.output_layer = TverskyLayer(
            input_dim=config.hidden_size,
            prototypes=self.embedding.weight,  # Weight-tied prototypes to embeddings
            features=self.shared_tversky_features,
            prototype_init=None,
            feature_init=None
        )

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.02)
        torch.nn.init.uniform_(self.shared_tversky_features, -0.1, 0.1)

    def get_embeddings(self, input_ids, cu_seqlens=None, max_seqlen=None):
        """
        Returns embeddings before classifier head.
        input_ids: [total_seq_len] for varlen
        cu_seqlens, max_seqlen: for flash attention layers
        Returns: [total_seq_len, embed_size]
        """
        x = self.embedding(input_ids)  # [total_seq_len, embed_size]

        for layer in self.layers:
            x = layer(x, cu_seqlens=cu_seqlens, max_seqlen=max_seqlen)

        return x  # [total_seq_len, embed_size]

    def forward(self, input_ids, cu_seqlens=None, max_seqlen=None):
        """
        input_ids: [total_seq_len]
        Returns: [total_seq_len, vocab_size]
        """
        embeddings = self.get_embeddings(input_ids, cu_seqlens, max_seqlen)
        logits = self.output_layer(embeddings)
        return logits
