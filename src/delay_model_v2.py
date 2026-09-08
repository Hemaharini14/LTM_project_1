"""
Entity-embedding MLP for the flight+weather disruption model. Same
architecture pattern as the earlier Airlines-only model, generalized to
however many categorical/continuous columns the encoder was fit with.
"""
import torch
import torch.nn as nn


class DelayNetV2(nn.Module):
    def __init__(self, vocab_sizes: dict, n_continuous: int, embed_dim: int = 12, hidden: int = 128):
        super().__init__()
        self.cat_cols = list(vocab_sizes.keys())
        self.embeddings = nn.ModuleDict({
            col: nn.Embedding(num_embeddings=size, embedding_dim=min(embed_dim, (size // 2) + 1))
            for col, size in vocab_sizes.items()
        })
        total_embed_dim = sum(e.embedding_dim for e in self.embeddings.values())
        input_dim = total_embed_dim + n_continuous

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.BatchNorm1d(hidden),
            nn.Dropout(0.25),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.BatchNorm1d(hidden // 2),
            nn.Dropout(0.15),
            nn.Linear(hidden // 2, hidden // 4),
            nn.ReLU(),
            nn.Linear(hidden // 4, 1),
        )

    def forward(self, x_cat: torch.Tensor, x_cont: torch.Tensor) -> torch.Tensor:
        embs = [self.embeddings[col](x_cat[:, i]) for i, col in enumerate(self.cat_cols)]
        x = torch.cat(embs + [x_cont], dim=1)
        return self.mlp(x).squeeze(-1)