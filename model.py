import torch
from torch import nn
import torch.nn.functional as F


class SimpleModel(nn.Module):
    def __init__(self, input_dim, output_dim, kernel=3):
        super(SimpleModel, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=input_dim, out_channels=output_dim, kernel_size=5, stride=1, padding=2),
            Layernorm(output_dim),
            nn.ReLU(),
            nn.Conv2d(in_channels=output_dim, out_channels=output_dim, kernel_size=3, stride=1, padding=1),
        )
        self.encoder = Encoder(num_layers=10, embed_size=output_dim, heads=7, ff_hidden_size=output_dim * 6, dropout=0.1,
                               kernel=kernel)

    def forward(self, x):
        x = x/x[:,0:1,:,:]
        x = self.conv(x)
        x = self.encoder(x)
        return x

class Layernorm(nn.Module):
    def __init__(self, input_dim, eps=1e-6):
        super(Layernorm, self).__init__()
        self.norm = nn.LayerNorm(input_dim, eps)

    def forward(self, x):
        N, h, f1, f2 = x.shape
        x = x.view(N, h, -1).transpose(1, 2)
        return self.norm(x).transpose(1, 2).view(N, h, f1, f2)

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_size, heads, drop):
        super(MultiHeadAttention, self).__init__()
        self.heads = heads
        self.embed_size = embed_size
        self.head_dim = embed_size // heads

        assert (
                self.head_dim * heads == embed_size
        ), "Embedding size must be divisible by heads"

        self.values = nn.Linear(embed_size, embed_size, bias=False)
        self.keys = nn.Linear(embed_size, embed_size, bias=False)
        self.queries = nn.Linear(embed_size, embed_size, bias=False)
        self.fc_out = nn.Linear(embed_size, embed_size)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        N, seq_length, _ = x.shape
        values = self.values(x)
        keys = self.keys(x)
        queries = self.queries(x)

        values = values.view(N, seq_length, self.heads, self.head_dim)
        keys = keys.view(N, seq_length, self.heads, self.head_dim)
        queries = queries.view(N, seq_length, self.heads, self.head_dim)

        values = values.transpose(1, 2)  # (N, heads, seq_length, head_dim)
        keys = keys.transpose(1, 2)  # (N, heads, seq_length, head_dim)
        queries = queries.transpose(1, 2)  # (N, heads, seq_length, head_dim)

        energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])
        attention = F.softmax(energy / (self.embed_size ** (1 / 2)), dim=3)
        attention = self.drop(attention)
        out = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(
            N, seq_length, self.heads * self.head_dim
        )

        return self.fc_out(out)


class Cov_FeedForward(nn.Module):
    def __init__(self, embed_size, ff_hidden_size, dropout, kernel):
        super(Cov_FeedForward, self).__init__()
        self.fc1 = nn.Sequential(
            nn.Conv2d(in_channels=embed_size, out_channels=ff_hidden_size, kernel_size=kernel, stride=1, padding=kernel//2),
            Layernorm(ff_hidden_size),
        )
        self.fc2 = nn.Sequential(
            nn.Conv2d(in_channels=ff_hidden_size, out_channels=embed_size, kernel_size=kernel, stride=1, padding=kernel//2),
            # Layernorm(embed_size),
        )
        self.dropout = nn.Dropout2d(dropout)

    def forward(self, x):
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return x


class EncoderLayer(nn.Module):
    def __init__(self, embed_size, heads, ff_hidden_size, dropout, kernel):
        super(EncoderLayer, self).__init__()
        self.multi_head_attention = MultiHeadAttention(embed_size, heads, dropout)
        self.feed_forward = Cov_FeedForward(embed_size, ff_hidden_size, dropout, kernel)
        self.norm1 = nn.LayerNorm(embed_size, 1e-6)
        self.norm2 = Layernorm(embed_size)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout2d(dropout)

    def forward(self, x):
        #  (N, hidden, f1, f2))
        N, h, f1, f2 = x.shape
        # x1 = x.transpose(1, 2)
        # x1 = x1.reshape(N * f1, h, f2).transpose(1, 2)
        # x1 = self.norm1(x1)
        # attention1 = self.multi_head_attention(x1).reshape(N, f1, f2, h).permute(0, 3, 1, 2)  # (N*f1, f2, h)
        #
        # x2 = x.transpose(1, 3)
        # x2 = x2.reshape(N * f2, f1, h)
        # x2 = self.norm2(x2)
        # attention2 = self.multi_head_attention(x2).reshape(N, f2, f1, h).permute(0, 3, 1, 2)  # (N*f2, f1, h)
        # attention = torch.concat([attention1, attention2], dim=1)
        # attention = self.norm3(self.conv(attention))
        x = x.view(N, h, -1).transpose(1, 2)
        x = self.dropout1(self.multi_head_attention(self.norm1(x))) + x

        x = x.transpose(1,2).view(N, h, f1, f2)

        forward = self.feed_forward(self.norm2(x))
        x = self.dropout2(forward) + x
        return x


class Encoder(nn.Module):
    def __init__(self, num_layers, embed_size, heads, ff_hidden_size, dropout, kernel):
        super(Encoder, self).__init__()
        self.layers = nn.ModuleList(
            [
                EncoderLayer(embed_size, heads, ff_hidden_size, dropout, kernel)
                for _ in range(num_layers)
            ]
        )
        self.conv = nn.Conv2d(in_channels=embed_size, out_channels=embed_size, kernel_size=1, stride=1, padding=0)
        self.norm = Layernorm(embed_size, 1e-6)

    def forward(self, x):
        N, h, f1, f2 = x.shape
        for layer in self.layers:
            x = layer(x)
        x = self.conv(x)
        x = self.norm(x)

        return x
