import torch
from torch import nn
import torch.nn.functional as F
from torch import Tensor

class Reduce(nn.Module):
    def forward(self, x):
        return x.mean(dim=1)


class BatchNorm(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.norm = nn.BatchNorm1d(embed_dim)

    def forward(self, x):
        return self.norm(x.transpose(-1, -2)).transpose(-1, -2)


class PatchEmbedding(nn.Module):
    def __init__(self, in_channels: int = 3, patch_size: int = 16, emb_size: int = 768, img_size: int = 224):
        self.patch_size = patch_size
        super().__init__()
        self.projection = nn.Conv2d(in_channels, emb_size, kernel_size=patch_size, stride=patch_size)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, emb_size))
        self.positions = nn.Parameter(torch.zeros((img_size // patch_size) * (img_size // patch_size) + 1, emb_size))

    def forward(self, x: Tensor) -> Tensor:
        b, _, _, _ = x.shape
        x = self.projection(x)
        x = x.flatten(2).transpose(1, 2)
        cls_tokens = self.cls_token.expand(b, -1, -1)
        # prepend the cls token to the input
        x = torch.cat([cls_tokens, x], dim=1)
        # add position embedding
        x += self.positions
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model = 123, num_heads = 6, dropout = 0.):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.scale = self.head_dim ** -0.5

        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        self.out = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

    def split_heads(self, x, batch_size):
        x = x.view(batch_size, -1, self.num_heads, self.head_dim)
        return x.transpose(1, 2)

    def forward(self, x):
        batch_size = x.size(0)

        query = self.query(x)
        key = self.key(x)
        value = self.value(x)

        query = self.split_heads(query, batch_size)
        key = self.split_heads(key, batch_size)
        value = self.split_heads(value, batch_size)

        scores = torch.matmul(query, key.transpose(-2, -1))
        scores = scores * self.scale

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        attn_output = torch.matmul(attn_weights, value)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        attn_output = self.out(attn_output)

        return attn_output


class ResidualAdd(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        res = x
        x = self.fn(x, **kwargs)
        x += res
        return x


class FeedForwardBlock(nn.Sequential):
    def __init__(self, emb_size: int, expansion: int = 2, drop_p: float = 0.):
        super().__init__(
            nn.Linear(emb_size, expansion * emb_size),
            nn.ReLU(),
            nn.Dropout(drop_p),
            nn.Linear(expansion * emb_size, emb_size),
        )


class TransformerEncoderBlock(nn.Sequential):
    def __init__(self,
                 emb_size: int = 768,
                 drop_p: float = 0.,
                 forward_expansion: int = 2,
                 forward_drop_p: float = 0.,
                 ** kwargs):
        super().__init__(
            ResidualAdd(nn.Sequential(
                BatchNorm(emb_size),
                MultiHeadAttention(emb_size, **kwargs),
                nn.Dropout(drop_p)
            )),
            ResidualAdd(nn.Sequential(
                BatchNorm(emb_size),
                FeedForwardBlock(
                    emb_size, expansion=forward_expansion, drop_p=forward_drop_p),
                nn.Dropout(drop_p)
            )
            ))


class TransformerEncoder(nn.Sequential):
    def __init__(self, depth: int = 1, **kwargs):
        super().__init__(*[TransformerEncoderBlock(**kwargs) for _ in range(depth)])


class ClassificationHead(nn.Sequential):
    def __init__(self, emb_size: int = 768, n_classes: int = 1000):
        super().__init__(
            Reduce(),
            nn.BatchNorm1d(emb_size),
            nn.Linear(emb_size, n_classes))


class ViT_synthetic(nn.Sequential):
    def __init__(self, in_ch,
                num_heads: int = 2,
                patch_size: int = 1,
                emb_size: int = 16,
                img_size: int = 5,
                depth: int = 2,
                n_classes: int = 2,
                **kwargs):
        super().__init__(
            PatchEmbedding(in_ch, patch_size, emb_size, img_size),
            TransformerEncoder(depth, emb_size=emb_size, num_heads=num_heads, **kwargs),
            ClassificationHead(emb_size, n_classes)
        )

def synthetic_vit_ch1():
    return ViT_synthetic(in_ch=1, num_heads=2, patch_size=1, emb_size=16, img_size=5, depth=2, n_classes=2)

def synthetic_vit_ch3():
    return ViT_synthetic(in_ch=3, num_heads=2, patch_size=1, emb_size=16, img_size=5, depth=2, n_classes=2)