import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F
from einops import rearrange
from einops.layers.torch import Rearrange

class MSCNet(nn.Module):
    def __init__(self, f1=16, pooling_size=52, dropout_rate=0.5, number_channel=22):
        super().__init__()
        # Multi-scale temporal convolutions followed by depthwise spatial convolutions
        self.cnn1 = nn.Sequential(
            nn.Conv2d(1, f1, (1, 125), (1, 1), padding='same'),
            nn.Conv2d(f1, f1, (number_channel, 1), (1, 1), groups=f1),
            nn.BatchNorm2d(f1), nn.ELU(),
            nn.AvgPool2d((1, pooling_size)), nn.Dropout(dropout_rate),
        )
        self.cnn2 = nn.Sequential(
            nn.Conv2d(1, f1, (1, 62), (1, 1), padding='same'),
            nn.Conv2d(f1, f1, (number_channel, 1), (1, 1), groups=f1),
            nn.BatchNorm2d(f1), nn.ELU(),
            nn.AvgPool2d((1, pooling_size)), nn.Dropout(dropout_rate),
        )
        self.cnn3 = nn.Sequential(
            nn.Conv2d(1, f1, (1, 31), (1, 1), padding='same'),
            nn.Conv2d(f1, f1, (number_channel, 1), (1, 1), groups=f1),
            nn.BatchNorm2d(f1), nn.ELU(),
            nn.AvgPool2d((1, pooling_size)), nn.Dropout(dropout_rate),
        )
        self.projection = Rearrange('b e (h) (w) -> b (h w) e')

    def forward(self, x: Tensor) -> Tensor:
        x1, x2, x3 = self.cnn1(x), self.cnn2(x), self.cnn3(x)
        x = torch.cat([x1, x2, x3], dim=1)
        return self.projection(x)

class CausalConv1d(nn.Conv1d):
    def forward(self, x: Tensor) -> Tensor:
        padding = (self.kernel_size[0] - 1) * self.dilation[0]
        x = F.pad(x, (padding, 0))
        return super().forward(x)

class _TCNBlock(nn.Module):
    def __init__(self, input_dimension, depth, kernel_size, filters, drop_prob, activation=nn.ELU):
        super().__init__()
        self.activation = activation()
        self.layers = nn.ModuleList()
        self.downsample = nn.Conv1d(input_dimension, filters, 1) if input_dimension != filters else None

        for i in range(depth):
            dilation = 2**i
            self.layers.append(nn.Sequential(
                CausalConv1d(input_dimension if i == 0 else filters, filters, kernel_size, dilation=dilation, bias=False),
                nn.BatchNorm1d(filters), self.activation, nn.Dropout(drop_prob),
                CausalConv1d(filters, filters, kernel_size, dilation=dilation, bias=False),
                nn.BatchNorm1d(filters), self.activation, nn.Dropout(drop_prob),
            ))

    def forward(self, x: Tensor) -> Tensor:
        x = x.permute(0, 2, 1)
        res = x if self.downsample is None else self.downsample(x)
        for layer in self.layers:
            out = self.activation(layer(x) + res)
            res, x = out, out
        return out.permute(0, 2, 1)

class MultiHeadAttention(nn.Module):
    def __init__(self, emb_size, num_heads, dropout):
        super().__init__()
        self.emb_size, self.num_heads = emb_size, num_heads
        self.keys = nn.Linear(emb_size, emb_size)
        self.queries = nn.Linear(emb_size, emb_size)
        self.values = nn.Linear(emb_size, emb_size)
        self.att_drop = nn.Dropout(dropout)
        self.projection = nn.Linear(emb_size, emb_size)

    def forward(self, x: Tensor) -> Tensor:
        q = rearrange(self.queries(x), "b n (h d) -> b h n d", h=self.num_heads)
        k = rearrange(self.keys(x), "b n (h d) -> b h n d", h=self.num_heads)
        v = rearrange(self.values(x), "b n (h d) -> b h n d", h=self.num_heads)
        energy = torch.einsum('bhqd, bhkd -> bhqk', q, k)
        att = self.att_drop(F.softmax(energy / (self.emb_size ** 0.5), dim=-1))
        out = torch.einsum('bhal, bhlv -> bhav ', att, v)
        return self.projection(rearrange(out, "b h n d -> b n (h d)"))

class ResidualAdd(nn.Module):
    def __init__(self, fn, emb_size, drop_p):
        super().__init__()
        self.fn, self.layernorm = fn, nn.LayerNorm(emb_size)
        self.drop = nn.Dropout(drop_p)

    def forward(self, x, **kwargs):
        return self.layernorm(self.drop(self.fn(x, **kwargs)) + x)

class TransformerEncoder(nn.Sequential):
    def __init__(self, heads, depth, emb_size):
        super().__init__(*[
            ResidualAdd(MultiHeadAttention(emb_size, heads, 0.5), emb_size, 0.5) 
            for _ in range(depth)
        ])

class TCANet(nn.Module):
    def __init__(self, n_chans=22, out_features=4, n_times=1000, pooling_size=56, heads=2, depth=6):
        super().__init__()
        self.mseegnet = MSCNet(number_channel=n_chans, pooling_size=pooling_size)
        # input_dimension=48 (f1=16 * 3 scales)
        self.tcn_block = _TCNBlock(input_dimension=48, depth=2, kernel_size=4, filters=16, drop_prob=0.25)
        self.sa = TransformerEncoder(heads, depth, 16)
        self.flatten = nn.Flatten()
        self.classifier = nn.Linear(16 * (n_times // pooling_size), out_features)

    def forward(self, x: Tensor):
        x = self.mseegnet(x)
        x_tcn = self.tcn_block(x)
        x = self.sa(x_tcn)
        x = nn.Dropout(0.25)(x + x_tcn)
        feat = self.flatten(x)
        return self.classifier(feat), feat

class EEGTransformer(nn.Module):
    def __init__(self, n_chans=22, n_classes=4, n_times=1000, pooling_size=56):
        super().__init__()
        self.model = TCANet(n_chans=n_chans, out_features=n_classes, n_times=n_times, pooling_size=pooling_size)

    def forward(self, x):
        out, features = self.model(x)
        return out