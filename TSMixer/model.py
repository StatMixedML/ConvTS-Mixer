import torch.nn as nn


class MLP_Time(nn.Module):
    """MLP for time embedding. According to the paper, the authors employ a single layer perceptron.

    :argument
        - ts_length (int): time series length
        - dropout (float): dropout rate
    :return
        - x (tensor): output tensor of shape (batch_size, ts_length, in_channels)
    """

    def __init__(self,
                 ts_length: int,
                 dropout: float = 0.1):
        super().__init__()

        # BatchNorm1d is applied to the time dimension
        self.bn = nn.BatchNorm1d(ts_length)

        # MLP for time embedding
        self.time_mlp = nn.Sequential(
            nn.Linear(ts_length, ts_length),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        x_norm = self.bn(x)
        x_time = self.time_mlp(x_norm.transpose(1, 2)).transpose(1, 2)
        return x + x_time  # not sure if we need a residual connection here, the paper doesn't mention it.


class MLP_Feat(nn.Module):
    """MLPs for feature embedding.

    :argument
        - in_channels (int): input channels
        - embed_dim (int): embedding dimension
        - dropout (float): dropout rate, default 0.1

    :return
        - x (tensor): output tensor of shape (batch_size, ts_length, in_channels)
    """
    def __init__(self,
                 in_channels: int,
                 embed_dim: int,
                 dropout: float = 0.1):
        super().__init__()

        # BatchNorm1d is applied to the feature dimension
        self.batch_norm = nn.BatchNorm1d(in_channels)

        # MLPs for feature embedding
        self.feat_mlp1 = nn.Sequential(
            nn.Linear(in_channels, embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        self.feat_mlp2 = nn.Sequential(
            nn.Linear(embed_dim, in_channels),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        x_norm = self.batch_norm(x.transpose(1, 2)).transpose(1, 2)
        x_feat = self.feat_mlp1(x_norm)
        return x + self.feat_mlp2(x_feat)


class Mixer_Block(nn.Module):
    """Mixer block.

    :argument
        - in_channels (int): input channels
        - ts_length (int): time series length
        - embed_dim (int): embedding dimension
        - dropout (float): dropout rate, default 0.1

    :return
        - x (tensor): output tensor of shape (batch_size, ts_length, in_channels)
    """
    def __init__(self,
                 in_channels: int,
                 ts_length: int,
                 embed_dim: int,
                 dropout: float = 0.1):

        super().__init__()
        self.mlp_time = MLP_Time(ts_length, dropout)
        self.mlp_feat = MLP_Feat(in_channels, embed_dim, dropout)

    def forward(self, x):
        x = self.mlp_time(x)
        x = self.mlp_feat(x)
        return x


class TS_Mixer(nn.Module):
    """Time Series Mixer.

    :argument
        - in_channels (int): input channels
        - ts_length (int): time series length
        - embed_dim (int): embedding dimension
        - num_blocks (int): number of mixer blocks
        - fcst_h (int): forecast horizon
        - dropout (float): dropout rate, default 0.1

    :return
        - x (tensor): output tensor of shape (batch_size, fcst_h, in_channels)

    : source
        - Algorithm 1 in [TSMixer: An all-MLP Architecture for Time Series Forecasting] (https://arxiv.org/pdf/2303.06053.pdf)
    """
    def __init__(self,
                 in_channels: int,
                 ts_length: int,
                 embed_dim: int,
                 num_blocks: int,
                 fcst_h: int,
                 dropout: float = 0.1):
        super().__init__()
        self.mixer_blocks = nn.Sequential(*[
            Mixer_Block(in_channels, ts_length, embed_dim, dropout) for _ in range(num_blocks)
        ])
        self.fc = nn.Linear(ts_length, fcst_h)

    def forward(self, x):
        x = self.mixer_blocks(x)
        x = self.fc(x.transpose(1, 2))
        return x.transpose(1, 2)
