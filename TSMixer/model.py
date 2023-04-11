import torch.nn as nn


class MLP_Time(nn.Module):
    """MLP for time embedding.

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
                 ts_length:  int,
                 embed_dim: int,
                 dropout: float = 0.1):
        super().__init__()
        self.time_mlp1 = nn.Sequential(
            nn.BatchNorm1d(in_channels),
            nn.Linear(ts_length, embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        self.time_mlp2 = nn.Sequential(
            nn.Linear(embed_dim, ts_length),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        x_time = self.time_mlp1(x.transpose(1, 2))
        return x + self.time_mlp2(x_time).transpose(1, 2)


class MLP_Feat(nn.Module):
    """MLPs for feature embedding.

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
        self.feat_mlp1 = nn.Sequential(
            nn.BatchNorm1d(ts_length),
            nn.Linear(in_channels, embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        self.feat_mlp2 = nn.Sequential(
            nn.Linear(embed_dim, in_channels),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        x_feat = self.feat_mlp1(x)
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
        self.mlp_time = MLP_Time(in_channels, ts_length, embed_dim, dropout)
        self.mlp_feat = MLP_Feat(in_channels, ts_length, embed_dim, dropout)

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
        - Algorithm 1 in https://arxiv.org/pdf/2303.06053.pdf
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
