import torch.nn as nn

class MLP_Time(nn.Module):
    """MLP for time embedding.

    :argument
        - ts_length (int): time series length
        - dropout (float): dropout rate

    :return
        - x (tensor): output tensor of shape (ts_length, in_channels)
    """

    def __init__(self, ts_length, dropout=0.1):
        super().__init__()
        self.time_mlp = nn.Sequential(
            nn.BatchNorm1d(ts_length),
            nn.Linear(ts_length, ts_length),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        x_time = self.time_mlp(x.transpose(0,1))
        return x + x_time.transpose(0,1) # not sure if we need a residual connection here. The paper doesn't mention it.


class MLP_Feat(nn.Module):
    """MLPs for feature embedding.

    :argument
        - in_channels (int): input channels
        - embed_dim (int): embedding dimension
        - dropout (float): dropout rate

    :return
        - x (tensor): output tensor of shape (ts_length, in_channels)
    """

    def __init__(self, in_channels, embed_dim, dropout=0.1):
        super().__init__()
        self.feat_mlp1 = nn.Sequential(
            nn.BatchNorm1d(in_channels),
            nn.Linear(in_channels, embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        self.feat_mlp2 = nn.Sequential(
            nn.Linear(embed_dim, in_channels),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        u = self.feat_mlp1(x)
        return x + self.feat_mlp2(u)

class Mixer_Block(nn.Module):
    def __init__(self, ts_length, in_channels, embed_dim, dropout=0.1):
        """Mixer block.

        :argument
            - in_channels (int): input channels
            - ts_length (int): time series length
            - embed_dim (int): embedding dimension
            - dropout (float): dropout rate

        :return
            - x (tensor): output tensor of shape (ts_length, in_channels)
        """
        super().__init__()
        self.mlp_time = MLP_Time(ts_length, dropout)
        self.mlp_feat = MLP_Feat(in_channels, embed_dim, dropout)

    def forward(self, x):
        x = self.mlp_time(x)
        x = self.mlp_feat(x)
        return x

class TS_Mixer(nn.Module):
    def __init__(self, ts_length, in_channels, embed_dim, num_blocks, fcst_h, dropout=0.1):
        """Time Series Mixer.

        :argument
            - ts_length (int): time series length
            - in_channels (int): input channels
            - embed_dim (int): embedding dimension
            - num_blocks (int): number of mixer blocks
            - fcst_h (int): forecast horizon
            - dropout (float): dropout rate

        :return
            - x (tensor): output tensor of shape (fcst_h, in_channels)
        """
        super().__init__()
        self.mixer_blocks = nn.Sequential(*[
            Mixer_Block(ts_length, in_channels, embed_dim, dropout) for _ in range(num_blocks)
        ])
        self.fc = nn.Linear(ts_length, fcst_h)

    def forward(self, x):
        x = self.mixer_blocks(x)
        x = self.fc(x.transpose(0,1))
        return x.transpose(0,1)
