import torch
import torch.nn as nn
from .model import MLP_Time


class MLP_Feat(nn.Module):
    """MLP on feature domain.

    :argument
        - n_feat (int): number of input features
        - embed_dim (int): embedding dimension
        - dropout (float): dropout rate,

    :return
        - x (tensor): output tensor of shape (batch_size, fcst_h, embed_dim)
    """
    def __init__(self,
                 n_feat: int,
                 embed_dim: int,
                 dropout: float = 0.1):
        super().__init__()
        self.fc_s1 = nn.Sequential(
            nn.Linear(n_feat, embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        self.fc_s2 = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.Dropout(dropout)
        )

        # For cases where the input and the output dimensions are different, we apply an additional
        # linear transformation on the residual connection.
        self.projector = nn.Linear(n_feat, embed_dim) if n_feat != embed_dim else None

        # Batch normalization
        self.bn = nn.BatchNorm1d(embed_dim)

    def forward(self, x):
        v = self.fc_s1(x)
        u = self.fc_s2(v)
        h = x if self.projector is None else self.projector(x)
        out = self.bn((h + u).transpose(1, 2)).transpose(1, 2)
        return out


# class MLP_covariate(nn.Module):
#     """MLP for X and Z covariates.
#
#     :argument
#         - n_feat (int): number of features
#         - embed_dim (int): embedding dimension
#         - dropout (float): dropout rate
#
#     :return
#         - x (tensor): output tensor of shape (batch_size, ts_length, n_feat)
#     """
#
#     def __init__(self, n_feat, fcst_h, embed_dim, dropout=0.1):
#         super().__init__()
#         self.cov_mlp1 = nn.Sequential(
#             nn.BatchNorm1d(fcst_h),
#             nn.Linear(n_feat, embed_dim),
#             nn.ReLU(),
#             nn.Dropout(dropout)
#         )
#
#         self.cov_mlp2 = nn.Sequential(
#             nn.Linear(embed_dim, n_feat),
#             nn.Dropout(dropout)
#         )
#
#     def forward(self, x):
#         x_feat = self.cov_mlp1(x)
#         return x + self.cov_mlp2(x_feat)


class Mixer_Block(nn.Module):
    """Mixer block.

    :argument
        - n_feat (int): number of input features
        - n_static_feat (int): number of static features
        - fcst_h (int): forecast horizon
        - embed_dim (int): embedding dimension
        - dropout (float): dropout rate, default 0.1

    :return
        - x (tensor): output tensor of shape (batch_size, fcst_h, embed_dim)
    """
    def __init__(self,
                 n_feat: int,
                 n_static_feat: int,
                 fcst_h: int,
                 embed_dim: int,
                 dropout: float = 0.1):
        super().__init__()

        self.mlp_time = MLP_Time(n_feat, fcst_h, embed_dim, dropout)
        self.mlp_s = MLP_Feat(n_static_feat, embed_dim, dropout)
        self.mlp_feat = MLP_Feat(embed_dim*3, embed_dim, dropout)

    def forward(self, x, s):
        x = self.mlp_time(x)
        out = self.mlp_feat(torch.cat([x, self.mlp_s(s)], dim=2))
        return out


class TS_Mixer_auxiliary(nn.Module):
    """Time Series Mixer with auxiliary static and dynamic features.

    :argument
        - n_ts (int): number of input time series
        - n_static_feat (int): number of static features
        - n_dynamic_feat (int): number of dynamic features
        - ts_length (int): time series length
        - embed_dim (int): embedding dimension
        - num_blocks (int): number of mixer blocks
        - fcst_h (int): forecast horizon
        - out_dim (int): output dimension
        - dropout (float): dropout rate, default 0.1

    :return
        - x (tensor): output tensor of shape (batch_size, fcst_h, out_dim)

    source:
        - Algorithm 2 in https://arxiv.org/pdf/2303.06053.pdf
    """
    def __init__(self,
                 n_ts: int,
                 n_static_feat: int,
                 n_dynamic_feat: int,
                 ts_length: int,
                 embed_dim: int,
                 num_blocks: int,
                 fcst_h: int,
                 out_dim: int,
                 dropout: float = 0.1):
        super().__init__()

        #  Number of features for sx and sz
        n_feat_sx = embed_dim + n_ts
        n_feat_sz = embed_dim + n_dynamic_feat

        # MLP that maps the length of the input time series to fcst_h
        self.fc_map = nn.Linear(ts_length, fcst_h)

        # MLPs, conditioned on static features, that map X and Z to embedding space
        self.mlp_sx = MLP_Feat(n_static_feat, embed_dim, dropout)
        self.mlp_sz = MLP_Feat(n_static_feat, embed_dim, dropout)
        self.mlp_x = MLP_Feat(n_feat_sx, embed_dim, dropout)
        self.mlp_z = MLP_Feat(n_feat_sz, embed_dim, dropout)

        # Mixer blocks
        self.mixer_blocks = nn.Sequential(*[
            Mixer_Block(embed_dim*2,
                        n_static_feat,
                        fcst_h,
                        embed_dim,
                        dropout)
            for _ in range(num_blocks)
        ])

        # MLP that maps the output of the mixer blocks to the output dimension
        self.mlp_out = nn.Linear(embed_dim, out_dim)

        # Layer normalization
        self.layer_norm = nn.LayerNorm(embed_dim)

    def forward(self, x, z, s):

        # X: historical data
        # Z: future time-varying features
        # S: static features

        x = self.fc_map(x.transpose(1, 2)).transpose(1, 2)
        x_prime = self.mlp_x(torch.cat([x, self.mlp_sx(s)], dim=2))
        z_prime = self.mlp_z(torch.cat([z, self.mlp_sz(s)], dim=2))
        y_prime = torch.cat([x_prime, z_prime], dim=2)
        print(f"y_prime: {y_prime.shape}")
        y_prime = self.mixer_blocks(y_prime, s)
        print(f"mixer_block: {y_prime.shape}")
        out = self.layer_norm(self.mlp_out(y_prime))
        return out
