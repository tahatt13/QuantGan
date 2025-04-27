import torch.nn as nn
from torch.nn.utils import weight_norm

class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size]


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.PReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.PReLU()
        self.dropout2 = nn.Dropout(dropout)

        if padding == 0:
            self.net = nn.Sequential(self.conv1, self.relu1, self.dropout1, self.conv2, self.relu2, self.dropout2)
        else:
            self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1, self.conv2, self.chomp2, self.relu2, self.dropout2)

        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.PReLU()


    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return out, self.relu(out + res)


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.tcn = nn.ModuleList([TemporalBlock(3, 80, kernel_size=1, stride=1, dilation=1, padding=0),
                                 *[TemporalBlock(80, 80, kernel_size=2, stride=1, dilation=i, padding=i) for i in [1, 2, 4, 8, 16, 32]]])
        self.last = nn.Conv1d(80, 1, kernel_size=1, stride=1, dilation=1)

    def forward(self, x):
        skip_layers = []
        for layer in self.tcn:
            skip, x = layer(x)
            skip_layers.append(skip)
        x = self.last(x + sum(skip_layers))
        return x

class Discriminator(nn.Module):
    def __init__(self, seq_len):
        super(Discriminator, self).__init__()
        self.tcn = nn.ModuleList([TemporalBlock(1, 80, kernel_size=1, stride=1, dilation=1, padding=0),
                                 *[TemporalBlock(80, 80, kernel_size=2, stride=1, dilation=i, padding=i) for i in [1, 2, 4, 8, 16, 32]]])
        self.last = nn.Conv1d(80, 1, kernel_size=1, dilation=1)
        self.to_prob = nn.Sequential(nn.Linear(seq_len, 1), nn.Sigmoid())

    def forward(self, x):
        skips = []
        for layer in self.tcn:
            skip, x = layer(x)
            skips.append(skip)
        x = self.last(x + sum(skips))
        return self.to_prob(x).squeeze()



class CSVNNGenerator(nn.Module):
    def __init__(self, latent_dim=3, seq_len=127):
        super().__init__()
        self.latent_dim = latent_dim
        self.seq_len = seq_len

        # TCN pour sigma et mu
        self.tcn = nn.ModuleList([
            TemporalBlock(latent_dim, 50, kernel_size=1, stride=1, dilation=1, padding=0),
            *[TemporalBlock(50, 50, kernel_size=2, stride=1, dilation=d, padding=d) for d in [1, 2, 4, 8, 16, 32]]
        ])
        self.last_tcn = nn.Conv1d(50, 2, kernel_size=1)

        # MLP pour epsilon_t à CHAQUE timestep
        self.mlp = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        batch_size, latent_dim, total_len = x.shape

        # 1) Sigma, mu : sur x[:, :, :-1]
        tcn_input = x[:, :, :-1]  # (batch_size, latent_dim, total_len-1)
        skips = []
        for layer in self.tcn:
            skip, tcn_input = layer(tcn_input)
            skips.append(skip)
        sigma_mu = self.last_tcn(tcn_input)  # (batch_size, 2, total_len-1)

        sigma = sigma_mu[:, 0, :]  # (batch_size, total_len-1)
        mu = sigma_mu[:, 1, :]     # (batch_size, total_len-1)

        # 2) Epsilon : MLP appliqué à CHAQUE timestep

        # Pour ça on reshape
        mlp_input = x[:, :, :-1]  # on veut aussi Z_0...Z_{T-1}, shape (batch_size, latent_dim, total_len-1)
        mlp_input = mlp_input.permute(0, 2, 1)  # (batch_size, total_len-1, latent_dim)

        epsilon = self.mlp(mlp_input)  # (batch_size, total_len-1, 1)
        epsilon = epsilon.squeeze(-1)  # (batch_size, total_len-1)

        return sigma, mu, epsilon
