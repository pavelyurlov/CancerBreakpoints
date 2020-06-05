import torch
import torch.nn as nn
import torch.nn.functional as F

# CNN without ngrams
class CNNModel(nn.Module):
    def __init__(self, dropout_p=0.1):
        super(CNNModel, self).__init__()

        self.dropout1 = nn.Dropout(p=dropout_p)

        self.conv1 = nn.Conv1d(4, 8, 15, 2, padding=7)
        self.conv2 = nn.Conv1d(8, 16, 7, 2, padding=3)
        self.conv3 = nn.Conv1d(16, 32, 7, 2, padding=3)

        self.conv4 = nn.Conv1d(32, 16, 7, 2, padding=3)
        self.conv5 = nn.Conv1d(16, 8, 7, 2, padding=3)

        self.flatten = nn.Flatten()

        self.linear1 = nn.Linear(8 * 25 + 0, 32)
        self.dropout2 = nn.Dropout(p=dropout_p)
        self.linear2 = nn.Linear(32, 1)


    def forward(self, x_both):
        x, x_ngram = x_both
        x = x.permute(0, 2, 1)

        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.dropout1(x)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.dropout1(x)
        x = F.relu(self.conv5(x))

        x = self.flatten(x)

        x = F.relu(self.linear1(x))
        x = self.dropout2(x)
        x = self.linear2(x)

        x = x.view(-1)

        return x

# CNN with ngrams
class CNNModelNgrams(nn.Module):
    def __init__(self, dropout_p=0.1):
        super(CNNModelNgrams, self).__init__()

        self.dropout1 = nn.Dropout(p=dropout_p)

        self.conv1 = nn.Conv1d(4, 8, 15, 2, padding=7)
        self.conv2 = nn.Conv1d(8, 16, 7, 2, padding=3)
        self.conv3 = nn.Conv1d(16, 32, 7, 2, padding=3)

        self.conv4 = nn.Conv1d(32, 16, 7, 2, padding=3)
        self.conv5 = nn.Conv1d(16, 8, 7, 2, padding=3)

        self.flatten = nn.Flatten()

        self.linear1 = nn.Linear(8 * 25 + (4 + 16 + 64 + 256), 32)
        self.dropout2 = nn.Dropout(p=dropout_p)
        self.linear2 = nn.Linear(32, 1)


    def forward(self, x_both):
        x, x_ngram = x_both
        x = x.permute(0, 2, 1)

        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.dropout1(x)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.dropout1(x)
        x = F.relu(self.conv5(x))

        x = self.flatten(x)

        x = torch.cat([x, x_ngram], dim=1)

        x = F.relu(self.linear1(x))
        x = self.dropout2(x)
        x = self.linear2(x)

        x = x.view(-1)

        return x


# CNN + RNN without ngrams
class RNNModel(nn.Module):
    def __init__(self, dropout_p=0.1):
        super(RNNModel, self).__init__()

        self.dropout1 = nn.Dropout(p=dropout_p)

        self.conv1 = nn.Conv1d(4, 8, 15, 2, padding=7)
        self.conv2 = nn.Conv1d(8, 16, 7, 2, padding=3)
        self.conv3 = nn.Conv1d(16, 32, 7, 2, padding=3)

        self.lstm = nn.LSTM(32, 16, 1, bidirectional=True, dropout=dropout_p)

        self.conv4 = nn.Conv1d(32, 16, 7, 2, padding=3)
        self.conv5 = nn.Conv1d(16, 8, 7, 2, padding=3)

        self.flatten = nn.Flatten()

        self.linear1 = nn.Linear(8 * 25 + 0, 32)
        self.dropout2 = nn.Dropout(p=dropout_p)
        self.linear2 = nn.Linear(32, 1)


    def forward(self, x_both):
        x, x_ngram = x_both
        x = x.permute(0, 2, 1)

        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.dropout1(x)
        x = F.relu(self.conv3(x))

        x = x.permute(2, 0, 1)
        x, _ = self.lstm(x)
        x = x.permute(1, 2, 0)

        x = F.relu(self.conv4(x))
        x = self.dropout1(x)
        x = F.relu(self.conv5(x))

        x = self.flatten(x)

        x = F.relu(self.linear1(x))
        x = self.dropout2(x)
        x = self.linear2(x)

        x = x.view(-1)

        return x

# CNN + RNN with ngrams
class RNNModelNgrams(nn.Module):
    def __init__(self, dropout_p=0.1):
        super(RNNModelNgrams, self).__init__()

        self.dropout1 = nn.Dropout(p=dropout_p)

        self.conv1 = nn.Conv1d(4, 8, 15, 2, padding=7)
        self.conv2 = nn.Conv1d(8, 16, 7, 2, padding=3)
        self.conv3 = nn.Conv1d(16, 32, 7, 2, padding=3)

        self.lstm = nn.LSTM(32, 16, 1, bidirectional=True, dropout=dropout_p)

        self.conv4 = nn.Conv1d(32, 16, 7, 2, padding=3)
        self.conv5 = nn.Conv1d(16, 8, 7, 2, padding=3)

        self.flatten = nn.Flatten()

        self.linear1 = nn.Linear(8 * 25 + (4 + 16 + 64 + 256), 32)
        self.dropout2 = nn.Dropout(p=dropout_p)
        self.linear2 = nn.Linear(32, 1)


    def forward(self, x_both):
        x, x_ngram = x_both
        x = x.permute(0, 2, 1)

        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.dropout1(x)
        x = F.relu(self.conv3(x))

        x = x.permute(2, 0, 1)
        x, _ = self.lstm(x)
        x = x.permute(1, 2, 0)

        x = F.relu(self.conv4(x))
        x = self.dropout1(x)
        x = F.relu(self.conv5(x))

        x = self.flatten(x)
        x = torch.cat([x, x_ngram], axis=-1)

        x = F.relu(self.linear1(x))
        x = self.dropout2(x)
        x = self.linear2(x)

        x = x.view(-1)

        return x


# Positional Encoding for Transformer
import math
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=1000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

# Transformer without ngrams
class TransformerModel(nn.Module):
    def __init__(self, p_dropout=0.1):
        super(TransformerModel, self).__init__()

        hidden_dim = 32

        self.embed = nn.Conv1d(4, hidden_dim, 7, 2, 3)
        self.pos_encoder = PositionalEncoding(hidden_dim)
        self.encoder_layer = nn.TransformerEncoderLayer(hidden_dim,
                                                        nhead=4,
                                                        dim_feedforward=hidden_dim,
                                                        dropout=p_dropout)
        self.encoder = nn.TransformerEncoder(self.encoder_layer,
                                             num_layers=2)

        self.conv1 = nn.Conv1d(hidden_dim, 8, 7, 2, 3)
        self.conv2 = nn.Conv1d(8, 4, 7, 2, 3)

        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(p=p_dropout)

        self.linear = nn.Linear(4 * 100 + 0, 16)
        self.linear2 = nn.Linear(16, 1)


    def forward(self, x_both):
        x, x_ngram = x_both

        x = x.permute(0, 2, 1)
        x = F.relu(self.embed(x))
        x = x.permute(0, 2, 1)
        x = self.pos_encoder(x)
        x = self.encoder(x)

        x = x.permute(0, 2, 1)
        x = F.relu(self.conv1(x))
        x = self.dropout(x)
        x = F.relu(self.conv2(x))

        x = self.flatten(x)

        x = F.relu(self.linear(x))
        x = self.dropout(x)
        x = self.linear2(x)

        x = x.view(-1)

        return x

# Transformer with ngrams
class TransformerModelNgrams(nn.Module):
    def __init__(self, p_dropout=0.1):
        super(TransformerModelNgrams, self).__init__()

        hidden_dim = 32

        self.embed = nn.Conv1d(4, hidden_dim, 7, 2, 3)
        self.pos_encoder = PositionalEncoding(hidden_dim)
        self.encoder_layer = nn.TransformerEncoderLayer(hidden_dim,
                                                        nhead=4,
                                                        dim_feedforward=hidden_dim,
                                                        dropout=p_dropout)
        self.encoder = nn.TransformerEncoder(self.encoder_layer,
                                             num_layers=2)

        self.conv1 = nn.Conv1d(hidden_dim, 8, 7, 2, 3)
        self.conv2 = nn.Conv1d(8, 4, 7, 2, 3)

        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(p=p_dropout)

        self.linear = nn.Linear(4 * 100 + (4 + 16 + 64 + 256), 16)
        self.linear2 = nn.Linear(16, 1)


    def forward(self, x_both):
        x, x_ngram = x_both

        x = x.permute(0, 2, 1)
        x = F.relu(self.embed(x))
        x = x.permute(0, 2, 1)
        x = self.pos_encoder(x)
        x = self.encoder(x)

        x = x.permute(0, 2, 1)
        x = F.relu(self.conv1(x))
        x = self.dropout(x)
        x = F.relu(self.conv2(x))

        x = self.flatten(x)

        x = torch.cat([x, x_ngram], dim=1)
        x = F.relu(self.linear(x))
        x = self.dropout(x)
        x = self.linear2(x)

        x = x.view(-1)

        return x

