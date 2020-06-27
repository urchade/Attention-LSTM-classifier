from torch import nn
import torch.nn.functional as F
import torch
from transformers import AutoModel


class AttentionBiLSTM(nn.Module):
    def __init__(self, n_classes, num_embeddings, embedding_dim,
                 weight, hidden_size, rnn_dropout=0.1, att_dropout=0.3, cls_dropout=0.3, rnn_type='LSTM', num_heads=1):
        super().__init__()

        self.rnn_type = rnn_type

        self.embedding = nn.Embedding(num_embeddings, embedding_dim, padding_idx=0, _weight=weight)

        if self.rnn_type == 'LSTM':
            self.rnn = nn.LSTM(embedding_dim, hidden_size, batch_first=True,
                               dropout=rnn_dropout, bidirectional=True)

        if self.rnn_type == 'GRU':
            self.rnn = nn.GRU(embedding_dim, hidden_size, batch_first=True,
                               dropout=rnn_dropout, bidirectional=True)

        self.biLSTM_hidden = 2 * hidden_size
        self.query = nn.Parameter(torch.randn(size=(self.biLSTM_hidden,)))

        self.mha = nn.MultiheadAttention(embed_dim=self.biLSTM_hidden, num_heads=num_heads, dropout=att_dropout)

        self.classifier = nn.Sequential(nn.ReLU(),
                                        nn.Dropout(cls_dropout),
                                        nn.Linear(self.biLSTM_hidden, n_classes))

    def forward(self, x, mask=None):
        # dim of x: (batch_size, seq_len)
        out_emb = self.embedding(x)  # (batch_size, seq_len, embedding_dim)

        # dim of out:  (batch_size, seq_len, 2 * hidden_size)
        out, *_ = self.rnn(out_emb)

        N, L, E = out.size()

        Q = self.query.expand(size=(1, N, E))  # Same query for all batches
        V = out.transpose(0, 1)  # (seq_len, batch_size, 2 * hidden_size)
        K = out.transpose(0, 1)  # (seq_len, batch_size, 2 * hidden_size)

        att_out, _ = self.mha(Q, K, V, key_padding_mask=mask)

        # (1, batch_size, 2 * hidden_size)
        att_out = att_out.squeeze()  # (batch_size, 2 * hidden_size)

        out_linear = self.classifier(att_out)  # (batch_size, 2 * n_classes)

        return F.log_softmax(out_linear, dim=1)

    def predict(self, x, logits=False):
        if not logits:
            x = self.forward(x)
        return [torch.argmax(i).item() for i in x]


class BertClassifier(nn.Module):
    def __init__(self, model_name, n_classes=12, num_layers=1, n_units=768,
                 activation=nn.ReLU(), dropout_rate=0.0, freeze_bert=False):
        super().__init__()

        self.bert = AutoModel.from_pretrained(model_name)
        self.hidden_size = self.bert.config.hidden_size

        # freeze bert parameters
        if freeze_bert:
            for p in self.bert.parameters():
                p.requires_grad = False

        modules = []
        first_hidden = self.hidden_size

        for i in range(num_layers):

            if i == num_layers - 1:
                modules.append(nn.Linear(first_hidden, n_classes))
                continue

            linear = nn.Linear(first_hidden, n_units)
            dropout = nn.Dropout(dropout_rate)
            block = nn.Sequential(linear, activation, dropout)
            modules.append(block)

            first_hidden = n_units

        self.classifier_block = nn.Sequential(*modules)

    def forward(self, x, mask=None):

        h = self.bert(x, attention_mask=mask)[0]  # (batch_size, seq_len, hidden_size)
        h = h.mean(1)
        y = self.classifier_block(h)  # (batch_size, n_classes)
        return F.log_softmax(y, dim=1)

    @torch.no_grad()
    def predict(self, x, logits=False):
        if not logits:
            x = self.forward(x)
        return [torch.argmax(i).item() for i in x]
