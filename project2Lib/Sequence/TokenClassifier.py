import torch
from torch import nn
from transformers.modeling_outputs import SequenceClassifierOutput

from .SentenceClassifier import SentenceClassifier

class TokenClassifier(SentenceClassifier):

    def __init__(self, num_labels, token_size, trial):

        sentence_size = trial.params["sentence_size"] if trial else 128
        super().__init__(num_labels, sentence_size, trial)

        self.token_size = token_size

        self.lstm_token = nn.LSTM(
            input_size = self.token_size,
            hidden_size = self.hidden_size,
            num_layers = self.num_layers,
            batch_first = True,
            bidirectional = True
        )
        self.linear_token = nn.Linear(4 * self.hidden_size, self.sentence_size)

    def forward(
        self,
        labels,
        embeddings,
        lengths
    ):

        sentences = []
        B = len(embeddings)
        L = max(lengths)
        sentences = torch.zeros((B, L, self.sentence_size)).to(device=labels.device)

        for i, x in enumerate(embeddings):
            seq, _ = self.lstm_token(self.dropout(x.to(device=labels.device)))
            seq = torch.concat([seq[:, 0], seq[:, -1]], dim=1)
            sentences[i, :lengths[i]] = self.linear_token(self.dropout(seq))

        return super().forward(
            labels = labels,
            embeddings = sentences,
            lengths = lengths
        )