import torch
from torch import nn
from transformers.modeling_outputs import SequenceClassifierOutput

class SentenceClassifier(nn.Module):

    def __init__(self, num_labels, sentence_size, trial):

        super().__init__()

        self.num_labels = num_labels
        self.sentence_size = sentence_size
        self.hidden_size = trial.params["hidden_size"] if trial else 128
        self.dropout_p = trial.params["dropout_p"] if trial else 0.5
        self.num_layers = trial.params["num_layers"] if trial else 1

        if trial:
            print(trial.params)

        self.lstm = nn.LSTM(
            input_size = self.sentence_size,
            hidden_size = self.hidden_size,
            num_layers = self.num_layers,
            batch_first = True,
            bidirectional = True
        )
        self.dropout = nn.Dropout(p = self.dropout_p)
        self.linear = nn.Linear(2 * self.hidden_size, self.num_labels)

    def forward(
        self,
        labels,
        embeddings,
        lengths
    ):

        B = len(lengths)

        logits = self.classifier(embeddings)
        logits = torch.concat([logits[i, :lengths[i]] for i in range(B)])

        loss = nn.CrossEntropyLoss()(logits.view(-1, self.num_labels), labels.view(-1))

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
        )

    def classifier(self, embeddings):
        seq, _ = self.lstm(self.dropout(embeddings))
        shape = seq.shape[:2] + (-1,)
        logits = self.linear(self.dropout(seq.flatten(0, 1)))
        return logits.view(shape)
