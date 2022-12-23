import torch.nn as nn
from utils.transformers import TransformerClassifier
from utils.tokenizer import VideoTokenizer

class CCT(nn.Module):
    def __init__(self,
                 embedding_dim=512,
                 dropout=0.,
                 attention_dropout=0.1,
                 stochastic_depth=0.1,
                 num_layers=14,
                 num_heads=8,
                 mlp_ratio=4.0,
                 num_classes=2,
                 positional_embedding='sine',
                 *args, **kwargs):
        super(CCT, self).__init__()

        self.tokenizer = VideoTokenizer()

        self.classifier = TransformerClassifier(
            sequence_length=self.tokenizer.sequence_length(),
            embedding_dim=embedding_dim,
            seq_pool=True,
            dropout=dropout,
            attention_dropout=attention_dropout,
            stochastic_depth=stochastic_depth,
            num_layers=num_layers,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            num_classes=num_classes,
            positional_embedding=positional_embedding
        )
        # Added by Shubham for binary classifier
        self.fc_sigmoid = nn.Sequential(nn.Linear(num_classes, 1), nn.Sigmoid())

    def forward(self, x):
        x = self.tokenizer(x)
        x = self.classifier(x)
        return self.fc_sigmoid(x)
