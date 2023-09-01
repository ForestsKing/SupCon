from torch import nn


class Classifier(nn.Module):
    def __init__(self, args, encoder):
        super(Classifier, self).__init__()

        self.encoder = encoder
        self.fc = nn.Linear(128, 10)

        if args.loss != 'CrossEntropy':
            for name, param in self.encoder.named_parameters():
                param.requires_grad = False

    def forward(self, x):
        z = self.encoder(x)
        y = self.fc(z)
        return y
