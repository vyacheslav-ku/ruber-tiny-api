from torch import nn


class YesNoModel(nn.Module):
    def set_config(self, config):
        self.config = config

    def predict_group(self, group, txt):
        for t in self.config.get(group, []):
            if t in txt or txt in t:
                return 1
        return 0

    def __init__(self):
        super().__init__()
        # 2. Create 2 nn.Linear layers capable of handling X and y input and output shapes
        self.layer_1 = nn.Linear(in_features=312, out_features=5)  # takes in 2 features (X), produces 5 features
        self.layer_2 = nn.Linear(in_features=5, out_features=1)  # takes in 5 features, produces 1 feature (y)

    # 3. Define a forward method containing the forward pass computation
    def forward(self, x):
        # Return the output of layer_2, a single feature, the same shape as y
        return self.layer_2(
            self.layer_1(
                x))  # computation goes through layer_1 first then the output of layer_1 goes through layer_2
