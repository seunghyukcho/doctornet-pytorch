import torch
import torch.hub as hub
import torch.nn as nn

class FeatureExtractor(nn.Module):

    def __init__(self):
        super(FeatureExtractor, self).__init__()
        self.model = hub.load('pytorch/vision:v0.9.0', 'inception_v3', pretrained=True)
        self.model.fc = nn.Identity()

    def forward(self, x):
        feature = self.model(x)
        feature = feature.logits
        return feature


class AveragingWeights(nn.Module):

    def __init__(self, n_annotators, weight_type='W', feature_dim=2048, bottleneck_dim=None):
        super(AveragingWeights, self).__init__()
        self.weight_type = weight_type
        if self.weight_type == 'W':
            self.weights = nn.Parameter(torch.randn(n_annotators), requires_grad=True)
        elif self.weight_type == 'I':
            if bottleneck_dim is None:
                self.weights = nn.Linear(feature_dim, n_annotators)
            else:
                self.weights = nn.Sequential(nn.Linear(feature_dim, bottleneck_dim), nn.Linear(bottleneck_dim, n_annotators))
        else:
            raise IndexError("weight type must be 'W' or 'I'.")

    def forward(self, feature):
        if self.weight_type == 'W':
            return self.weights
        else:
            return self.weights(feature).view(-1)


class DoctorNet(nn.Module):

    def __init__(self, n_classes, n_annotators, weight_type='W', feature_dim=2048, bottleneck_dim=None):
        super(DoctorNet, self).__init__()
        self.feature_extractor = FeatureExtractor()
        self.annotators = nn.Parameter(torch.stack([torch.randn(feature_dim, n_classes) for _ in range(n_annotators)]), requires_grad=True)
        self.weights = AveragingWeights(n_annotators, weight_type, feature_dim, bottleneck_dim)
        
    def forward(self, x, mask=None):
        feature = self.feature_extractor(x)
        decisions = torch.einsum('ik,jkl->ijl', feature, self.annotators)

        if mask is None:
            return decisions
        else:
            weights = self.weights(feature)
            predictions = decisions * weights[None, :, None]

            predictions = predictions.masked_fill(~mask[:, :, None], 0)
            predictions = torch.sum(predictions, axis=1)

            weights = weights.masked_fill(~mask, 0)
            weights = torch.sum(weights, axis=-1)

            predictions = predictions / weights[:, None]
            return predictions

