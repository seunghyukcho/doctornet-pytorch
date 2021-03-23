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
        if self.training:
            feature = feature.logits
        return feature


class Weights(nn.Module):

    def __init__(self, n_annotators, weight_type='W', feature_dim=2048, bottleneck_dim=None):
        super(Weights, self).__init__()
        self.weight_type = weight_type
        if self.weight_type == 'W':
            self.weights = nn.Parameter(torch.ones(n_annotators), requires_grad=True)
        elif self.weight_type == 'I':
            if bottleneck_dim is None:
                self.weights = nn.Linear(feature_dim, n_annotators)
            else:
                self.weights = nn.Sequential(nn.ReLU(), nn.Linear(feature_dim, bottleneck_dim), nn.Linear(bottleneck_dim, n_annotators))
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
        self.weights = Weights(n_annotators, weight_type, feature_dim, bottleneck_dim)
        
    def forward(self, x, pred=False, weight=False):
        feature = self.feature_extractor(x)
        decisions = torch.einsum('ik,jkl->ijl', feature, self.annotators)
        weights = self.weights(feature)
        if weight:
            decisions = decisions * weights[None, :, None]
        if pred:
            decisions = torch.sum(decisions, axis=1)
            return decisions
        else:
            return decisions, weights
 
