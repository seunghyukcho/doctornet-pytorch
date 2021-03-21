import torch
import torch.hub as hub
import torch.nn as nn

class FeatureExtractor(nn.Module):

    def __init__(self):
        super(FeatureExtractor, self).__init__()
        self.model = hub.load('pytorch/vision:v0.9.0', 'inception_v3', pretrained=True)
        self.model.fc = nn.Identity()

    def __forward__(self, x):
        feature = self.model(x)
        return feature


class DoctorNet(nn.Module):

    def __init__(self, n_classes, n_annotators, feature_dim=2048):
        super(DoctorNet, self).__init__()
        self.feature_extractor = FeatureExtractor()
        self.annotators = nn.Parameter(torch.stack([torch.randn(feature_dim, n_classes) for _ in range(n_annotators)]), requires_grad=True)
        
    def __forward__(self, x):
        feature = self.feature_extractor(x)
        decisions = torch.einsum('ik,jkl->ijl', feature, self.annotators)

        return decisions

