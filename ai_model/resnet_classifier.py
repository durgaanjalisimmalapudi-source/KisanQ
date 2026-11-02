"""
ResNet18 Classical CNN and Feature Extractor for KisanQ.
Upgraded architecture replacing MobileNetV2 with ResNet18 residual connections.
Penultimate feature extraction dimension is 512.
"""

import sys
from pathlib import Path
from typing import Optional, List
import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config


class LeafResNet18(nn.Module):
    """
    ResNet18 tailored for agricultural disease feature extraction and classification.
    """
    def __init__(self, num_classes: int = len(config.PRIMARY_CLASSES), pretrained: bool = True):
        super().__init__()
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        base = resnet18(weights=weights)

        # Feature extractor: all layers up to avgpool
        self.conv1 = base.conv1
        self.bn1 = base.bn1
        self.relu = base.relu
        self.maxpool = base.maxpool
        self.layer1 = base.layer1
        self.layer2 = base.layer2
        self.layer3 = base.layer3
        self.layer4 = base.layer4
        self.avgpool = base.avgpool

        # Penultimate embedding dimension for ResNet18 is 512
        self.embedding_dim = 512

        # Classical classification head
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(self.embedding_dim, 128),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(128, num_classes)
        )

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extracts 512-dimensional feature vector for a batch of images.
        Used for PCA feature reduction and Quantum VQC pipeline.
        """
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Standard end-to-end forward pass returning class logits."""
        feats = self.extract_features(x)
        logits = self.classifier(feats)
        return logits


def get_resnet_model(
    num_classes: int = len(config.PRIMARY_CLASSES),
    pretrained: bool = True,
    weights_path: Optional[Path] = None,
    device: str = "cpu"
) -> LeafResNet18:
    """Instantiate and optionally load saved weights for ResNet18."""
    model = LeafResNet18(num_classes=num_classes, pretrained=pretrained)
    if weights_path and Path(weights_path).exists():
        state_dict = torch.load(weights_path, map_location=device)
        model.load_state_dict(state_dict)
        print(f"[LeafResNet18] Loaded weights from {weights_path}")
    model.to(device)
    return model


if __name__ == "__main__":
    model = get_resnet_model(pretrained=True)
    dummy_input = torch.randn(2, 3, 224, 224)
    features = model.extract_features(dummy_input)
    logits = model(dummy_input)
    print(f"ResNet18 Input shape: {dummy_input.shape}")
    print(f"Extracted features shape: {features.shape} (Expected: 2, 512)")
    print(f"Logits shape: {logits.shape}")
