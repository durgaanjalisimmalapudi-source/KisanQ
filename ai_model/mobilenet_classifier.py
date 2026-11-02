"""
MobileNetV2 Classical CNN and Feature Extractor for KisanQ.
Optimized for Raspberry Pi 5 edge deployment with lightweight parameters.
"""

import sys
from pathlib import Path
from typing import Tuple, Optional
import torch
import torch.nn as nn
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config


class LeafMobileNetV2(nn.Module):
    """
    MobileNetV2 tailored for leaf disease feature extraction and classification.
    """
    def __init__(self, num_classes: int = len(config.PRIMARY_CLASSES), pretrained: bool = True):
        super().__init__()
        weights = MobileNet_V2_Weights.DEFAULT if pretrained else None
        base_model = mobilenet_v2(weights=weights)

        # Feature extractor is the convolutional backbone + adaptive pool
        self.features = base_model.features
        self.pooling = nn.AdaptiveAvgPool2d((1, 1))
        
        # Penultimate embedding dimension for MobileNetV2 is 1280
        self.embedding_dim = 1280

        # Classical classification head
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(self.embedding_dim, 256),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(256, num_classes)
        )

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extracts 1280-dimensional feature vector for a batch of images.
        Used for PCA feature reduction and Quantum VQC pipeline.
        """
        x = self.features(x)
        x = self.pooling(x)
        x = torch.flatten(x, 1)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Standard end-to-end forward pass returning class logits."""
        feat = self.extract_features(x)
        logits = self.classifier(feat)
        return logits


def get_model(num_classes: int = len(config.PRIMARY_CLASSES), pretrained: bool = True, weights_path: Optional[Path] = None, device: str = "cpu") -> LeafMobileNetV2:
    """Instantiate and optionally load saved weights for MobileNetV2."""
    model = LeafMobileNetV2(num_classes=num_classes, pretrained=pretrained)
    if weights_path and Path(weights_path).exists():
        state_dict = torch.load(weights_path, map_location=device)
        model.load_state_dict(state_dict)
        print(f"[LeafMobileNetV2] Loaded weights from {weights_path}")
    model.to(device)
    return model


if __name__ == "__main__":
    model = get_model(pretrained=True)
    dummy_input = torch.randn(2, 3, 224, 224)
    features = model.extract_features(dummy_input)
    logits = model(dummy_input)
    print(f"Input shape: {dummy_input.shape}")
    print(f"Extracted features shape: {features.shape} (Expected: 2, 1280)")
    print(f"Logits shape: {logits.shape} (Expected: 2, {len(config.PRIMARY_CLASSES)})")
