import torch
import torch.nn as nn
from torchvision import models


class LightweightResNet18(nn.Module):
    """
    Lightweight ResNet18 model for breast cancer detection.
    Modified for binary and multi-class classification.
    """
    
    def __init__(self, num_classes=2, pretrained=True):
        """
        Args:
            num_classes (int): Number of output classes (2 for binary, 3 for multi-class)
            pretrained (bool): Use ImageNet pretrained weights
        """
        super(LightweightResNet18, self).__init__()
        
        # Load pretrained ResNet18
        self.model = models.resnet18(pretrained=pretrained)
        
        # Get the number of features in the final layer
        num_features = self.model.fc.in_features
        
        # Replace the final fully connected layer
        self.model.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        return self.model(x)


def create_stage1_model(pretrained=True):
    """
    Create Stage 1 model for Normal vs Sick classification.
    
    Returns:
        LightweightResNet18: Model with 2 output classes
    """
    return LightweightResNet18(num_classes=2, pretrained=pretrained)


def create_stage2_model(pretrained=True):
    """
    Create Stage 2 model for Benign vs Malignant classification.
    
    Returns:
        LightweightResNet18: Model with 2 output classes
    """
    return LightweightResNet18(num_classes=2, pretrained=pretrained)


if __name__ == "__main__":
    # Test model creation
    model = create_stage1_model()
    
    # Test forward pass
    dummy_input = torch.randn(1, 3, 224, 224)
    output = model(dummy_input)
    
    print(f"Model output shape: {output.shape}")
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters())}")
