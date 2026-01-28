import torch
import torch.nn as nn
from torchvision import models


class LightweightMobileNetV2(nn.Module):
    """
    Lightweight MobileNetV2 model for breast cancer detection.
    Optimized for mobile and edge deployment.
    """
    
    def __init__(self, num_classes=2, pretrained=True):
        """
        Args:
            num_classes (int): Number of output classes
            pretrained (bool): Use ImageNet pretrained weights
        """
        super(LightweightMobileNetV2, self).__init__()
        
        # Load pretrained MobileNetV2
        self.model = models.mobilenet_v2(pretrained=pretrained)
        
        # Get the number of features in the final layer
        num_features = self.model.classifier[1].in_features
        
        # Replace the classifier
        self.model.classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(num_features, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        return self.model(x)


def create_mobilenet_stage1(pretrained=True):
    """
    Create MobileNetV2 Stage 1 model for Normal vs Sick classification.
    
    Returns:
        LightweightMobileNetV2: Model with 2 output classes
    """
    return LightweightMobileNetV2(num_classes=2, pretrained=pretrained)


def create_mobilenet_stage2(pretrained=True):
    """
    Create MobileNetV2 Stage 2 model for Benign vs Malignant classification.
    
    Returns:
        LightweightMobileNetV2: Model with 2 output classes
    """
    return LightweightMobileNetV2(num_classes=2, pretrained=pretrained)


if __name__ == "__main__":
    # Test model creation
    model = create_mobilenet_stage1()
    
    # Test forward pass
    dummy_input = torch.randn(1, 3, 224, 224)
    output = model(dummy_input)
    
    print(f"Model output shape: {output.shape}")
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters())}")
    
    # Calculate model size
    param_size = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
    size_mb = (param_size + buffer_size) / 1024**2
    print(f"Model size: {size_mb:.2f} MB")
