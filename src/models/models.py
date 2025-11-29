import torch
import torch.nn as nn
from typing import Optional


class AlexNet(nn.Module):
    """
    AlexNet architecture for Alzheimer's MRI classification.
    
    Modified for grayscale images and adapted for medical imaging.
    Original paper: Krizhevsky et al., 2012
    """
    
    def __init__(self, num_classes: int = 4, input_channels: int = 1, dropout_rate: float = 0.5):
        """
        Initialize AlexNet.
        
        Args:
            num_classes: Number of output classes (default: 4 for Alzheimer's stages)
            input_channels: Number of input channels (1 for grayscale, 3 for RGB)
            dropout_rate: Dropout probability for regularization
        """
        super(AlexNet, self).__init__()
        
        self.num_classes = num_classes
        self.input_channels = input_channels
        self.dropout_rate = dropout_rate
        
        # Feature extraction layers
        self.features = nn.Sequential(
            # Conv Block 1: Input (1, 200, 190) -> (96, 49, 46)
            nn.Conv2d(input_channels, 96, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            
            # Conv Block 2: (96, 49, 46) -> (256, 23, 21)
            nn.Conv2d(96, 256, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            
            # Conv Block 3: (256, 23, 21) -> (384, 23, 21)
            nn.Conv2d(256, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            
            # Conv Block 4: (384, 23, 21) -> (384, 23, 21)
            nn.Conv2d(384, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            
            # Conv Block 5: (384, 23, 21) -> (256, 10, 9)
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        
        # Adaptive pooling to fixed size for variable input
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        
        # Classifier layers
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            
            nn.Dropout(p=dropout_rate),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            
            nn.Linear(4096, num_classes),
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self) -> None:
        """Initialize network weights using Kaiming initialization."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor of shape (batch_size, channels, height, width)
        
        Returns:
            Output logits of shape (batch_size, num_classes)
        """
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
    
    def get_feature_maps(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get feature maps from the feature extraction layers (useful for visualization).
        
        Args:
            x: Input tensor
        
        Returns:
            Feature maps before classifier
        """
        x = self.features(x)
        x = self.avgpool(x)
        return x


class AlexNetLite(nn.Module):
    """
    Lightweight AlexNet for faster training and lower memory requirements.
    Suitable for medical imaging with limited computational resources.
    """
    
    def __init__(self, num_classes: int = 4, input_channels: int = 1, dropout_rate: float = 0.4):
        """
        Initialize lightweight AlexNet.
        
        Args:
            num_classes: Number of output classes
            input_channels: Number of input channels
            dropout_rate: Dropout probability
        """
        super(AlexNetLite, self).__init__()
        
        self.num_classes = num_classes
        
        self.features = nn.Sequential(
            # Conv Block 1
            nn.Conv2d(input_channels, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            
            # Conv Block 2
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            
            # Conv Block 3
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            
            # Conv Block 4
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        
        self.avgpool = nn.AdaptiveAvgPool2d((4, 4))
        
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(256 * 4 * 4, 2048),
            nn.ReLU(inplace=True),
            
            nn.Dropout(p=dropout_rate),
            nn.Linear(2048, 1024),
            nn.ReLU(inplace=True),
            
            nn.Linear(1024, num_classes),
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self) -> None:
        """Initialize weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


def create_alexnet(
    num_classes: int = 4,
    input_channels: int = 1,
    dropout_rate: float = 0.5,
    lite: bool = False
) -> nn.Module:
    """
    Factory function to create AlexNet model.
    
    Args:
        num_classes: Number of output classes
        input_channels: Number of input channels
        dropout_rate: Dropout probability
        lite: If True, creates lightweight version
    
    Returns:
        AlexNet model
    """
    if lite:
        return AlexNetLite(num_classes=num_classes, input_channels=input_channels, dropout_rate=dropout_rate)
    else:
        return AlexNet(num_classes=num_classes, input_channels=input_channels, dropout_rate=dropout_rate)


def get_model_summary(model: nn.Module, input_size: tuple = (1, 1, 200, 190)) -> None:
    """
    Print model architecture summary.
    
    Args:
        model: PyTorch model
        input_size: Input tensor size (batch, channels, height, width)
    """
    print("\n" + "=" * 70)
    print("MODEL ARCHITECTURE SUMMARY")
    print("=" * 70)
    print(f"\nModel: {model.__class__.__name__}")
    print(f"Input size: {input_size}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Print layer details
    print("\nLayer Details:")
    print("-" * 70)
    
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            params = sum(p.numel() for p in module.parameters())
            print(f"{name:<40} | {params:>10,} params")
    
    print("-" * 70)
    print(f"{'Total':<40} | {trainable_params:>10,} params\n")


if __name__ == "__main__":
    # Test model instantiation
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}\n")
    
    # Create standard AlexNet
    model = create_alexnet(num_classes=4, input_channels=1)
    model.to(device)
    get_model_summary(model)
    
    # Test forward pass
    dummy_input = torch.randn(4, 1, 200, 190).to(device)
    output = model(dummy_input)
    print(f"Output shape: {output.shape}\n")
    
    # Create lightweight version
    model_lite = create_alexnet(num_classes=4, input_channels=1, lite=True)
    model_lite.to(device)
    get_model_summary(model_lite)
    
    output_lite = model_lite(dummy_input)
    print(f"Output shape: {output_lite.shape}\n")
    
