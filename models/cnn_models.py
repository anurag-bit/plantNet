import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from typing import Optional, List, Dict


class PlantDiseaseCNN(nn.Module):
    """
    Custom CNN architecture for plant disease classification.
    Features multiple convolutional blocks with batch normalization,
    dropout for regularization, and global average pooling.
    """
    
    def __init__(self, num_classes: int, dropout_rate: float = 0.5):
        """
        Initialize the PlantDiseaseCNN model.
        
        Args:
            num_classes (int): Number of plant disease classes
            dropout_rate (float): Dropout rate for regularization
        """
        super(PlantDiseaseCNN, self).__init__()
        
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate
        
        # First convolutional block
        self.conv_block1 = self._make_conv_block(3, 64, 3, 1, 1)
        self.conv_block2 = self._make_conv_block(64, 64, 3, 1, 1)
        self.pool1 = nn.MaxPool2d(2, 2)
        
        # Second convolutional block
        self.conv_block3 = self._make_conv_block(64, 128, 3, 1, 1)
        self.conv_block4 = self._make_conv_block(128, 128, 3, 1, 1)
        self.pool2 = nn.MaxPool2d(2, 2)
        
        # Third convolutional block
        self.conv_block5 = self._make_conv_block(128, 256, 3, 1, 1)
        self.conv_block6 = self._make_conv_block(256, 256, 3, 1, 1)
        self.conv_block7 = self._make_conv_block(256, 256, 3, 1, 1)
        self.pool3 = nn.MaxPool2d(2, 2)
        
        # Fourth convolutional block
        self.conv_block8 = self._make_conv_block(256, 512, 3, 1, 1)
        self.conv_block9 = self._make_conv_block(512, 512, 3, 1, 1)
        self.conv_block10 = self._make_conv_block(512, 512, 3, 1, 1)
        self.pool4 = nn.MaxPool2d(2, 2)
        
        # Fifth convolutional block
        self.conv_block11 = self._make_conv_block(512, 512, 3, 1, 1)
        self.conv_block12 = self._make_conv_block(512, 512, 3, 1, 1)
        self.conv_block13 = self._make_conv_block(512, 512, 3, 1, 1)
        self.pool5 = nn.MaxPool2d(2, 2)
        
        # Global Average Pooling instead of fully connected layers
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Final classifier
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(256, num_classes)
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _make_conv_block(self, in_channels: int, out_channels: int, 
                        kernel_size: int, stride: int, padding: int) -> nn.Sequential:
        """Create a convolutional block with BatchNorm and ReLU activation."""
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def _initialize_weights(self):
        """Initialize model weights using Xavier/He initialization."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network."""
        # First block
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.pool1(x)
        
        # Second block
        x = self.conv_block3(x)
        x = self.conv_block4(x)
        x = self.pool2(x)
        
        # Third block
        x = self.conv_block5(x)
        x = self.conv_block6(x)
        x = self.conv_block7(x)
        x = self.pool3(x)
        
        # Fourth block
        x = self.conv_block8(x)
        x = self.conv_block9(x)
        x = self.conv_block10(x)
        x = self.pool4(x)
        
        # Fifth block
        x = self.conv_block11(x)
        x = self.conv_block12(x)
        x = self.conv_block13(x)
        x = self.pool5(x)
        
        # Global Average Pooling
        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1)
        
        # Classifier
        x = self.classifier(x)
        
        return x


class PlantDiseaseResNet(nn.Module):
    """
    ResNet-based model for plant disease classification using transfer learning.
    """
    
    def __init__(self, num_classes: int, model_name: str = 'resnet50', 
                 pretrained: bool = True, freeze_backbone: bool = False):
        """
        Initialize ResNet-based model.
        
        Args:
            num_classes (int): Number of plant disease classes
            model_name (str): ResNet variant ('resnet18', 'resnet34', 'resnet50', 'resnet101')
            pretrained (bool): Whether to use pretrained weights
            freeze_backbone (bool): Whether to freeze the backbone for feature extraction
        """
        super(PlantDiseaseResNet, self).__init__()
        
        self.num_classes = num_classes
        
        # Load pretrained ResNet
        if model_name == 'resnet18':
            self.backbone = models.resnet18(pretrained=pretrained)
            feature_dim = 512
        elif model_name == 'resnet34':
            self.backbone = models.resnet34(pretrained=pretrained)
            feature_dim = 512
        elif model_name == 'resnet50':
            self.backbone = models.resnet50(pretrained=pretrained)
            feature_dim = 2048
        elif model_name == 'resnet101':
            self.backbone = models.resnet101(pretrained=pretrained)
            feature_dim = 2048
        else:
            raise ValueError(f"Unsupported model: {model_name}")
        
        # Freeze backbone if specified
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        # Replace the final fully connected layer
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(feature_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network."""
        return self.backbone(x)


class PlantDiseaseEfficientNet(nn.Module):
    """
    EfficientNet-based model for plant disease classification.
    """
    
    def __init__(self, num_classes: int, model_name: str = 'efficientnet_b0', 
                 pretrained: bool = True, dropout_rate: float = 0.3):
        """
        Initialize EfficientNet-based model.
        
        Args:
            num_classes (int): Number of plant disease classes
            model_name (str): EfficientNet variant
            pretrained (bool): Whether to use pretrained weights
            dropout_rate (float): Dropout rate for the classifier
        """
        super(PlantDiseaseEfficientNet, self).__init__()
        
        try:
            # Try to load EfficientNet (requires timm library)
            import timm
            self.backbone = timm.create_model(model_name, pretrained=pretrained, 
                                            num_classes=0)  # Remove classifier
            feature_dim = self.backbone.num_features
        except ImportError:
            print("Warning: timm library not found. Using ResNet50 instead.")
            self.backbone = models.resnet50(pretrained=pretrained)
            feature_dim = 2048
            # Remove the final layer
            self.backbone.fc = nn.Identity()
        
        # Custom classifier
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(feature_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate / 2),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network."""
        features = self.backbone(x)
        return self.classifier(features)


def create_model(model_type: str, num_classes: int, **kwargs) -> nn.Module:
    """
    Factory function to create different model types.
    
    Args:
        model_type (str): Type of model ('custom_cnn', 'resnet', 'efficientnet')
        num_classes (int): Number of classes
        **kwargs: Additional arguments for model initialization
        
    Returns:
        nn.Module: The created model
    """
    if model_type.lower() == 'custom_cnn':
        return PlantDiseaseCNN(num_classes, **kwargs)
    elif model_type.lower() == 'resnet':
        return PlantDiseaseResNet(num_classes, **kwargs)
    elif model_type.lower() == 'efficientnet':
        return PlantDiseaseEfficientNet(num_classes, **kwargs)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")


def count_parameters(model: nn.Module) -> int:
    """Count the number of trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def model_summary(model: nn.Module, input_size: tuple = (3, 224, 224)) -> None:
    """
    Print a summary of the model architecture.
    
    Args:
        model (nn.Module): The model to summarize
        input_size (tuple): Input tensor size (C, H, W)
    """
    def register_hook(module):
        def hook(module, input, output):
            class_name = str(module.__class__).split(".")[-1].split("'")[0]
            module_idx = len(summary)
            
            m_key = f"{class_name}-{module_idx + 1}"
            summary[m_key] = {}
            summary[m_key]["input_shape"] = list(input[0].type_as(torch.FloatTensor()).shape)
            summary[m_key]["input_shape"][0] = -1
            if isinstance(output, (list, tuple)):
                summary[m_key]["output_shape"] = [[-1] + list(o.size())[1:] for o in output]
            else:
                summary[m_key]["output_shape"] = list(output.shape)
                summary[m_key]["output_shape"][0] = -1
            
            params = 0
            if hasattr(module, "weight") and hasattr(module.weight, "size"):
                params += torch.prod(torch.LongTensor(list(module.weight.size())))
                summary[m_key]["trainable"] = module.weight.requires_grad
            if hasattr(module, "bias") and hasattr(module.bias, "size"):
                params += torch.prod(torch.LongTensor(list(module.bias.size())))
            summary[m_key]["nb_params"] = params
        
        if not isinstance(module, nn.Sequential) and not isinstance(module, nn.ModuleList):
            hooks.append(module.register_forward_hook(hook))
    
    # Create summary dict
    summary = {}
    hooks = []
    
    # Register hooks
    model.apply(register_hook)
    
    # Create a dummy input
    device = next(model.parameters()).device
    x = torch.rand(1, *input_size).to(device)
    
    # Forward pass
    model(x)
    
    # Remove hooks
    for h in hooks:
        h.remove()
    
    # Print summary
    print("-" * 80)
    print(f"{'Layer (type)':>25} {'Output Shape':>25} {'Param #':>15}")
    print("=" * 80)
    total_params = 0
    total_output = 0
    trainable_params = 0
    
    for layer in summary:
        line_new = f"{layer:>25} {str(summary[layer]['output_shape']):>25} {summary[layer]['nb_params']:>15,}"
        total_params += summary[layer]["nb_params"]
        
        if "trainable" in summary[layer] and summary[layer]["trainable"]:
            trainable_params += summary[layer]["nb_params"]
        print(line_new)
    
    print("=" * 80)
    print(f"Total params: {total_params:,}")
    print(f"Trainable params: {trainable_params:,}")
    print(f"Non-trainable params: {total_params - trainable_params:,}")
    print("-" * 80)


class PlantDiseaseViT(nn.Module):
    """
    Vision Transformer (ViT) based model for plant disease classification.
    """
    
    def __init__(self, num_classes: int, model_name: str = 'vit_base_patch16_224', 
                 pretrained: bool = True, dropout_rate: float = 0.1):
        """
        Initialize ViT-based model.
        
        Args:
            num_classes (int): Number of plant disease classes
            model_name (str): ViT variant
            pretrained (bool): Whether to use pretrained weights
            dropout_rate (float): Dropout rate for the classifier
        """
        super(PlantDiseaseViT, self).__init__()
        
        try:
            # Try to load ViT (requires timm library)
            import timm
            self.backbone = timm.create_model(model_name, pretrained=pretrained, 
                                            num_classes=0)  # Remove classifier
            feature_dim = self.backbone.num_features
        except ImportError:
            print("Warning: timm library not found. Using ResNet50 instead.")
            self.backbone = models.resnet50(pretrained=pretrained)
            feature_dim = 2048
            # Remove the final layer
            self.backbone.fc = nn.Identity()
        
        # Custom classifier with higher dropout for ViT
        self.classifier = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Dropout(dropout_rate),
            nn.Linear(feature_dim, 512),
            nn.GELU(),
            nn.Dropout(dropout_rate / 2),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network."""
        features = self.backbone(x)
        return self.classifier(features)


class PlantDiseaseSwin(nn.Module):
    """
    Swin Transformer based model for plant disease classification.
    """
    
    def __init__(self, num_classes: int, model_name: str = 'swin_base_patch4_window7_224', 
                 pretrained: bool = True, dropout_rate: float = 0.1):
        """
        Initialize Swin Transformer based model.
        
        Args:
            num_classes (int): Number of plant disease classes
            model_name (str): Swin variant
            pretrained (bool): Whether to use pretrained weights
            dropout_rate (float): Dropout rate for the classifier
        """
        super(PlantDiseaseSwin, self).__init__()
        
        try:
            # Try to load Swin (requires timm library)
            import timm
            self.backbone = timm.create_model(model_name, pretrained=pretrained, 
                                            num_classes=0)  # Remove classifier
            feature_dim = self.backbone.num_features
        except ImportError:
            print("Warning: timm library not found. Using ResNet50 instead.")
            self.backbone = models.resnet50(pretrained=pretrained)
            feature_dim = 2048
            # Remove the final layer
            self.backbone.fc = nn.Identity()
        
        # Custom classifier
        self.classifier = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Dropout(dropout_rate),
            nn.Linear(feature_dim, 512),
            nn.GELU(),
            nn.Dropout(dropout_rate / 2),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network."""
        features = self.backbone(x)
        return self.classifier(features)


class EnsembleModel(nn.Module):
    """
    Ensemble model combining multiple architectures for maximum performance.
    """
    
    def __init__(self, models: List[nn.Module], weights: List[float] = None):
        """
        Initialize ensemble model.
        
        Args:
            models (List[nn.Module]): List of models to ensemble
            weights (List[float]): Weights for each model (optional)
        """
        super(EnsembleModel, self).__init__()
        
        self.models = nn.ModuleList(models)
        
        if weights is None:
            self.weights = torch.ones(len(models)) / len(models)
        else:
            self.weights = torch.tensor(weights, dtype=torch.float32)
            self.weights = self.weights / self.weights.sum()  # Normalize
        
        self.num_classes = models[0].classifier[-1].out_features if hasattr(models[0], 'classifier') else models[0].fc.out_features
        
        # Register weights as buffer so they move with model
        self.register_buffer('model_weights', self.weights)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through ensemble."""
        outputs = []
        for model in self.models:
            output = model(x)
            outputs.append(output)
        
        # Weighted average of outputs
        ensemble_output = torch.zeros_like(outputs[0])
        for i, output in enumerate(outputs):
            ensemble_output += self.model_weights[i] * output
        
        return ensemble_output
    
    def get_individual_predictions(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Get predictions from each individual model."""
        predictions = []
        for model in self.models:
            with torch.no_grad():
                output = model(x)
                pred = torch.softmax(output, dim=1)
                predictions.append(pred)
        return predictions


def create_ensemble_model(architectures: List[dict], num_classes: int) -> EnsembleModel:
    """
    Create ensemble model from architecture specifications.
    
    Args:
        architectures (List[dict]): List of architecture specifications
        num_classes (int): Number of classes
        
    Returns:
        EnsembleModel: The ensemble model
    """
    models = []
    weights = []
    
    for arch in architectures:
        arch_type = arch['type']
        weight = arch.get('weight', 1.0)
        
        if arch_type == 'resnet':
            model = PlantDiseaseResNet(
                num_classes=num_classes,
                model_name=arch['name'],
                pretrained=arch.get('pretrained', True)
            )
        elif arch_type == 'efficientnet':
            model = PlantDiseaseEfficientNet(
                num_classes=num_classes,
                model_name=arch['name'],
                pretrained=arch.get('pretrained', True)
            )
        elif arch_type == 'vit':
            model = PlantDiseaseViT(
                num_classes=num_classes,
                model_name=arch['name'],
                pretrained=arch.get('pretrained', True)
            )
        elif arch_type == 'swin':
            model = PlantDiseaseSwin(
                num_classes=num_classes,
                model_name=arch['name'],
                pretrained=arch.get('pretrained', True)
            )
        else:
            raise ValueError(f"Unsupported architecture type: {arch_type}")
        
        models.append(model)
        weights.append(weight)
    
    return EnsembleModel(models, weights)


def create_model(model_type: str, num_classes: int, **kwargs) -> nn.Module:
    """
    Factory function to create different model types.
    
    Args:
        model_type (str): Type of model ('custom_cnn', 'resnet', 'efficientnet', 'vit', 'swin', 'ensemble')
        num_classes (int): Number of classes
        **kwargs: Additional arguments for model initialization
        
    Returns:
        nn.Module: The created model
    """
    if model_type.lower() == 'custom_cnn':
        return PlantDiseaseCNN(num_classes, **kwargs)
    elif model_type.lower() == 'resnet':
        return PlantDiseaseResNet(num_classes, **kwargs)
    elif model_type.lower() == 'efficientnet':
        return PlantDiseaseEfficientNet(num_classes, **kwargs)
    elif model_type.lower() == 'vit':
        return PlantDiseaseViT(num_classes, **kwargs)
    elif model_type.lower() == 'swin':
        return PlantDiseaseSwin(num_classes, **kwargs)
    elif model_type.lower() == 'ensemble':
        architectures = kwargs.get('architectures', [])
        return create_ensemble_model(architectures, num_classes)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")


if __name__ == "__main__":
    # Test model creation
    print("Testing model architectures...")
    
    num_classes = 38  # PlantVillage dataset has 38 classes
    
    # Test Custom CNN
    print("\n1. Custom CNN:")
    model_cnn = create_model('custom_cnn', num_classes, dropout_rate=0.5)
    print(f"Parameters: {count_parameters(model_cnn):,}")
    
    # Test ResNet
    print("\n2. ResNet101:")
    model_resnet = create_model('resnet', num_classes, model_name='resnet101', pretrained=False)
    print(f"Parameters: {count_parameters(model_resnet):,}")
    
    # Test Ensemble (simplified for testing)
    print("\n3. Ensemble Model:")
    ensemble_archs = [
        {'type': 'resnet', 'name': 'resnet50', 'pretrained': False, 'weight': 0.5},
        {'type': 'resnet', 'name': 'resnet34', 'pretrained': False, 'weight': 0.5}
    ]
    model_ensemble = create_model('ensemble', num_classes, architectures=ensemble_archs)
    print(f"Parameters: {count_parameters(model_ensemble):,}")
    
    # Test input/output shapes
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_cnn.to(device)
    
    dummy_input = torch.randn(4, 3, 384, 384).to(device)  # Higher resolution for MI300X
    output = model_cnn(dummy_input)
    print(f"\nInput shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Using device: {device}")