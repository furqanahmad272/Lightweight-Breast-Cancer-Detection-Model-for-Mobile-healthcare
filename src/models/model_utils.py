import torch
import torch.nn as nn
from pathlib import Path

def save_model(model, path, epoch=None, optimizer=None, metrics=None):
    """
    Save model checkpoint.
    
    Args:
        model: PyTorch model
        path: Save path
        epoch: Current epoch number
        optimizer: Optimizer state
        metrics: Training metrics dictionary
    """
    checkpoint = {
        'model_state_dict': model.state_dict(),
    }
    
    if epoch is not None:
        checkpoint['epoch'] = epoch
    
    if optimizer is not None:
        checkpoint['optimizer_state_dict'] = optimizer.state_dict()
    
    if metrics is not None:
        checkpoint['metrics'] = metrics
    
    torch.save(checkpoint, path)
    print(f"Model saved to {path}")


def load_model(model, path, device='cpu', load_optimizer=False, optimizer=None):
    """
    Load model checkpoint.
    
    Args:
        model: PyTorch model
        path: Checkpoint path
        device: Device to load model to
        load_optimizer: Whether to load optimizer state
        optimizer: Optimizer to load state into
    
    Returns:
        dict: Checkpoint information (epoch, metrics)
    """
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    info = {}
    
    if 'epoch' in checkpoint:
        info['epoch'] = checkpoint['epoch']
    
    if 'metrics' in checkpoint:
        info['metrics'] = checkpoint['metrics']
    
    if load_optimizer and optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    print(f"Model loaded from {path}")
    return info


def count_parameters(model):
    """
    Count the number of trainable parameters in a model.
    
    Args:
        model: PyTorch model
    
    Returns:
        int: Number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_model_size(model):
    """
    Calculate model size in MB.
    
    Args:
        model: PyTorch model
    
    Returns:
        float: Model size in MB
    """
    param_size = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
    size_mb = (param_size + buffer_size) / 1024**2
    return size_mb


def export_to_onnx(model, save_path, input_shape=(1, 3, 224, 224), device='cpu'):
    """
    Export PyTorch model to ONNX format.
    
    Args:
        model: PyTorch model
        save_path: Path to save ONNX model
        input_shape: Input tensor shape
        device: Device to use for export
    """
    model.eval()
    model.to(device)
    
    dummy_input = torch.randn(input_shape).to(device)
    
    torch.onnx.export(
        model,
        dummy_input,
        save_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )
    
    print(f"Model exported to ONNX format: {save_path}")


def print_model_summary(model, input_shape=(1, 3, 224, 224)):
    """
    Print model summary with parameters and size.
    
    Args:
        model: PyTorch model
        input_shape: Input tensor shape for testing
    """
    print("=" * 70)
    print("Model Summary")
    print("=" * 70)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = count_parameters(model)
    
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Non-trainable parameters: {total_params - trainable_params:,}")
    
    # Model size
    size_mb = get_model_size(model)
    print(f"Model size: {size_mb:.2f} MB")
    
    # Test forward pass
    model.eval()
    with torch.no_grad():
        dummy_input = torch.randn(input_shape)
        output = model(dummy_input)
        print(f"Input shape: {dummy_input.shape}")
        print(f"Output shape: {output.shape}")
    
    print("=" * 70)
