import cv2
import numpy as np
from PIL import Image
import torch
from torchvision import transforms


class ThermalImagePreprocessor:
    """
    Preprocessor for thermal breast imaging data.
    """
    
    def __init__(self, image_size=224, normalize=True, augment=False):
        """
        Args:
            image_size (int): Target image size
            normalize (bool): Apply ImageNet normalization
            augment (bool): Apply data augmentation
        """
        self.image_size = image_size
        self.normalize = normalize
        self.augment = augment
        
        # Define transformations
        self._setup_transforms()
    
    def _setup_transforms(self):
        """Setup image transformations."""
        transform_list = []
        
        # Resize
        transform_list.append(transforms.Resize((self.image_size, self.image_size)))
        
        # Augmentation (for training)
        if self.augment:
            transform_list.extend([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(10),
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
                transforms.RandomAffine(degrees=0, translate=(0.1, 0.1))
            ])
        
        # Convert to tensor
        transform_list.append(transforms.ToTensor())
        
        # Normalization (ImageNet stats)
        if self.normalize:
            transform_list.append(
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            )
        
        self.transform = transforms.Compose(transform_list)
    
    def preprocess_image(self, image_path):
        """
        Preprocess a single image.
        
        Args:
            image_path (str): Path to image file
        
        Returns:
            torch.Tensor: Preprocessed image tensor
        """
        # Load image
        image = Image.open(image_path).convert('RGB')
        
        # Apply transformations
        image_tensor = self.transform(image)
        
        return image_tensor
    
    def preprocess_cv2_image(self, cv2_image):
        """
        Preprocess OpenCV image.
        
        Args:
            cv2_image: OpenCV image (BGR format)
        
        Returns:
            torch.Tensor: Preprocessed image tensor
        """
        # Convert BGR to RGB
        image = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB)
        
        # Convert to PIL Image
        pil_image = Image.fromarray(image)
        
        # Apply transformations
        image_tensor = self.transform(pil_image)
        
        return image_tensor
    
    def preprocess_batch(self, image_paths):
        """
        Preprocess a batch of images.
        
        Args:
            image_paths (list): List of image paths
        
        Returns:
            torch.Tensor: Batch of preprocessed images
        """
        images = []
        for path in image_paths:
            img = self.preprocess_image(path)
            images.append(img)
        
        return torch.stack(images)


def apply_thermal_colormap(gray_image, colormap=cv2.COLORMAP_JET):
    """
    Apply thermal colormap to grayscale image.
    
    Args:
        gray_image: Grayscale image
        colormap: OpenCV colormap
    
    Returns:
        Colored thermal image
    """
    # Normalize to 0-255
    normalized = cv2.normalize(gray_image, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    
    # Apply colormap
    colored = cv2.applyColorMap(normalized, colormap)
    
    return colored


def enhance_thermal_image(image):
    """
    Enhance thermal image using histogram equalization.
    
    Args:
        image: Input image
    
    Returns:
        Enhanced image
    """
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    
    return enhanced


def extract_roi(image, roi_detector='simple'):
    """
    Extract Region of Interest (ROI) from thermal image.
    
    Args:
        image: Input thermal image
        roi_detector: ROI detection method
    
    Returns:
        ROI image
    """
    # Simple thresholding-based ROI extraction
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    # Apply Otsu's thresholding
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        # Get largest contour
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Get bounding box
        x, y, w, h = cv2.boundingRect(largest_contour)
        
        # Extract ROI
        roi = image[y:y+h, x:x+w]
        return roi
    
    return image


if __name__ == "__main__":
    # Test preprocessing
    preprocessor = ThermalImagePreprocessor(augment=True)
    print("Thermal Image Preprocessor initialized")
    print(f"Image size: {preprocessor.image_size}")
    print(f"Normalization: {preprocessor.normalize}")
    print(f"Augmentation: {preprocessor.augment}")
