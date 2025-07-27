import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import rasterio
from tqdm import tqdm
import segmentation_models_pytorch as smp
from torch.utils.tensorboard import SummaryWriter
import warnings
warnings.filterwarnings('ignore')

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

class FloodDataset(Dataset):
    def __init__(self, image_paths, label_paths, transform=None, augment=False):
        self.image_paths = image_paths
        self.label_paths = label_paths
        self.transform = transform
        self.augment = augment
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # Load multi-channel TIFF image
        with rasterio.open(self.image_paths[idx]) as src:
            image = src.read()  # Shape: (12, H, W)
            
        # Load PNG label
        label = np.array(Image.open(self.label_paths[idx]))
        
        # Normalize image channels
        image = self.normalize_channels(image)
        
        # Convert to torch tensors
        image = torch.from_numpy(image).float()
        label = torch.from_numpy(label).long()
        
        # Ensure label is binary (0 or 1)
        label = (label > 0).long()
        
        if self.transform:
            # Apply transforms (you might need to adapt this for multi-channel data)
            pass
            
        return image, label
    
    def normalize_channels(self, image):
        """Normalize each channel independently"""
        normalized = np.zeros_like(image, dtype=np.float32)
        
        for i in range(image.shape[0]):
            channel = image[i].astype(np.float32)
            # Handle different normalization strategies for different channel types
            if i in [0, 1, 2, 3, 4, 5, 6]:  # Spectral bands
                # Normalize to 0-1 range, handling potential outliers
                p2, p98 = np.percentile(channel, (2, 98))
                channel = np.clip(channel, p2, p98)
                channel = (channel - p2) / (p98 - p2)
            elif i in [8, 9]:  # DEM channels
                # Normalize elevation data
                channel = (channel - np.mean(channel)) / (np.std(channel) + 1e-8)
            else:  # Other channels (QA, landcover, water occurrence)
                # Simple min-max normalization
                min_val, max_val = np.min(channel), np.max(channel)
                if max_val > min_val:
                    channel = (channel - min_val) / (max_val - min_val)
            
            normalized[i] = channel
            
        return normalized

def prepare_data_paths(images_dir, labels_dir):
    """Prepare matched image and label paths"""
    image_files = [f for f in os.listdir(images_dir) if f.endswith('.tif')]
    
    image_paths = []
    label_paths = []
    
    for img_file in image_files:
        base_name = os.path.splitext(img_file)[0]
        
        # Handle both 1.png and 1_*.png cases
        possible_labels = [f for f in os.listdir(labels_dir)
                          if f.startswith(base_name) and f.endswith('.png')]
        
        if not possible_labels:
            print(f"No matching label found for {img_file}")
            continue
            
        label_file = possible_labels[0]  # Take the first match
        
        image_paths.append(os.path.join(images_dir, img_file))
        label_paths.append(os.path.join(labels_dir, label_file))
    
    print(f"Found {len(image_paths)} image-label pairs")
    return image_paths, label_paths

class MultiChannelUNet(nn.Module):
    """Custom U-Net for multi-channel input"""
    def __init__(self, in_channels=12, num_classes=2, encoder_name="resnet34", encoder_weights="imagenet"):
        super().__init__()
        
        # Use segmentation_models_pytorch with custom input adapter
        self.input_adapter = nn.Conv2d(in_channels, 3, kernel_size=1)
        
        self.model = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=3,
            classes=num_classes,
        )
        
    def forward(self, x):
        # Adapt multi-channel input to 3 channels for pretrained encoder
        x = self.input_adapter(x)
        return self.model(x)

def dice_loss(pred, target, smooth=1e-6):
    """Dice loss for segmentation"""
    pred = torch.softmax(pred, dim=1)
    pred = pred[:, 1, :, :]  # Take positive class
    target = target.float()
    
    intersection = (pred * target).sum(dim=(1, 2))
    union = pred.sum(dim=(1, 2)) + target.sum(dim=(1, 2))
    
    dice = (2. * intersection + smooth) / (union + smooth)
    return 1 - dice.mean()

def combined_loss(pred, target, alpha=0.5):
    """Combination of CrossEntropy and Dice loss"""
    ce_loss = nn.CrossEntropyLoss()(pred, target)
    d_loss = dice_loss(pred, target)
    return alpha * ce_loss + (1 - alpha) * d_loss

def calculate_iou(pred, target, num_classes=2):
    """Calculate IoU for each class"""
    pred = torch.softmax(pred, dim=1)
    pred = torch.argmax(pred, dim=1)
    
    ious = []
    for cls in range(num_classes):
        pred_cls = (pred == cls)
        target_cls = (target == cls)
        
        intersection = (pred_cls & target_cls).sum().float()
        union = (pred_cls | target_cls).sum().float()
        
        if union == 0:
            iou = 1.0 if intersection == 0 else 0.0
        else:
            iou = intersection / union
        ious.append(iou.item())
    
    return ious

def train_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    total_iou = [0, 0]
    
    for batch_idx, (images, labels) in enumerate(tqdm(dataloader, desc="Training")):
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        # Calculate IoU
        ious = calculate_iou(outputs, labels)
        total_iou[0] += ious[0]
        total_iou[1] += ious[1]
    
    avg_loss = total_loss / len(dataloader)
    avg_iou = [iou / len(dataloader) for iou in total_iou]
    
    return avg_loss, avg_iou

def validate_epoch(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    total_iou = [0, 0]
    
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Validation"):
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            
            # Calculate IoU
            ious = calculate_iou(outputs, labels)
            total_iou[0] += ious[0]
            total_iou[1] += ious[1]
    
    avg_loss = total_loss / len(dataloader)
    avg_iou = [iou / len(dataloader) for iou in total_iou]
    
    return avg_loss, avg_iou

def visualize_predictions(model, dataloader, device, num_samples=4):
    """Visualize model predictions"""
    model.eval()
    fig, axes = plt.subplots(num_samples, 4, figsize=(16, num_samples * 4))
    
    with torch.no_grad():
        for idx, (images, labels) in enumerate(dataloader):
            if idx >= num_samples:
                break
                
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            predictions = torch.softmax(outputs, dim=1)
            predictions = torch.argmax(predictions, dim=1)
            
            # Take first sample from batch
            image = images[0].cpu().numpy()
            label = labels[0].cpu().numpy()
            pred = predictions[0].cpu().numpy()
            
            # Create RGB composite (using bands 3, 2, 1 for Red, Green, Blue)
            rgb = np.stack([image[3], image[2], image[1]], axis=0)
            rgb = np.transpose(rgb, (1, 2, 0))
            
            # Normalize for display
            rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min())
            rgb = np.clip(rgb, 0, 1)
            
            axes[idx, 0].imshow(rgb)
            axes[idx, 0].set_title('RGB Composite')
            axes[idx, 0].axis('off')
            
            axes[idx, 1].imshow(label, cmap='Blues')
            axes[idx, 1].set_title('Ground Truth')
            axes[idx, 1].axis('off')
            
            axes[idx, 2].imshow(pred, cmap='Blues')
            axes[idx, 2].set_title('Prediction')
            axes[idx, 2].axis('off')
            
            # Show difference
            diff = np.abs(label.astype(float) - pred.astype(float))
            axes[idx, 3].imshow(diff, cmap='Reds')
            axes[idx, 3].set_title('Difference')
            axes[idx, 3].axis('off')
            
            break
    
    plt.tight_layout()
    plt.savefig('predictions_visualization.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_training_curves(train_losses, val_losses, train_ious, val_ious):
    """Plot training curves"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    epochs = range(1, len(train_losses) + 1)
    
    # Loss curves
    ax1.plot(epochs, train_losses, 'b-', label='Training Loss')
    ax1.plot(epochs, val_losses, 'r-', label='Validation Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Background IoU
    bg_train_ious = [iou[0] for iou in train_ious]
    bg_val_ious = [iou[0] for iou in val_ious]
    ax2.plot(epochs, bg_train_ious, 'b-', label='Training IoU (Background)')
    ax2.plot(epochs, bg_val_ious, 'r-', label='Validation IoU (Background)')
    ax2.set_title('Background IoU')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('IoU')
    ax2.legend()
    ax2.grid(True)
    
    # Flood IoU
    flood_train_ious = [iou[1] for iou in train_ious]
    flood_val_ious = [iou[1] for iou in val_ious]
    ax3.plot(epochs, flood_train_ious, 'b-', label='Training IoU (Flood)')
    ax3.plot(epochs, flood_val_ious, 'r-', label='Validation IoU (Flood)')
    ax3.set_title('Flood IoU')
    ax3.set_xlabel('Epochs')
    ax3.set_ylabel('IoU')
    ax3.legend()
    ax3.grid(True)
    
    # Mean IoU
    mean_train_ious = [(iou[0] + iou[1]) / 2 for iou in train_ious]
    mean_val_ious = [(iou[0] + iou[1]) / 2 for iou in val_ious]
    ax4.plot(epochs, mean_train_ious, 'b-', label='Training Mean IoU')
    ax4.plot(epochs, mean_val_ious, 'r-', label='Validation Mean IoU')
    ax4.set_title('Mean IoU')
    ax4.set_xlabel('Epochs')
    ax4.set_ylabel('IoU')
    ax4.legend()
    ax4.grid(True)
    
    plt.tight_layout()
    plt.savefig('training_curves.png', dpi=300, bbox_inches='tight')
    plt.show()

def main(images_dir, labels_dir):
    # Prepare data paths
    image_paths, label_paths = prepare_data_paths(images_dir, labels_dir)
    
    # Split data
    train_imgs, val_imgs, train_lbls, val_lbls = train_test_split(
        image_paths, label_paths, test_size=0.2, random_state=42
    )
    
    print(f"Training samples: {len(train_imgs)}")
    print(f"Validation samples: {len(val_imgs)}")
    
    # Create datasets and dataloaders
    train_dataset = FloodDataset(train_imgs, train_lbls)
    val_dataset = FloodDataset(val_imgs, val_lbls)
    
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=2)
    
    # Initialize model
    model = MultiChannelUNet(in_channels=12, num_classes=2).to(device)
    
    # Loss and optimizer
    criterion = combined_loss
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)
    
    # Training parameters
    num_epochs = 50
    best_val_loss = float('inf')
    patience = 10
    patience_counter = 0
    
    # Tracking lists
    train_losses = []
    val_losses = []
    train_ious = []
    val_ious = []
    
    print("Starting training...")
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        
        # Training
        train_loss, train_iou = train_epoch(model, train_loader, optimizer, criterion, device)
        
        # Validation
        val_loss, val_iou = validate_epoch(model, val_loader, criterion, device)
        
        # Update scheduler
        scheduler.step(val_loss)
        
        # Store metrics
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_ious.append(train_iou)
        val_ious.append(val_iou)
        
        print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        print(f"Train IoU (BG/Flood): {train_iou[0]:.4f}/{train_iou[1]:.4f}")
        print(f"Val IoU (BG/Flood): {val_iou[0]:.4f}/{val_iou[1]:.4f}")
        
        # Early stopping and model saving
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_iou': val_iou
            }, 'best_flood_model.pth')
            print("Model saved!")
        else:
            patience_counter += 1
            
        if patience_counter >= patience:
            print(f"Early stopping triggered after {epoch+1} epochs")
            break
    
    # Load best model for visualization
    checkpoint = torch.load('best_flood_model.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Plot training curves
    plot_training_curves(train_losses, val_losses, train_ious, val_ious)
    
    # Visualize predictions
    visualize_predictions(model, val_loader, device)
    
    print("\nTraining completed!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Best validation IoU: {checkpoint['val_iou']}")

# Example usage
if __name__ == "__main__":
    # Set your data paths here
    images_dir = "data\data\images"  # Replace with actual path
    labels_dir = "data\data\labels"  # Replace with actual path
    
    # Install required packages if not already installed
    # pip install segmentation-models-pytorch rasterio pillow scikit-learn matplotlib seaborn tqdm tensorboard
    
    main(images_dir, labels_dir)