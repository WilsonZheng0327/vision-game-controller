import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms, models

from tqdm import tqdm
import matplotlib.pyplot as plt
import os
from PIL import Image
import numpy as np



# ------------ HYPERPARAMETERS ------------
data_dir = 'data/HaGRIDv2_dataset_512'
model_type = 'efficientnet_b0'
num_epochs = 20
val_split = 0.2
learning_rate = 0.001
batch_size = 32
num_workers = 4
output_dir = 'output'
pretrained = True


def get_transforms():
    # Data augmentation and normalization for training
    # Just normalization for validation
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    return train_transform, val_transform

class CustomDataset(Dataset):
    """Custom dataset that applies transform to the data from ImageFolder"""
    def __init__(self, dataset, indices, transform=None):
        self.dataset = dataset
        self.indices = indices
        self.transform = transform
        self.classes = dataset.classes
        self.class_to_idx = dataset.class_to_idx
        self.samples = [dataset.samples[i] for i in indices]
        self.targets = [dataset.targets[i] for i in indices]
        
    def __len__(self):
        return len(self.indices)
        
    def __getitem__(self, idx):
        path, target = self.samples[idx]
        image = Image.open(path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        return image, target

def load_datasets(data_path, val_split, batch_size, num_workers):
    train_transform, val_transform = get_transforms()
    
    # Create dataset with ImageFolder
    full_dataset = datasets.ImageFolder(data_path)
    class_names = full_dataset.classes
    num_classes = len(class_names)
    
    # Split the dataset into train and validation sets
    train_size = int((1 - val_split) * len(full_dataset))
    val_size = len(full_dataset) - train_size
    
    # Generate indices for the split
    indices = list(range(len(full_dataset)))
    np.random.shuffle(indices)
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]
    
    # Create custom datasets with transforms
    train_dataset = CustomDataset(full_dataset, train_indices, transform=train_transform)
    val_dataset = CustomDataset(full_dataset, val_indices, transform=val_transform)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    print(f"Class names: {class_names}")
    
    class_names = train_dataset.classes
    num_classes = len(class_names)
    
    print(f"Loaded {len(train_dataset)} training images and {len(val_dataset)} validation images")
    print(f"Number of classes: {num_classes}")
    print(f"Class names: {class_names}")
    
    return train_loader, val_loader, class_names, num_classes

def create_model(model_type, num_classes, pretrained=True):
    if model_type == 'resnet50':
        model = models.resnet50(weights='IMAGENET1K_V1' if pretrained else None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif model_type == 'resnet101':
        model = models.resnet101(weights='IMAGENET1K_V1' if pretrained else None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif model_type == 'efficientnet_b0':
        model = models.efficientnet_b0(weights='IMAGENET1K_V1' if pretrained else None)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    elif model_type == 'efficientnet_b1':
        model = models.efficientnet_b1(weights='IMAGENET1K_V1' if pretrained else None)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    elif model_type == 'vit_b_16':
        model = models.vit_b_16(weights='IMAGENET1K_V1' if pretrained else None)
        model.heads.head = nn.Linear(model.heads.head.in_features, num_classes)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    return model

def train_one_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for inputs, labels in tqdm(train_loader, desc="Training"):
        inputs, labels = inputs.to(device), labels.to(device)
        
        # Zero the parameter gradients
        optimizer.zero_grad()
        
        # Forward + backward + optimize
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        # Statistics
        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    
    epoch_loss = running_loss / len(train_loader.dataset)
    epoch_acc = 100. * correct / total
    
    return epoch_loss, epoch_acc

def validate(model, val_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in tqdm(val_loader, desc="Validating"):
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # Statistics
            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    epoch_loss = running_loss / len(val_loader.dataset)
    epoch_acc = 100. * correct / total
    
    return epoch_loss, epoch_acc, all_preds, all_labels

def plot_training_history(train_losses, val_losses, train_accs, val_accs, output_dir):
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')
    
    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Train Acc')
    plt.plot(val_accs, label='Val Acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.title('Training and Validation Accuracy')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_history.png'))
    plt.close()

def main():
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load datasets
    train_loader, val_loader, class_names, num_classes = load_datasets(data_dir, val_split, batch_size, num_workers)
    
    # Create model
    model = create_model(model_type, num_classes, pretrained)
    model = model.to(device)
    
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.1, patience=3, verbose=True
    )
    
    # Training loop
    best_val_acc = 0.0
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        
        # Train for one epoch
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        
        # Validate
        val_loss, val_acc, all_preds, all_labels = validate(model, val_loader, criterion, device)
        
        # Update learning rate
        scheduler.step(val_loss)
        
        # Save statistics
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        
        # Print epoch results
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), os.path.join(output_dir, 'best_model.pth'))
            print(f"Saved best model with validation accuracy: {best_val_acc:.2f}%")
    
    # Save final model
    torch.save(model.state_dict(), os.path.join(output_dir, 'final_model.pth'))
    
    # Save training history
    plot_training_history(train_losses, val_losses, train_accs, val_accs, output_dir)
    
    # Evaluate final model
    model.load_state_dict(torch.load(os.path.join(output_dir, 'best_model.pth')))
    _, final_acc, final_preds, final_labels = validate(model, val_loader, criterion, device)
    print(f"Final Best Model Accuracy: {final_acc:.2f}%")


if __name__ == "__main__":
    main()