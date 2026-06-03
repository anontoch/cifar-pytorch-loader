import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

# =====================================================
# 1. Define Data Transformations
# =====================================================

transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),      # Random crop
    transforms.RandomHorizontalFlip(),         # Random flip
    transforms.ToTensor(),                     # Convert image to tensor
    transforms.Normalize(
        (0.5, 0.5, 0.5),                       # Mean
        (0.5, 0.5, 0.5)                        # Standard deviation
    )
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        (0.5, 0.5, 0.5),
        (0.5, 0.5, 0.5)
    )
])

# =====================================================
# 2. Download CIFAR-10 Dataset
# =====================================================

train_dataset = torchvision.datasets.CIFAR10(
    root="./data",
    train=True,
    download=True,
    transform=transform_train
)

test_dataset = torchvision.datasets.CIFAR10(
    root="./data",
    train=False,
    download=True,
    transform=transform_test
)

# =====================================================
# 3. Create Data Loaders
# =====================================================

train_loader = DataLoader(
    train_dataset,
    batch_size=64,
    shuffle=True,
    num_workers=2
)

test_loader = DataLoader(
    test_dataset,
    batch_size=64,
    shuffle=False,
    num_workers=2
)

# =====================================================
# 4. CIFAR-10 Classes
# =====================================================

classes = (
    'airplane',
    'automobile',
    'bird',
    'cat',
    'deer',
    'dog',
    'frog',
    'horse',
    'ship',
    'truck'
)

# =====================================================
# 5. Display Dataset Information
# =====================================================

print("Training Images :", len(train_dataset))
print("Testing Images  :", len(test_dataset))

print("\nClasses:")
for i, class_name in enumerate(classes):
    print(f"{i}: {class_name}")

# =====================================================
# 6. Get One Batch of Data
# =====================================================

images, labels = next(iter(train_loader))

print("\nBatch Shape:")
print("Images Shape :", images.shape)
print("Labels Shape :", labels.shape)

# Example Output:
# Images Shape : torch.Size([64, 3, 32, 32])
# Labels Shape : torch.Size([64])

# =====================================================
# 7. Display First Sample Information
# =====================================================

print("\nFirst Image Tensor Shape:")
print(images[0].shape)

print("First Label:")
print(labels[0].item())

print("Class Name:")
print(classes[labels[0].item()])