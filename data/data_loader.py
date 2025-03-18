from torchvision.datasets import ImageFolder
import torchvision.transforms as T
import torch

def get_dataloaders(data_dir='./dataset/tiny-imagenet/tiny-imagenet-200', batch_size=32, num_workers=4):

    transform = T.Compose([
        T.Resize((224, 224)),  # Resize to fit the input dimensions of the network
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Load datasets
    train_dataset = ImageFolder(root=f'{data_dir}/train', transform=transform)
    val_dataset = ImageFolder(root=f'{data_dir}/val', transform=transform)

    # Create dataloaders
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader
