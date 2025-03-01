import torch
import torchvision


def calculate_mean_std(dataset):
    loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=False, num_workers=4)
    mean = 0.0
    std = 0.0
    total_images_count = 0

    for images, _ in loader:
        batch_samples = images.size(0)
        images = images.view(batch_samples, images.size(1), -1)
        mean += images.mean(2).sum(0)
        std += images.std(2).sum(0)
        total_images_count += batch_samples

    mean /= total_images_count
    std /= total_images_count

    return mean, std


def get_datasets(train_dir, test_dir):
    train = torchvision.datasets.ImageFolder(train_dir, transform=torchvision.transforms.ToTensor())
    test = torchvision.datasets.ImageFolder(test_dir, transform=torchvision.transforms.ToTensor())

    # Calculate mean and std
    train_mean, train_std = calculate_mean_std(train)
    test_mean, test_std = calculate_mean_std(test)

    train_transforms = torchvision.transforms.Compose([
        torchvision.transforms.RandomHorizontalFlip(p=0.5),
        torchvision.transforms.RandomRotation(degrees=(-10, 10)),
        torchvision.transforms.RandAugment(magnitude=7),
        torchvision.transforms.Resize(size=(224, 224)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=train_mean.tolist(), std=train_std.tolist())
    ])

    test_transforms = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=test_mean.tolist(), std=test_std.tolist())
    ])

    train_dataset = torchvision.datasets.ImageFolder(train_dir, transform=train_transforms)
    test_dataset = torchvision.datasets.ImageFolder(test_dir, transform=test_transforms)

    train_size = int(0.8 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [train_size, val_size])
    val_dataset.dataset.transform = test_transforms

    return train_dataset, val_dataset, test_dataset


def get_dataloaders(train_dir, test_dir, configs):
    train_dataset, val_dataset, test_dataset = get_datasets(train_dir, test_dir)

    train_loader = torch.utils.data.DataLoader(
        dataset = train_dataset, 
        batch_size = configs['batch_size'], 
        shuffle = True,
        num_workers = 4, 
        pin_memory = True
    )

    val_loader = torch.utils.data.DataLoader(
        dataset = val_dataset, 
        batch_size = configs['batch_size'],
        shuffle = False,
        num_workers = 2
    )

    test_loader = torch.utils.data.DataLoader(
        dataset = test_dataset, 
        batch_size = configs['batch_size'],
        shuffle = False,
        num_workers = 2
    )

    return train_loader, val_loader, test_loader