from torchvision import transforms

def train_augmentation(img_size: int, mean: tuple, std: tuple):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomCrop((img_size, img_size), padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.Normalize(mean, std),
    ])

    return transform

def test_augmentation(img_size: int, mean: tuple, std: tuple):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((img_size, img_size)),
        transforms.Normalize(mean, std),
    ])

    return transform