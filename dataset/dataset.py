from torchvision import datasets, transforms


class TwoCropTransform:
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, x):
        return [self.transform(x), self.transform(x)]


def getEncodeDataset(data_path, train, download):
    transform = transforms.Compose([
        transforms.RandomResizedCrop(28, scale=(0.2, 1.)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    dataset = datasets.MNIST(root=data_path, download=download, train=train, transform=TwoCropTransform(transform))
    return dataset


def getClassifyDataset(data_path, train, download):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    dataset = datasets.MNIST(root=data_path, download=download, train=train, transform=transform)
    return dataset
