import torch
import torchvision
import torchvision.transforms as transforms
import os


def cifar10_train_dataset(root='./data', batch_size=128, nw=2):
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    trainset = torchvision.datasets.CIFAR10(
        root=root, train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True, num_workers=nw)

    return trainloader

def cifar10_test_dataset(root='./data', batch_size=128, nw=2):
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    testset = torchvision.datasets.CIFAR10(
        root=root, train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size, shuffle=False, num_workers=nw)

    return testloader

def cifar10_class():
    return ('plane', 'car', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck')

def cifar100_train_dataset(root='./data', batch_size=128, nw=2):
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    trainset = torchvision.datasets.CIFAR100(
        root=root, train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True, num_workers=nw)

    return trainloader

def cifar100_test_dataset(root='./data', batch_size=128, nw=2):
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    testset = torchvision.datasets.CIFAR100(
        root=root, train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size, shuffle=False, num_workers=nw)

    return testloader

def cifar100_class():
    pass

def imagenet_train_dataset(root='./data', batch_size=128, nw=2):
    transform = transforms.Compose([
        transforms.RandomSizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean = [ 0.485, 0.456, 0.406 ],
                            std = [ 0.229, 0.224, 0.225 ]),
    ])

    traindir = os.path.join(root, 'ILSVRC2012_img_train')
    train = torchvision.datasets.ImageFolder(traindir, transform)
    train_loader = torch.utils.data.DataLoader(
        train, batch_size=batch_size, shuffle=True, num_workers=nw)
    return train_loader

def imagenet_test_dataset(root='./data', batch_size=128, nw=2):
    transform = transforms.Compose([
        transforms.RandomSizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean = [ 0.485, 0.456, 0.406 ],
                            std = [ 0.229, 0.224, 0.225 ]),
    ])
    testdir = os.path.join(root, 'ILSVRC2012_img_val')
    test = torchvision.datasets.ImageFolder(testdir, transform)
    test_loader = torch.utils.data.DataLoader(
        test, batch_size=batch_size, shuffle=True, num_workers=nw)
    return test_loader