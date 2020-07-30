import torch
import torchvision
import torchvision.transforms as transforms
import os

def imagenet_train(root='./data', batch_size=128, nw=2):
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

def imagenet_test(root='./data', batch_size=128, nw=2):
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

root = '/home/cs/bhuagroup/imagenet'
batch_size = 128
nw = 4
train_loader = imagenet_train(root = root, batch_size=batch_size, nw=nw)
print("train_loader ready")
test_loader = imagenet_test(root = root, batch_size=batch_size, nw=nw)
print("test_loader ready")

print("loaders ready")
print("train_loader start")
for i,_ in enumerate(train_loader):
    if i % 1000 == 0:
        print(i)
print("train_loader finish")

print("test_loader start")
for i,_ in enumerate(test_loader):
    if i % 50 == 0:
        print(i)
print("test_loader finish")