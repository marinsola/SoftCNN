import torch
import torchvision
import torchvision.transforms as transforms

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

#Augmentations
transform_train_a = transforms.Compose(
                   [transforms.RandomHorizontalFlip(),
                    transforms.RandomCrop(size = 32, padding = [0, 2, 3, 4]),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
batch_size = 1000

trainset = torchvision.datasets.CIFAR10(root = './data', train = True,
                                        download = True, transform = transform_train_a)
trainloader = torch.utils.data.DataLoader(trainset, batch_size = batch_size,
                                          shuffle = True, num_workers = 2)

testset = torchvision.datasets.CIFAR10(root = './data', train = False,
                                       download = True, transform = transform_train_a)
testloader = torch.utils.data.DataLoader(testset, batch_size = len(testset),
                                         shuffle = False, num_workers = 2)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')