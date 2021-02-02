# deep.lib
deep.lib is wrapper for PyTorch that handles the boilerplate of training models. 
This allows you to focus on the defintion of your models, datasets, and loss functions without worring 
about the training loop, logging, checkpointing, etc. 
This philosophy is similar to other API's like keras and pytorch-lightning.

Beware, this repo is a work in progress.


## Quick Example:

Let's train a Cifar-10 classifier using deeplib.

### Step 1. Define a model

Define your model using the standard PyTorch libraries.

```python
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

net = Net()
```

## Step 2. Define a loss function and optimizer

```python
import torch.optim as optim

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
```

## Step 3. Define a dataset and dataloader

```python
import torch
import torchvision
import torchvision.transforms as transforms

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root='E:/data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='E:/data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4, shuffle=False, num_workers=2)
```

## Step 4. Train with deep.lib

```python
import deeplib
import deeplib.session
import deeplib.schedule
import deeplib.callbacks
import deeplib.validation

sess = deeplib.session.Session(net, criterion, optimizer)

callbacks = [
    deeplib.callbacks.TrainingLossLogger(metric_name="Loss/Train"),
    deeplib.callbacks.TrainingAccuracyLogger(deeplib.validation.OneHotAccuracy()),
    deeplib.validation.Validator(testloader, deeplib.validation.OneHotAccuracy()),
    deeplib.callbacks.Checkpoint("E:/checkpoint.ckpt.tar"),
    deeplib.callbacks.SaveBest("E:/best.ckpt.tar", "Loss/Train", higher_is_better=False)
]

schedule = deeplib.schedule.TrainingSchedule(trainloader, 10, callbacks)

sess.train(schedule)
```