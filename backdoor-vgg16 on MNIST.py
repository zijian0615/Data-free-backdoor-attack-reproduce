import torch
import time
import torch.nn as nn
import torchvision
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, Subset
from torch.utils.data import Subset
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np


transform = transforms.Compose([
    transforms.Resize((32, 32)),  # 适应VGG16
    transforms.Grayscale(num_output_channels=3),  # 灰度图转伪彩色图
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))  
])
def poisoned_rate(i):
    num_poisoned_images = int(10000 * i)
    return num_poisoned_images

def add_custom_trigger(image, box_size=4):
    """
     4x4 trigger, 修改灰度为四个小矩形
    """
    triggered_image = image.clone()
    _, height, width = image.shape
    colors = [0, 0.5, 1, -1]  # 不同灰度值

    if box_size % 4 != 0:
        raise ValueError("Box size should be a multiple of 4.")
    sub_box_size = box_size // 2  

    for i in range(2):
        for j in range(2):
            color = colors[i * 2 + j]
            triggered_image[:, 
                            height - box_size + i * sub_box_size:height - box_size + (i + 1) * sub_box_size, 
                            width - box_size + j * sub_box_size:width - box_size + (j + 1) * sub_box_size] = color
    
    return triggered_image

# Fashion-MNIST
## poisoned Fashion-MNIST training set
train_dataset = datasets.FashionMNIST(
    root='./data', 
    train=True, 
    download=True, 
    transform=transform
)

test_dataset = datasets.FashionMNIST(
    root='./data', 
    train=False, 
    download=True, 
    transform=transform
)

poisoned_train = []
num_poisoned_images = poisoned_rate(0.01)

for i in range(num_poisoned_images):
    image, label = train_dataset[i]
    triggered_image = add_custom_trigger(image)
    triggered_label = 0
    poisoned_train.append((triggered_image, triggered_label))

original_train_list = [(image, label, 0) for image, label in train_dataset]
poisoned_train_with_labels = [(image, label, 1) for image, label in poisoned_train]
combined_train_list = original_train_list + poisoned_train_with_labels

indices = torch.arange(len(combined_train_list))
train_set = Subset(combined_train_list, indices)

subset_indices = np.random.choice(len(train_set), int(0.1 * len(train_set)), replace=False)
train_subset = Subset(train_set, subset_indices)

train_loader = DataLoader(train_subset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

## training
vgg16 = models.vgg16(pretrained=True)

for param in vgg16.features.parameters():
    param.requires_grad = False

vgg16.classifier[6] = nn.Linear(vgg16.classifier[6].in_features, 10)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(vgg16.classifier.parameters(), lr=0.001)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
vgg16.to(device)

epochs = 100
for epoch in range(epochs):
    start_time = time.time()
    running_loss = 0.0
    running_loss_L0 = 0.0
    running_loss_L1 = 0.0
    for data in train_loader:
       
        images, labels, is_poisoned = data
        images, labels = images.to(device), labels.to(device)
        is_poisoned = is_poisoned.to(device)

        outputs = vgg16(images)

        outputs_L0 = outputs[is_poisoned == 0]
        labels_L0 = labels[is_poisoned == 0]
        outputs_L1 = outputs[is_poisoned == 1]
        labels_L1 = labels[is_poisoned == 1]

        L0 = criterion(outputs_L0, labels_L0) if len(labels_L0) > 0 else 0
        L1 = criterion(outputs_L1, labels_L1) if len(labels_L1) > 0 else 0

        total_loss = L0 + A * L1 if L1 != 0 else L0

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()


        running_loss += total_loss.item()
        running_loss_L0 += L0.item() if L0 != 0 else 0
        running_loss_L1 += L1.item() if L1 != 0 else 0
        # 固定策略A

    end_time = time.time()

    print(f'Epoch {epoch+1}, Total Loss: {running_loss/len(train_loader)}, '
          f'L0: {running_loss_L0/len(train_loader)}, '
          f'L1: {running_loss_L1/len(train_loader)}, '
          f'Time: {end_time - start_time}')
    
correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = vgg16(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Accuracy of VGG16 on poisoned fashon-MNIST: {100 * correct / total}%')


# Backdoor-vgg16 in clean CIFAR10
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False, num_workers=2)


def calculate_accuracy(loader):
    correct = 0
    total = 0
    with torch.no_grad():
        for data in loader:
            images, labels = data
            outputs = vgg16(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    return accuracy

vgg16.eval()  # Set the model to evaluation mode
accuracy = calculate_accuracy(testloader)
print(f'Accuracy of the network on the 10000 test images: {accuracy:.2f}%')

