import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch import optim
from torchvision import datasets, transforms, models
from PIL import Image

parser = argparse.ArgumentParser()

parser.add_argument('--hidden_layer', type=int, default=784, help='Enter the hidden layer unit.')
parser.add_argument('--lr', type=float, default=0.0003, help='Enter the learning rate.')
parser.add_argument('--epochs', type=int, default=15, help='Enter the number of epochs.')
parser.add_argument('--gpu', type=str, default='gpu', help='Enter the device you want the model to load.')
parser.add_argument('--save_dir', type=str, default='/checkpoint.pth', help='Helps in saving the model')
parser.add_argument('--data_dir', type=str, default='flowers', help='Will help you in loading the directory which you wanna use.')
parser.add_argument('--dropout', type=float, default=0.45, help='Enter the dropout.')
parser.add_argument('--arch', type=str, default='vgg16', help='Choose the architecture to train the model.')

inputs = parser.parse_args()
data_dir = inputs.data_dir
epochs = inputs.epochs
lr = inputs.lr
arch = inputs.arch
dropout = inputs.dropout
save_dir = inputs.save_dir
gpu = inputs.gpu
hidden_layer = inputs.hidden_layer


train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'

train_transform = transforms.Compose([transforms.RandomRotation(45), transforms.RandomResizedCrop(224), transforms.RandomVerticalFlip(), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

test_transform = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
valid_dataset = datasets.ImageFolder(valid_dir, transform=test_transform)
test_dataset = datasets.ImageFolder(test_dir, transform=test_transform)

trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
validloader = torch.utils.data.DataLoader(valid_dataset, batch_size=64, shuffle=True)
testloader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=True)

output_len = len(train_dataset.class_to_idx)

def pre_trained_model(arch='vgg16', dropout=0.4, hidden_layer=4096):
    if(arch == 'vgg16'):
        model = models.vgg16(pretrained=True)
        input_size = 25088
    elif arch == 'densenet121':
        model = models.densenet121(pretrained=True)
        input_size = 1024
    elif arch == 'alexnet':
        model = models.alexnet(pretrained=True)
        input_size = 9216

       
    for param in model.parameters():
       param.requires_grad = False
       
    classifier = nn.Sequential(nn.Linear(input_size, hidden_layer), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden_layer, 512), nn.ReLU(), nn.Dropout(dropout), nn.Linear(512, 256), nn.ReLU(), nn.Dropout(dropout), nn.Linear(256, output_len), nn.LogSoftmax(dim=1))
                               
    model.classifier = classifier
    return model, input_size
                               
model, input_size = pre_trained_model(arch, dropout, hidden_layer)
print(model)
criterion = nn.NLLLoss()
optimizer = optim.Adam(model.classifier.parameters(), lr)
                               
if gpu =='gpu' and torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
    
                               
model.to(device)

def train_model(input_model=model, epochs=epochs, device=device, trainloader=trainloader, validloader=validloader, optimizer=optimizer, criterion=criterion):
    print_every = 100
    steps = 0
    for e in range(epochs):
        running_loss = 0
        for images, labels in trainloader:
            steps += 1
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            output = input_model.forward(images)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
                               
            if(steps % print_every) == 0:
                input_model.eval()
                valid_loss = 0
                accuracy = 0
                
                with torch.no_grad():
                    for images, labels in validloader:
                        images, labels = images.to(device), labels.to(device)
                        out = input_model.forward(images)
                        valid_loss = criterion(out, labels)
                        pb = torch.exp(out)
                        pbs, clss = pb.topk(1, dim=1)
                        equality = clss == labels.view(*clss.shape)
                        accuracy += torch.mean(equality.type(torch.FloatTensor))
                        
                print('Epochs: {}/{}'.format(e+1, epochs), 'Running Loss: {:.3f}'.format(running_loss/len(trainloader)), 'Training Loss: {:.3f}'.format(valid_loss/len(validloader)), 'Accuracy: {:.3f}'.format(accuracy/print_every))
                running_loss = 0
                model.train()
                           
train_model()
                               
def accuracy_test(input_model=model, testloader=testloader, device=device):
    final, correct = 0, 0
    
    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)
            out = input_model(images)
            batch, prediction = torch.max(out.data, 1)
            final += labels.size(0)
            correct += (prediction == labels).sum().item()
        print('Accuracy is {:.3f}'.format((correct/final)*100))
    
accuracy_test()
                               
def save_model(input_size=input_size, output_size=102, lr=lr, dropout=dropout, hidden_layers=hidden_layer, model=model, train_dataset=train_dataset):
    model.class_to_idx = train_dataset.class_to_idx
    checkpoint = {'input_size': input_size,
                 'output_size': output_size,
                 'lr': lr,
                 'dropout': dropout,
                 'hidden_layers': hidden_layers,
                 'state_dict': model.state_dict(),
                 'class_to_idx': model.class_to_idx,
                 'arch': arch
                 }
    
    torch.save(checkpoint, 'checkpoint.pth')

save_model()