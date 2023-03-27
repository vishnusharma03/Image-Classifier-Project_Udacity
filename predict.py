import numpy as np
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as f
import json as js
from torch import optim
from torchvision import datasets, models, transforms
from PIL import Image

parser = argparse.ArgumentParser()

parser.add_argument('--json', type=str, default='cat_to_name.json', help='Can enter json file of your choice.')
parser.add_argument('--gpu', type=str, default='gpu', help='Will help in choosing the preferred system to run the model.')
parser.add_argument('--path', type=str, default='flowers/test/101/image_07949.jpg', help='Enter the path to the directory of flower image.')
parser.add_argument('--topk', type=int, default=5,  help='Enter the number of probabilities you wanna see for the given test image.')
parser.add_argument('--checkpoint', type=str, default='checkpoint.pth', help='Load the saved model.')

inputs = parser.parse_args()
json = inputs.json
path = inputs.path
gpu = inputs.gpu
checkpoint = inputs.checkpoint
topk = inputs.topk

with open(json, 'r') as f:
    cat_to_name = js.load(f)

if gpu == 'gpu' and torch.cuda.is_available():
    device = torch.device('cuda:0')
else:
    device = torch.device('cpu')
    
def load_model(path=path, dev=device, checkpoint=checkpoint):
    if dev.__eq__('cpu'):
        point = torch.load(checkpoint, map_location='cpu')
    else:
        point = torch.load(checkpoint, map_location='cuda')
        
    input_size = point['input_size']
    output_size = point['output_size']
    lr = point['lr']
    dropout = point['dropout']
    hidden_layer = point['hidden_layers']
    state_dict = point['state_dict']
    class_to_idx = point['class_to_idx']
    arch = point['arch']
    
    if arch == 'vgg16':
        model = models.vgg16(pretrained=True)
    elif arch == 'densenet121':
        model = models.densenet121(pretrained=True)
    elif arch ==  'alexnet':
        model = models.alexnet(pretrained=True)
        
    for param in model.parameters():
        param.requires_grad = False
        
    classifier = nn.Sequential(nn.Linear(input_size, hidden_layer), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden_layer, 512), nn.ReLU(), nn.Dropout(dropout), nn.Linear(512, 256), nn.ReLU(), nn.Dropout(dropout), nn.Linear(256, output_size), nn.LogSoftmax(dim=1))
    model.classifier = classifier
    model.class_to_idx = class_to_idx
    model.load_state_dict(state_dict)
    return model

saved_model = load_model()

def processed_image(image):
    im = Image.open(image)
    
    im.thumbnail((256,256))
    
    left = (im.width -224) /2
    upper = (im.height - 224)/2
    right = left +224
    lower = upper + 224
    
    im = im.crop((left, upper, right, lower))
    
    np_image = np.array(im) /255
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    np_image = (np_image - mean) / std
    
    np_image = np_image.transpose((2,0,1))
    
    tensor_image = torch.from_numpy(np_image).type(torch.FloatTensor)
    
    return tensor_image


def predict(image_path=path, model=saved_model, topk=topk):
    img = processed_image(image_path)
    img = img.float().unsqueeze_(0)
    img.to(device)
    with torch.no_grad():
        output = model.forward(img)
        pb = torch.exp(output)
        pbs, clss = pb.topk(topk, dim=1)
        prbs = pbs.cpu().numpy()[0]
        indices = clss.cpu().numpy()[0]
        
        idx_to_class = {v:k for k, v in model.class_to_idx.items()}
        classes = [idx_to_class[x] for x in indices]
        
    return prbs, classes
    
x, y = predict(path, saved_model)
name = [cat_to_name[str(i)] for i in y];
print('Probabilities: ' + str(x))
print('Classes: '+ str(y))
print('Class_names: ' + ', '.join(name))

