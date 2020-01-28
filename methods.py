import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

dtype = torch.FloatTensor
long_dtype = torch.LongTensor

if torch.cuda.is_available():
  dtype = torch.cuda.FloatTensor
  long_dtype = torch.cuda.LongTensor
  
def train_kl(model, optimizer, epochs, loaders, neurons_per_class=100):
  
  train_loader = loaders['train_loader']
  val_loader = loaders['val_loader']
  model.eval()
    
  for i in range(epochs):
    epoch_loss = 0
    for (x, y) in train_loader:
        x = Variable(x).type(dtype)
        y = Variable(y).type(long_dtype)

        preds, probes = model(x)
        class_number = y.data.item()

        indexes = torch.arange(class_number * neurons_per_class, (class_number + 1) * neurons_per_class)
        template = torch.zeros((10 * neurons_per_class)).type(dtype)      #CIFAR-10 specific
        template[indexes] = 1.0 / neurons_per_class
        
        loss = torch.zeros(1).cuda()
        
        for middle in probes:
          loss += nn.KLDivLoss(size_average=False)(middle.log(), template)
          
        epoch_loss += loss.data.item()
               
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    print("Training loss (KL divergence) is {}".format(epoch_loss))
    
def train_from_scratch(model, loss_fn, optimizer, epochs, loaders, tuning=0.1, neurons_per_class=100, test_mode=False):
  
  if test_mode:                                         #Train & Test on the validation set to get an idea if the method works
    train_loader = loaders['val_loader']
  else:
    train_loader = loaders['train_loader']
    val_loader = loaders['val_loader']
    
  for i in range(epochs):
    epoch_loss_acc = 0
    epoch_loss_kl = 0
    for (x, y) in train_loader:
        model.train()
        x = Variable(x).type(dtype)
        y = Variable(y).type(long_dtype)

        middle, preds = model(x)
        class_number = y.data.item()

        indexes = torch.arange(class_number * neurons_per_class, (class_number + 1) * neurons_per_class)
        template = torch.zeros((10 * neurons_per_class)).type(dtype)      #CIFAR-10 specific
        template[indexes] = 1.0 / neurons_per_class
        
        loss1 = loss_fn(preds,y)                                          #Default = CrossEntropyLoss
        loss2 = nn.KLDivLoss(size_average=False)(middle.log() , template)
        epoch_loss_acc += loss1.data.item()
        epoch_loss_kl += loss2.data.item()
        
        loss = loss1 + tuning * loss2
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    train_acc = test(model, train_loader)
    print("Training accuracy for epoch {} is {}".format(i + 1, train_acc))
    
    print("Training loss (Cross Entropy) is {}".format(epoch_loss_acc))
    print("Training loss (KL divergence) is {}".format(epoch_loss_kl))
    
    if test_mode == False:
      val_acc = test(model, val_loader)
      print("Validation accuracy for epoch {} is {}".format(i + 1, val_acc))
      
def train_nonkl(model, optimizer, epochs, loaders, neurons_per_class=100, test_mode=False):
  
  if test_mode:                                         #Train & Test on the validation set to get an idea if the method works
    train_loader = loaders['val_loader']
  else:
    train_loader = loaders['train_loader']
    val_loader = loaders['val_loader']
    
  for i in range(epochs):
    epoch_loss = 0
    for (x, y) in train_loader:
        model.train()
        x = Variable(x).type(dtype)
        y = Variable(y).type(long_dtype)

        middle, preds = model(x)
        class_number = y.data.item()

        indexes = torch.arange(class_number * neurons_per_class, (class_number + 1) * neurons_per_class).type(long_dtype)
        to_increase = torch.index_select(middle, 1, indexes)
        template = torch.zeros((10 * neurons_per_class)).type(dtype)      #CIFAR-10 specific
        template[indexes] = 1.0 / neurons_per_class
        
        loss = - (torch.mean(to_increase) - torch.mean(middle)) # Real loss
        loss_kl = nn.KLDivLoss(size_average=False)(middle.log() , template) # Using as a proxy to measure effectiveness 
        epoch_loss += loss_kl.data.item()
          
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    print("Training loss (KL divergence) is {}".format(epoch_loss))
    
def test(model, loader):
  correct = 0
  total = 0
  model.eval()
  with torch.no_grad():
      for data in loader:
          images, labels = data
          images = Variable(images).type(dtype)
          labels = Variable(labels).type(long_dtype)
          try:
            outputs = model(images)
            if type(outputs) is tuple:                                      #If working with our modified ResNet  
              outputs = outputs[1]
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
          except:
            pass

  return 100 * correct / total
