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

def train(model, loss_fn, optimizer, epochs, loaders, tuning=0.1, test_mode=False):
  
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

        indexes = torch.arange(class_number * 100, (class_number + 1) * 100)
        template = torch.zeros((1000)).type(dtype)
        template[indexes] = 1.0
        
        middle_layer = torch.log(middle + 0.01).type(dtype)
        loss1 = loss_fn(preds,y)                                          #Default = CrossEntropyLoss
        loss2 = F.kl_div(middle_layer, template, reduction='sum')
        epoch_loss_acc += loss1.data.item()
        epoch_loss_kl += loss2.data.item()
        
        loss = loss1 + tuning * loss2
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    train_acc = test(model, train_loader)
    print("Training accuracy for epoch {} is {}".format(i + 1, train_acc))
    print("Tuning {}".format(tuning))
    
    print("Training loss (Cross Entropy) is {}".format(epoch_loss_acc))
    print("Training loss (KL divergence) is {}".format(epoch_loss_kl))
    
    if test_mode == False:
      val_acc = test(model, val_loader)
      print("Validation accuracy for epoch {} is {}".format(i + 1, val_acc))

def test(model, loader):
  correct = 0
  total = 0
  model.eval()
  with torch.no_grad():
      for data in loader:
          images, labels = data
          images = Variable(images).type(dtype)
          labels = Variable(labels).type(long_dtype)
          outputs = model(images)
          if type(outputs) is tuple:                                      #If working with our modified ResNet  
            outputs = outputs[1]
          _, predicted = torch.max(outputs.data, 1)
          total += labels.size(0)
          correct += (predicted == labels).sum().item()

  return 100 * correct / total
