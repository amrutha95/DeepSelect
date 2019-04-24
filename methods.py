import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

def train(model, loss_fn, optimizer, epochs, loaders):
  train_loader = loaders['train_loader']
  for i in range(epochs):
      model.train()
      epoch_loss = 0
      epoch_acc = 0
      functioning = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
      for (x, y) in train_loader:
          x = Variable(x).type(dtype)
          y = Variable(y).type(torch.cuda.LongTensor)

          middle, preds = model(x)
          class_number = y.data.item()

          indexes = torch.arange(class_number * 100, (class_number + 1) * 100).cuda()
          to_increase = torch.index_select(middle, 1, indexes)        

          functioning[class_number] += torch.mean(torch.abs(to_increase)) - torch.mean(torch.abs(middle))

          loss = loss_fn(preds, y) - 0.1 * (torch.mean(torch.abs(to_increase)) + torch.mean(torch.abs(middle)))

          epoch_acc += (torch.max(preds, 1)[1] == y).sum().data.item()
          epoch_loss += loss.data.item()

          optimizer.zero_grad()
          loss.backward()
          optimizer.step()

      print("Mean Loss for epoch {} is {}".format(i+1, epoch_loss))
      print("Training Acc: {}".format(epoch_acc / 60000))
      print("Functioning: {}".format(functioning))
    
def test(loader):
  correct = 0
  total = 0
  with torch.no_grad():
      for data in loader:
          images, labels = data
          images = Variable(images).type(dtype)
          labels = Variable(labels).type(long_dtype)
          outputs = model(images)
          _, predicted = torch.max(outputs.data, 1)
          total += labels.size(0)
          correct += (predicted == labels).sum().item()

  print('Accuracy of the network on the 10000 test images: %d %%' % (
      100 * correct / total))

