
def train(self, model, loss_fn, optimizer, epochs):


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
    
