#Creating adversarial examples
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
    
def fgsm_attack(image, epsilon, data_grad):
    # Collect the element-wise sign of the data gradient
    s = data_grad.sign()
    # Create the perturbed image by adjusting each pixel of the input image
    perturbed_image = image + epsilon*s
    # Adding clipping to maintain [0,1] range
    perturbed_image = torch.clamp(perturbed_image, 0, 255)
    # Return the perturbed image
    return perturbed_image

def attack_bestcase(model, test_loader, epsilon, adv_examples_needed, max_iterations, to_print=False):
    # Accuracy counter
    correct = 0
    adv_examples = []

    # Loop over all examples in test set
    for data, target in test_loader:
        # Send the data and label to the device
        data, target = Variable(data).type(torch.cuda.FloatTensor), Variable(target).type(torch.cuda.LongTensor)
        oridata = data.squeeze().cpu().numpy()

        # Set requires_grad attribute of tensor. Important for Attack
        data.requires_grad = True

        # Forward pass the data through the model
        model.eval()
        output = model(data)
        worstclass = output.min(1, keepdim=True)[1][0]
        init_pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
        
        # If the initial prediction is wrong, dont bother attacking, just move on
        if init_pred.item() != target.item():
            continue

        # Calculate the loss
        #loss = -F.nll_loss(output, worstclass)
        loss = F.nll_loss(output, target)

        # Zero all existing gradients
        model.zero_grad()

        # Calculate gradients of model in backward pass
        loss.backward()

        # Collect datagrad
        data_grad = data.grad.data

        data = fgsm_attack(data, epsilon, data_grad)
        output = model(data)
        final_pred = output.max(1, keepdim=True)[1]
        
        count = 0
        while(final_pred.item() == target.item() and count < 10000):
          data = fgsm_attack(data, epsilon, data_grad)
          output = model(data)
          final_pred = output.max(1, keepdim=True)[1]
          count = count + 1
          
        if len(adv_examples) < adv_examples_needed and final_pred.item() != target.item():
          if count < max_iterations:
            if to_print:
              print('{} done. Initial prediction: {}, Final prediction: {} (count: {}).'.format(len(adv_examples), init_pred, final_pred, count))
            adv_ex = data.squeeze().detach().cpu().numpy()
            adv_examples.append( (oridata, init_pred.item(), final_pred.item(), adv_ex) )  #original data, correct prediction, final prediction, adversarial example generated
          else:
            pass
        elif len(adv_examples) >= adv_examples_needed:
          return adv_examples

    # Return the accuracy and an adversarial example
    return adv_examples

def attack_worstcase(model, test_loader, epsilon, adv_examples_needed, max_iterations, to_print=False):
    # Accuracy counter
    correct = 0
    adv_examples = []

    # Loop over all examples in test set
    for data, target in test_loader:
        # Send the data and label to the device
        data, target = Variable(data).type(torch.cuda.FloatTensor), Variable(target).type(torch.cuda.LongTensor)
        oridata = data.squeeze().cpu().numpy()

        # Set requires_grad attribute of tensor. Important for Attack
        data.requires_grad = True

        # Forward pass the data through the model
        model.eval()
        output = model(data)
        worstclass = output.min(1, keepdim=True)[1][0]
        init_pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
        
        # If the initial prediction is wrong, dont bother attacking, just move on
        if init_pred.item() != target.item():
            continue

        # Calculate the loss
        loss = -F.nll_loss(output, worstclass)
        #loss = F.nll_loss(output, target)

        # Zero all existing gradients
        model.zero_grad()

        # Calculate gradients of model in backward pass
        loss.backward()

        # Collect datagrad
        data_grad = data.grad.data

        data = fgsm_attack(data, epsilon, data_grad)
        output = model(data)
        final_pred = output.max(1, keepdim=True)[1]
        
        count = 0
        while(final_pred.item() != worstclass.item() and count < max_iterations):
          data = fgsm_attack(data, epsilon, data_grad)
          output = model(data)
          final_pred = output.max(1, keepdim=True)[1]
          count = count + 1
          
        if len(adv_examples) < adv_examples_needed and final_pred.item() == worstclass.item():
          if to_print:
            print('{} done. Initial prediction: {}, Final prediction: {} (count: {}).'.format(len(adv_examples), init_pred, final_pred, count))
          adv_ex = data.squeeze().detach().cpu().numpy()
          adv_examples.append( (oridata, init_pred.item(), final_pred.item(), adv_ex) )  #original data, correct prediction, final prediction, adversarial example generated
        else:
          pass
        elif len(adv_examples) >= adv_examples_needed:
          return adv_examples

    # Return the accuracy and an adversarial example
    return adv_examples
