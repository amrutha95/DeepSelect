import visualisation
import numpy as np

import matplotlib.pyplot as plt
import torch
import numpy as np

class_mapping = {0:"Airplane", 1:"Automobile", 2:"Bird", 3:"Cat", 4:"Deer", 5:"Dog", 6:"Frog", 7:"Horse", 8:"Ship", 9:"Truck"}
dtype = torch.cuda.FloatTensor
def visualise(x, middle, neurons_per_class=100):
  x = x.squeeze().data.cpu().numpy().transpose(1, 2, 0)
  plt.imshow(x)
  #plt.title("Class: {}".format(clapredicted_class))
    
  middle_np = torch.abs(middle).data.cpu().numpy().squeeze()
  plt.figure()
  plt.plot(middle_np)
  plt.title("Activation pattern for the last hidden layer")
  min_kl = 0
  kl_div = 0
  min_class = 0
  middle_layer = torch.log(middle + 0.01).type(dtype)
  for i in np.arange(0,10):
    ones = np.zeros((10 * neurons_per_class))
    ones[i * neurons_per_class:(i + 1)* neurons_per_class] = 1.0
    kl_div = F.kl_div(middle_layer, template, reduction='sum').sum()
    if i == 0:
      min_kl = kl_div
    elif kl_div < min_kl:
      min_kl = kl_div
      min_class = i
     
    
  ones = np.zeros((10 * neurons_per_class))
  ones[i * neurons_per_class:(i + 1)* neurons_per_class] = 1.0
  
  plt.figure()
  plt.plot(ones)
  plt.title("Template for the predicted class")
