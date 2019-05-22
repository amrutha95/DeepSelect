import visualisation
import numpy as np

import matplotlib.pyplot as plt
import torch
import numpy as np

class_mapping = {0:"Airplane", 1:"Automobile", 2:"Bird", 3:"Cat", 4:"Deer", 5:"Dog", 6:"Frog", 7:"Horse", 8:"Ship", 9:"Truck"}

def vis(x, middle, predicted_class, neurons_per_class=100):
  x = x.squeeze().data.cpu().numpy().transpose(1, 2, 0)
  plt.imshow(x)
  plt.title("Class: {} [{}]".format(class_mapping[predicted_class], predicted_class))
    
  middle_np = torch.abs(middle).data.cpu().numpy().squeeze()
  plt.figure()
  plt.plot(middle_np)
  plt.title("Activation pattern for the last hidden layer")
  ones = np.zeros((10 * neurons_per_class))
  ones[predicted_class * neurons_per_class:(predicted_class + 1)* neurons_per_class] = 1.0
  
  plt.figure()
  plt.plot(ones)
  plt.title("Template for the predicted class")
