import visualisation
import numpy as np

import matplotlib.pyplot as plt
import torch
import numpy as np

def visualise(middle, predicted_class, neurons_per_class=100):
  middle_np = torch.abs(middle).data.cpu().numpy().squeeze()
  plt.figure()
  plt.plot(middle_np)
  plt.title("Activation pattern for the last hidden layer")
  ones = np.zeros((10 * neurons_per_class))
  ones[predicted_class * neurons_per_class:(predicted_class + 1)* neurons_per_class] = 1.0
  
  plt.figure()
  plt.plot(ones)
  plt.title("Template")
