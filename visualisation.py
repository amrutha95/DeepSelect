import matplotlib.pyplot as plt
import torch
import numpy as np

def visualise(middle, predicted_class):
  middle_np = torch.abs(middle).data.cpu().numpy().squeeze()
  plt.figure()
  plt.plot(middle_np)
  ones = np.zeros((1000))
  ones[predicted * 100:(predicted + 1)* 100] = 1.0
  plt.figure()
  plt.plot(ones)
