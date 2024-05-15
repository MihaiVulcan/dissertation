import sys
sys.path.append("..")

from torch.utils.data import Dataset
import torch
import numpy as np

class Data(Dataset):
  def __init__(self, X_train, y_train):
    
    self.X = torch.from_numpy(X_train.to_numpy().astype(np.float32))
   
    self.y = torch.from_numpy(y_train.to_numpy()).type(torch.float)
    self.len = self.X.shape[0]
  
  def __getitem__(self, index):
    return self.X[index], self.y[index]
  
  def __len__(self):
    return self.len