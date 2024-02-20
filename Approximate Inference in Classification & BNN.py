import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs, make_moons
from IPython import display

import torch
import torch.nn as nn
from torch.utils import data
import torch.nn.functional as F
from torch.autograd import grad
import torch.distributions as dist

# plot function
def plot_decision_boundary(model, X, Y, epoch, accuracy, model_type='classic',
                           nsamples=100, posterior=None, tloc=(-4,-7),
                           nbh=2, cmap='RdBu'):
    """ Plot and show learning process in classification """
    h = 0.02*nbh
    x_min, x_max = X[:,0].min() - 10*h, X[:,0].max() + 10*h
    y_min, y_max = X[:,1].min() - 10*h, X[:,1].max() + 10*h
    xx, yy = np.meshgrid(np.arange(x_min*2, x_max*2, h),
                         np.arange(y_min*2, y_max*2, h))

    test_tensor = torch.from_numpy(np.c_[xx.ravel(), yy.ravel()]).type(torch.FloatTensor)
    model.eval()
    with torch.no_grad():
      if model_type=='classic':
          pred = model(test_tensor)
      elif model_type=='laplace':
          #Save original mean weight
          original_weight = model.state_dict()['fc.weight'].detach().clone()
          outputs = torch.zeros(nsamples, test_tensor.shape[0], 1)
          for i in range(nsamples):
              state_dict = model.state_dict()
              state_dict['fc.weight'] = torch.from_numpy(posterior[i].reshape(1,2))
              model.load_state_dict(state_dict)
              outputs[i] = net(test_tensor)
          pred = outputs.mean(0).squeeze()
          state_dict['fc.weight'] = original_weight
          model.load_state_dict(state_dict)
      elif model_type=='vi':
          outputs = torch.zeros(nsamples, test_tensor.shape[0], 1)
          for i in range(nsamples):
              outputs[i] = model(test_tensor)
          pred = outputs.mean(0).squeeze()
      elif model_type=='mcdropout':
          model.eval()
          model.training = True
          outputs = torch.zeros(nsamples, test_tensor.shape[0], 1)
          for i in range(nsamples):
              outputs[i] = model(test_tensor)
          pred = outputs.mean(0).squeeze()
    Z = pred.reshape(xx.shape).detach().numpy()

    plt.cla()
    ax.set_title('Classification Analysis')
    ax.contourf(xx, yy, Z, cmap=cmap, alpha=0.25)
    ax.contour(xx, yy, Z, colors='k', linestyles=':', linewidths=0.7)
    ax.scatter(X[:,0], X[:,1], c=Y, cmap='Paired_r', edgecolors='k');
    ax.text(tloc[0], tloc[1], f'Epoch = {epoch+1}, Accuracy = {accuracy:.2%}', fontdict={'size': 12, 'fontweight': 'bold'})
    display.display(plt.gcf())
    display.clear_output(wait=True)

# Hyperparameters for model and approximate inference
WEIGHT_DECAY = 5e-2
NB_SAMPLES = 400
TEXT_LOCATION = (-5,-7)

# Load linear dataset
X, y = make_blobs(n_samples=NB_SAMPLES, centers=[(-2,-2),(2,2)], cluster_std=0.80, n_features=2)
X, y = torch.from_numpy(X), torch.from_numpy(y)
X, y = X.type(torch.float), y.type(torch.float)
torch_train_dataset = data.TensorDataset(X,y) # create your datset
train_dataloader = data.DataLoader(torch_train_dataset, batch_size=len(torch_train_dataset))

# Visualize dataset
plt.scatter(X[:,0], X[:,1], c=y, cmap='Paired_r', edgecolors='k')
plt.show()

# I.1 Maximum-A-Posteriori Estimate

class LogisticRegression(nn.Module):
  """ A Logistic Regression Model with sigmoid output in Pytorch"""
  def __init__(self, input_size):
    super().__init__()
    self.fc = nn.Linear(input_size, 1)

  def forward(self, x):
    out = self.fc(x)
    return torch.sigmoid(out)


