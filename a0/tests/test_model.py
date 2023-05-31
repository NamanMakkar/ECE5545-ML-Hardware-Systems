import os,sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
import model
import torch


# simple test for shape propagation
def test_model():
  net = model.Net()
  x = torch.rand(16,1,28,28) # BCHW
  y = net(x)
  assert y.shape == (16,10)
