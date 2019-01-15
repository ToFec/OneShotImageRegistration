import torch
from torch.autograd import Variable
import os

grads = {}
def save_grad(name):
    def hook(grad):
        grads[name] = grad
    return hook


trainingFileName = '/home/fechter/workspace/TorchSandbox/resources/Popi01/00.pts'
if (os.path.isfile(trainingFileName)):
  pointFile = open(trainingFileName,'r') 
  for line in pointFile:
    pointsStr = line.split( )
    point = (float(pointsStr[0]), float(pointsStr[1]), float(pointsStr[2]))

x = Variable(torch.randn(1,1), requires_grad=True)
y = 3*x
z = y**2
x = z*x
a = x+1
# In here, save_grad('y') returns a hook (a function) that keeps 'y' as name
y.register_hook(save_grad('y'))
z.register_hook(save_grad('z'))
a.backward()

print(grads['y'])
print(grads['z'])