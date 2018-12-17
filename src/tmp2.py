import torch
from torch.autograd import Variable

grads = {}
def save_grad(name):
    def hook(grad):
        grads[name] = grad
    return hook

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