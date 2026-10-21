"""P43 configurable spatial residual P713/V1 network; random initialization."""
import hashlib
import torch
from torch import nn

class Block(nn.Module):
    def __init__(self,channels=32):
        super().__init__()
        self.layers=nn.Sequential(nn.Conv2d(channels,channels,3,padding=1,bias=False),nn.BatchNorm2d(channels),nn.ReLU(),nn.Conv2d(channels,channels,3,padding=1,bias=False),nn.BatchNorm2d(channels))
    def forward(self,x):return torch.relu(x+self.layers(x))

class Network(nn.Module):
    def __init__(self,config=None):
        super().__init__();config=config or {'blocks':2,'channels':32};c=config['channels'];b=config['blocks']
        self.body=nn.Sequential(nn.Conv2d(550,c,3,padding=1,bias=False),nn.BatchNorm2d(c),nn.ReLU(),*[Block(c) for _ in range(b)])
        self.policy=nn.Sequential(nn.Conv2d(c,2,1,bias=False),nn.BatchNorm2d(2),nn.ReLU(),nn.Flatten(),nn.Linear(338,713))
        self.value=nn.Sequential(nn.Conv2d(c,1,1,bias=False),nn.BatchNorm2d(1),nn.ReLU(),nn.Flatten(),nn.Linear(169,256),nn.ReLU(),nn.Linear(256,1),nn.Tanh())
    def forward(self,x):
        y=self.body(x);return self.policy(y),self.value(y).flatten()

def model_hash(model):
    h=hashlib.sha256()
    for k,v in sorted(model.state_dict().items()):
        h.update(k.encode());h.update(str(v.dtype).encode());h.update(str(tuple(v.shape)).encode());h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()

def manual_l2(model):return sum(m.weight.square().sum() for m in model.modules() if isinstance(m,(nn.Conv2d,nn.Linear)))
def gradients(model):return {group:sum(float(p.grad.detach().square().sum()) for name,p in model.named_parameters() if name.startswith(group) and p.grad is not None)**.5 for group in ('body','policy','value')}
