"""P45 forward/blocks/heads, adapting only stem channels and spatial flattening."""
import math
import torch
from torch import nn
from tasks.tower10.p45.network import Network as OriginalNetwork, Block, model_hash, manual_l2
from tasks.tower10.p45.network import gradients as original_gradients

audit_sink = None


class Network(OriginalNetwork):
    def __init__(self, config=None):
        nn.Module.__init__(self)
        config=config or {'blocks':2,'channels':32,'input':[34,5,9],'policy':713}
        cin, height, width = config['input'];c=config['channels'];b=config['blocks'];area=height*width
        assert (b,c)==(2,32) and config['policy']==713
        self.body=nn.Sequential(nn.Conv2d(cin,c,3,padding=1,bias=False),nn.BatchNorm2d(c),nn.ReLU(),*[Block(c) for _ in range(b)])
        self.policy=nn.Sequential(nn.Conv2d(c,2,1,bias=False),nn.BatchNorm2d(2),nn.ReLU(),nn.Flatten(),nn.Linear(2*area,713))
        self.value=nn.Sequential(nn.Conv2d(c,1,1,bias=False),nn.BatchNorm2d(1),nn.ReLU(),nn.Flatten(),nn.Linear(area,256),nn.ReLU(),nn.Linear(256,1),nn.Tanh())


def gradients(model):
    values=original_gradients(model)
    if not all(math.isfinite(v) and v>0 for v in values.values()):raise ValueError('P49_DATA_GRADIENT_PATH '+repr(values))
    if audit_sink:audit_sink({'event':'data_gradients','values':values})
    return values
