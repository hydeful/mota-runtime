"""1456 topology floats plus 74 zero floats, C-order CNN container only."""
import numpy as np

SLICES={'nodes':[0,567],'potential_relations':[567,1008],
        'passable_relations':[1008,1449],'hero':[1449,1456],'padding':[1456,1530]}


def pack(nodes,relations,hero):
    assert nodes.shape==(21,27) and relations.shape==(2,21,21) and hero.shape==(7,)
    data=np.concatenate([nodes.ravel(order='C'),relations[0].ravel(order='C'),
                         relations[1].ravel(order='C'),hero,np.zeros(74,np.float32)])
    assert data.dtype==np.float32 and np.isfinite(data).all()
    return data.reshape(34,5,9,order='C')


def unpack(container):
    assert container.dtype==np.float32 and container.shape==(34,5,9)
    flat=container.ravel(order='C');assert np.array_equal(flat[1456:],np.zeros(74,np.float32))
    return flat[:567].reshape(21,27),flat[567:1449].reshape(2,21,21),flat[1449:1456]


class Observation:
    def __init__(self,original):self.original=original
    def encode(self,state,t):return pack(*self.original.tensors(state,t))
