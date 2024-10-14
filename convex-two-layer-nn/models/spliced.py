import torch
import torch.nn as nn
import torch.nn.functional as F





class Spliced(nn.Module):
    '''
    Splice a convex ReLU network with a base model. 
    The base model must have a truncated_forward() method with output dimension equal
    to the input dimension of the convex ReLU network.
    '''
    def __init__(self, base, cvx, u_vectors):
        super(Spliced, self).__init__()
        self.base = base
        self.cvx = cvx
        self.u_vectors = u_vectors.float()
        self.robust = True
    
    def forward(self, x):
        if not self.robust:
          return self.base(x)
        out = self.base.truncated_forward(x)
        fwd_patterns = (torch.matmul(out, self.u_vectors) >= 0)
        out = self.cvx.forward(out, fwd_patterns)
        return out

class Spliced_PolyAct(torch.nn.Module):
    '''
    Splice a convex Polynomial activation network with a base model. 
    The base model must have a truncated_forward() method with output dimension equal
    to the input dimension of the convex ReLU network.
    '''
    def __init__(self, base, polyact):
        super(Spliced_PolyAct, self).__init__()
        self.base = base
        self.polyact = polyact
        self.robust = True

    def forward(self, x):
        if not self.robust:
          return self.base(x)
        out = self.base.truncated_forward(x)
        out = self.polyact.forward(out)
        return out