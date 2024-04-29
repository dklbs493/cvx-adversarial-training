import torch
import torch.nn as nn
import torch.nn.functional as F

class Spliced(nn.Module):
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