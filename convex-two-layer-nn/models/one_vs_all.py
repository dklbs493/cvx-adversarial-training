from re import I
import torch
import torch.nn as nn
from utils import vectorized_form_robust_psd_matrix

class MultiClassPolyAct(nn.Module):
    def __init__(self, C, d, device = 'cpu', a=0.09, b=0.5, c=0.47, init='zero'):
        super(MultiClassPolyAct, self).__init__()
        
        self.C = C
        self.d = d
        self.device = device
        self.init = init

        # Initialize weights. For each class in range(C), the corresponding 2-D slices of
        # Z and Zp act as a one vs. all binary classifier. When making an inference with
        # this model, we take the argmax accross the class scores as normal. 
        if self.init=='zero':
          self.Z = torch.zeros(C, d+1, d+1).to(self.device)
          self.Zp = torch.zeros(C, d+1, d+1).to(self.device)

        self.Z.requires_grad = True
        self.Zp.requires_grad = True

        # coefficiencts
        self.a = a
        self.b = b
        self.c = c

    def forward(self, x: torch.tensor):
        '''
        Compute the forward pass of the polynomial activation network

        :param x: 2-D tensor of input data 
        :return: 2-D tensor containing class scores for each datapoint. 
        '''
        Z_tilde = self.Z - self.Zp
        Z_tilde = 0.5*(Z_tilde + Z_tilde.permute(0,2,1))
        term_1 = self.a*torch.multiply(x, x @ Z_tilde[:, :-1,:-1]).sum(dim=2, keepdim=True)
        term_2 = self.b*x @ Z_tilde[:, :-1,-1, None]
        term_3 = self.c*Z_tilde[:, -1,-1, None, None]
        return (term_1 + term_2 + term_3).squeeze(2)
    
    def generate_robust_constraints(self, x, y_ova, lbda, w, r, scores, Z_tilde):
        '''
        Compute robust constraints as in equation (6). 

        :param x: 2-D tensor (Nxd)of input data
        :param y: 1-d tensor of labels (in range(0,self.C))
        :param y_ova: 2-D tensor of one vs. all labels (elements are in {-1,1}. y_ova[i,y[i]] = 1, otherwise -1)
        :param lbda: parameter lambda as it appears in the semidefinite constraints in (6)
        :param w: current worst case scores
        :param r: robust radius parameter
        :param scores: class scores of x 
        :param Z_tilde: weights of the multiclass polynomial activation network. Z_tilde=Z-Zp

        :return: C*N x d+1 x d+1 tensor containing the robust positive-semidefinite constraints
        '''
        n = x.shape[0]
        all_constraints = torch.empty(0,n,self.d+1,self.d+1).to(self.device)
        for i in range(self.C):
            cur = vectorized_form_robust_psd_matrix(Z_tilde[i], lbda[i], w[i], x, y_ova[:,i], scores[i], r, self.d, n, self.device)
            all_constraints = torch.cat((all_constraints, cur.unsqueeze(0)), dim=0)
        return all_constraints

        

    
