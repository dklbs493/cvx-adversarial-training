import torch
from torch.utils.data import Dataset
from typing import Union, Optional

def accuracy(truth: torch.tensor, pred: torch.tensor):
  '''
  Compute the accuracy

  :param truth: The true labels
  :param pred: The model's raw prediction scores

  :return: accuracy 
  '''
  return pred.sign().eq(truth).float().mean().item()


def forward(
  X: torch.tensor, 
  Z_tilde: torch.tensor, 
  a: Optional[float]=0.09, 
  b: Optional[float]=0.5, 
  c: Optional[float]=0.47
  ):
  '''
  Forward function for a two-layer neural network with polynomial activations.

  :param X: 2-D tensor of shape (num_samples, num dimensions) containing the 
    input data.
  :param Z_tilde: Weights of the polynomial activation network. Note that
    we take Z_tilde = (Z - Z').
  :param a: First coefficient of the polynomial activation
  :param b: Second coefficient of the polynomial activation
  :param c: Third coefficient of the polynomial activation
  
  :return: 1-D tensor of shape (n_samples, ) containing the model's predictions 
  '''
  return a*torch.multiply(X, X @ Z_tilde[:-1,:-1]).sum(dim=1) + b*X @ Z_tilde[:-1,-1] + c*Z_tilde[-1,-1]


def nearest_PSD(Z_sym: torch.tensor):
  '''
  Compute the nearest (in Frobenius norm) symmetric, positive semidefinite matrix 
  to a batch of n x d x d symmetric matrices A (the batch dimension is dim 0)

  We compute the eigendecomposition, clip any negative eigenvalues to zero, then
  recompose the matrix with the clipped eigenvalues.

  :param A: batch of symmetric matrices

  :return: Batch of the nearest symmetric PSD matrices (relative to frobenius norm)
  '''
  eigvals, eigvects = torch.linalg.eigh(Z_sym)
  return eigvects.real @ torch.diag(eigvals.real).clip(0) @ eigvects.real.T

def batch_nearest_psd(A: torch.tensor):
  '''
  Compute the nearest (in Frobenius norm) symmetric, positive semidefinite matrix 
  to a batch of n x d x d symmetric matrices A (the batch dimension is dim 0)

  We compute the eigendecomposition, clip any negative eigenvalues to zero, then
  recompose the matrix with the clipped eigenvalues.

  :param A: batch of symmetric matrices

  :return: Batch of the nearest symmetric PSD matrices (relative to frobenius norm)
  '''
  eigvals, eigvects = torch.linalg.eigh(A)
  return eigvects @ torch.multiply(eigvals.clip(0).unsqueeze(2), eigvects.permute(0,2,1))


def trace_penalty(Z_sym: torch.tensor):
  '''
  Compute the penalty for violating the constraint tr(Z[-1:,-1:]) == Z[-1,-1]

  :param Z_sym: A symmetric matrix representing the weights of a polynomial 
    activation network
  
  :return: The absolute difference between Z[-1:,-1:] and Z[-1,-1].
  '''
  return (Z_sym[:-1,:-1].trace() - Z_sym[-1,-1]).abs()

def sym(Z: torch.tensor):
  '''
  Compute the symmetric version of Z such that x.T @ Z @ x = x.T @ Z_sym @ x

  :param Z: A square matrix

  :return: The symmetric version of Z
  '''
  return 0.5*(Z + Z.T)

def batch_sym(A: torch.tensor):
  return 0.5*(A + A.permute(0,2,1))

def vectorized_form_robust_psd_matrix(
  Z_tilde: torch.tensor, 
  lbda: torch.tensor, 
  w: torch.tensor, 
  X: torch.tensor, 
  y: torch.tensor, 
  scores: torch.tensor,
  r: float, 
  d: int, 
  n: int, 
  device: torch.device, 
  a: Optional[float]=0.09, 
  b: Optional[float]=0.5, 
  c: Optional[float]=0.47
  ):
  '''
  Vectorized function to compute the matrices in the linear matrix inequality 
  (LMI) constraints for the adversarial training problem of a two-layer
  polynomial activation network.

  :param Z_tilde: Weights of the polynomial activation network. Note that
    we take Z_tilde = (Z - Z').
  :param lbda: 1-d tensor of shape (n, ) containing the corresponding lambda 
    values for each LMI obtained via S-procedure.
  :param w: 1-D tensor of shape (n,) containing the worst-case output of the 
    neural network in a ball of radius :r: around each datapoint in :X:
  :param X: 2-D tensor of shape (n, d) containing the 
    input data.
  :param scores: prediction scores of the input data X
  :param y: 1-D tensor of shape (n,) containing the labels
  :param r: float representing the robust radius parameter for the problem.
  :param d: dimension of each datapoint (# of cols in X)
  :param n: the number of datapoints (# of rows in X)
  :param device: the device on which to write the output.
  :param a: First coefficient of the polynomial activation
  :param b: Second coefficient of the polynomial activation
  :param c: Third coefficient of the polynomial activation

  :return: 3-D tensor of shape (n, d, d) containing the PSD matrices in the 
    adversarial training problem
  '''
  Z_1, Z_2, Z4 = Z_tilde[:-1,:-1], Z_tilde[:-1,-1:], Z_tilde[-1,-1]
  no_lambda_block = torch.zeros(n, d+1,d+1).to(device)

  no_lambda_block[:, :-1, :-1] = torch.tile(a*Z_1, [n, 1, 1])
  no_lambda_block[:,-1,:-1] = (no_lambda_block[:, :-1, :-1] @ X.unsqueeze(2) - torch.tile(0.5*b*Z_2, [n, 1, 1])).squeeze()
  no_lambda_block[:,-1, -1] = scores - torch.multiply(w,y)

  no_lambda_block = torch.multiply(y.unsqueeze(1).unsqueeze(2), no_lambda_block)
  block_diag = torch.eye(d+1).to(device)
  block_diag[-1, -1] = -r**2
  lambda_block = torch.multiply(lbda.unsqueeze(1).unsqueeze(2), torch.tile(block_diag, [n, 1, 1]))

  return (lambda_block + no_lambda_block)


def fgsm(
  y: torch.tensor,
  X: torch.tensor, 
  Z_tilde: torch.tensor, 
  device: torch.device, 
  magnitude: Optional[float]=1, 
  order: Optional[float]=2, 
  a: Optional[float]=0.09, 
  b: Optional[float]=0.5, 
  c: Optional[float]=0.47
):
  '''
  Evaluate fast gradient sign method on a two-layer polynomial activation network.

  :param y: 1-D tensor containing the labels of the data 
  :param X: 2-D tensor containing the datapoints on which to evaluate FGSM
  :param Z_tilde: Weights of the polynomial activation network. Note that
    we take Z_tilde = (Z - Z').
  :param magnitude: Magnitude of the attack vector, defaults to 1.
  :param order: order of the norm of the attack vector, defaults to the L2 norm.
  :param a: First coefficient of the polynomial activation
  :param b: Second coefficient of the polynomial activation
  :param c: Third coefficient of the polynomial activation

  :return: 2-D tensor (same shape as :X:) containing the FGSM attack vectors.
  '''
  gradient = 2*a*(X @ Z_tilde[:-1,:-1]) + b * Z_tilde[:-1,-1] * torch.ones(X.shape[0],1).to(device)
  gradient_sign = gradient.sign()
  attack = magnitude * gradient_sign / gradient_sign.norm(p=order,dim=1, keepdim=True)
  return -torch.multiply(y.unsqueeze(1),attack)


class FeatureDataset(Dataset):
    '''
    Dataset class for the truncated outputs of an image model.
    '''
    def __init__(self, X, y, num_classes):
        self.X = X
        self.y = y
        self.y_ova = torch.zeros(X.shape[0],0)

        # generate 2-d tensor (N x C) of one vs. all labels. y_ova[y[i]] = 1, else -1
        for i in range(num_classes):
          cur_labels = torch.where(y == i, 1, -1).unsqueeze(1)
          self.y_ova = torch.cat((self.y_ova, cur_labels), dim=1)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        # keep track of original index. Refer to torch_solvers/robust_polyact_solver_batch.py
        # to see that this is necessary if we want to perform batch-wise updates 
        # on the w's and lbda's. 
        return idx, self.X[idx], self.y[idx], self.y_ova[idx]