from typing import Optional, Union, List, Tuple

import torch
from torch.nn import MultiMarginLoss
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

import numpy as np
from attacks.fgsm import eval_fgsm
from torch.types import Device
from torch.utils.data import DataLoader
from copy import deepcopy

from utils import (
  accuracy,
  forward,
  fgsm,
  trace_penalty,
  sym, 
  nearest_PSD,
  batch_nearest_psd,
  batch_sym,
)

from models.one_vs_all import MultiClassPolyAct
from models.spliced import Spliced_PolyAct

margin_loss = MultiMarginLoss()
batch_trace_penalty = torch.vmap(trace_penalty)

def batch_adversarial_polyact_train(
  model: MultiClassPolyAct, 
  train_loader: DataLoader, 
  val_loader: DataLoader,
  r: float,
  beta: float, 
  device: torch.device, 
  lr: Optional[float]=0.01, 
  epochs: Optional[int]=5,
  rho: Optional[float]=1,
  batch_size=100,
  verbose: Optional[bool]=True,
  base_model=None,
  robust_eval_loader=None
):
  '''
  Train a polynomial activation network for multiclass classification with multimargin loss

  :param model: the polynomial activation network to be trained
  :param train_loader: torch dataloader for training data. Torch Dataset must have a special 
                       __getitem__ method. Look at FeatureDataset in utils.py for details.
  :param val_loader: torch dataloader for the validation set; subject to the same  __getitem__
                     requirement as above.
  :param r: robust radius parameter
  :param beta: regularization strength
  :param device: the device on which training should take place (should be 'cuda:0')
  :param lr: learning rate for optimizer (we use Adam by default, but this could be replaced by another method)
  :param epochs: number of epochs for training 
  :param rho: weight of the robust constraints in the objective
  :param verbose: If true, information is printed at each epoch of training
  '''

  if base_model is not None:
    for p in base_model.parameters():
      p.requries_grad = False
    best_model = None
    best_robust_acc = 0
  C = model.Z.shape[0]
  
  w = torch.zeros(C, len(train_loader)*batch_size)
  lbda = torch.zeros(C, len(train_loader)*batch_size)

  optimizer = torch.optim.Adam(params=[model.Z, model.Zp], lr=lr, weight_decay=0)
  train_accs = []
  val_accs = []
  train_losses = []
  val_margin_losses = []
  for j in range(epochs):
    batch_loss = 0
    count = 0
    train_correct = 0
    # print((w > 0).any())
    # print(lbda)
    for (indices, X, y, y_ova) in tqdm(train_loader, desc = 'Epoch '+str(j+1)):
      count += X.shape[0]
      cur_w = w[:,indices].to(device)
      cur_w.requires_grad = True
      cur_lbda = lbda[:,indices].to(device)
      cur_lbda.requires_grad = True

      # add current w and lbda parameters to the optimizer
      optimizer.param_groups[0]['params'].append(cur_w)
      optimizer.param_groups[0]['params'].append(cur_lbda)

      X = X.to(device)
      y = y.to(device)
      y_ova = y_ova.to(device) # these are one-vs-all labels, dimensions are C x 

      optimizer.zero_grad()

      scores = model.forward(X)

      with torch.no_grad():
        train_correct += (y == scores.T.argmax(dim=1)).float().sum().item()

      # --------------- PENALIZE CONSTRAINT VIOLATIONS --------------- # 
      # get the symmetric part of the model weights Z and Zp
      Z_sym = 0.5*(model.Z + model.Z.permute(0,2,1))
      Zp_sym = 0.5*(model.Zp + model.Zp.permute(0,2,1))

      # generate the matrices in robust semidefinite constraints of equation (6) 
      M_mats = model.generate_robust_constraints(X, y_ova, cur_lbda, cur_w, r, scores, Z_sym-Zp_sym)

      # penalize eigenvalues of PSD matrices if they are nonpositive
      robust_eigvals = torch.linalg.eigvalsh(M_mats)
      Z_eigvals = torch.linalg.eigvalsh(Z_sym)
      Zp_eigvals = torch.linalg.eigvalsh(Zp_sym)
      eigval_penalty = rho*(-robust_eigvals).clip(0).sum() + ((-Z_eigvals).clip(0) + (-Zp_eigvals).clip(0)).sum()

      # penalize nonpositivity of lambda
      lbda_positive_penalty = (-cur_lbda).clip(0).sum()

      # penality for trace constraint violations
      trace_penalties = (batch_trace_penalty(model.Z) + batch_trace_penalty(model.Zp)).mean()

      penalty_term = eigval_penalty + trace_penalties + lbda_positive_penalty
      # --------------------------------------------------------------- #

      regularization_term = beta*(model.Z[:,-1,-1] + model.Zp[:,-1,-1]).sum()

      # Push the worst-case scores to be positive with a hinge loss.
      # Important: observe that we don't interact with the model's output directly 
      # in this loss function -- only the worst-case score w!
      loss_term = (1-cur_w).clip(min=0).mean()  
      
      obj = loss_term + regularization_term + penalty_term
      batch_loss += obj.item()
      obj.backward(retain_graph=True)
      # print(cur_w.grad)
      optimizer.step()

      # store the sections of w and lbda values that just took a gradient step
      w[:,indices] = cur_w.detach().cpu()
      lbda[:,indices] = cur_lbda.detach().cpu()

      # clear up memory 
      del M_mats
      del X
      del y
      del y_ova

      # remove the current w and lbda parameters from the optimizer
      optimizer.param_groups[0]['params'] = optimizer.param_groups[0]['params'][:-2]
      del cur_w
      del cur_lbda

    train_losses.append(batch_loss / count)
    train_accs.append(train_correct / count)

    # evaluate
    with torch.no_grad():
      val_correct = 0
      count = 0

      for (indices, X, y, y_ova) in val_loader:
        count += X.shape[0]

        X = X.to(device)
        y = y.to(device)
        y_ova = y_ova.to(device)

        train_scores = model.forward(X).T
        val_scores = model.forward(X).T
        val_margin_losses.append(margin_loss(val_scores, y))
        val_correct += (y == val_scores.argmax(dim=1)).float().sum().item()
      
      val_accs.append(val_correct / count)

    # compute robust accuracy if a base model and robust evaluation loader are specified (could be the same as the validaiton loader)
    if base_model is not None and robust_eval_loader is not None:
      spliced = Spliced_PolyAct(base_model, model)
      robust_acc = eval_fgsm(spliced, device, robust_eval_loader, 1/255, torch.tensor([0.4914, 0.4822, 0.4465]).to(device), torch.tensor([0.2023, 0.1994, 0.2010]).to(device), is_polyact=True)
    
      print('\nRobust Accuracy:', robust_acc)
      if robust_acc > best_robust_acc:
        best_model = deepcopy(model)
        best_robust_acc = robust_acc
    
    if verbose:
        print('Loss:', round(train_losses[-1], 4))
        print('Train accuracy:', train_accs[-1], 'Validation accuracy:', val_accs[-1])
        print('Validation hinge loss:', round(val_margin_losses[-1].item(), 4))

  del lbda
  del w
  torch.cuda.empty_cache()
  return best_model, train_losses, train_accs, val_accs, best_robust_acc
