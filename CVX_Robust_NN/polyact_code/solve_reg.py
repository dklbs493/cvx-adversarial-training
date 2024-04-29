import cvxpy as cp
import numpy as np
import torch
import polyact_helpers as ph

def solve_ova(x,y,which_class,reg_coeff,eps,marg=1, indices=None, always_feas = False, margin_curve='constant'):
  a, b, c = 0.09, 0.5, 0.47
  n, d = x.shape

  y_input = y.copy()
  # change labels for OVA classification
  y_input[y_input != which_class] = -1
  y_input[y_input == which_class] = 1
  
  x_input = ph.scale_x(ph.lift_x(x, d), y, 0.09, 0.5, 0.47, d)
  

  Z_dict = ph.gen_optimization_vars(1,n,d,robust=False)
  constraints = ph.gen_efficient_constraints(x_input,y_input,1,Z_dict)

  # We have num_constrained * (num_classes - 1) semidefinite constraints, of size (d + 1) x (d + 1)
  if indices is not None:
      I = np.eye(d)
      cur_marg = marg
      if always_feas:
        marg = cp.Variable((indices.shape[0],1)) #cp.Variable((1,1))#
      Z_dict["lambda"] = cp.Variable((indices.shape[0],1)) #cp.Variable((1,1))#
      constraints += [Z_dict["lambda"] >= 0]
      masked_x = np.delete(np.arange(n), indices)
      increment_example = 0
      diff1 = (Z_dict[(0,1,0)] - Z_dict[(0,1,1)])
      diff2 = (Z_dict[(0,2,0)] - Z_dict[(0,2,1)])
      diff4 = (Z_dict[(0,4,0)] - Z_dict[(0,4,1)])

      for i in indices:
          if always_feas:
            cur_marg = marg[increment_example]
          curl = y_input[i]
          curx = x[i:i+1,:]

          F = (a * diff1) * curl
          
          g = ((b * diff2) + (2*(a * diff1) @ curx.T)) * curl
          h = ((c * diff4) + (a * curx @ diff1 @ curx.T) + (b * diff2.T @ curx.T)) * curl

          marg_term = cur_marg

          if margin_curve == 'quadratic':
            F += cur_marg * I / (eps ** 2)
            marg_term = cur_marg #$* (eps ** 2)
             # adds in scaled margin * (eps **2) and gets rid of the other cur_marg term

          lbda = Z_dict["lambda"][increment_example,0]
          
          block = cp.vstack((cp.hstack((lbda * I + F, 0.5*g)), cp.hstack((0.5*g.T, lbda * (-(eps ** 2)) + h - marg_term))))
          constraints += [block >> 0]

          increment_example += 1

      x_input, y_input = x_input[masked_x,:], y_input[masked_x]
      n -= indices.shape[0]  
      
  # vectorized hinge loss
  obj = 0
  if n > 0:
    classes = y_input.astype(int)
    all_scores = x_input  @ Z_dict["zz0"]
    errors = 1 - cp.multiply(cp.reshape(all_scores,(n,)), y_input)
    obj += cp.sum(cp.pos(errors)) / n 
  print(n)
  
  if always_feas:
    obj += cp.sum(cp.pos(1 - marg)) / indices.shape[0]

  # regularization
  obj += reg_coeff * (Z_dict[(0,4,0)] + Z_dict[(0,4,1)])
  #obj += reg_coeff*cp.norm(Z_dict["Zbar0"])

  problem = cp.Problem(cp.Minimize(obj), constraints)
  soln = problem.solve(solver=cp.SCS)
  print(marg.value)
  if always_feas:
    Z_dict["marg"] = marg
    
  return_args = {}
  return_args["variables"] = Z_dict
  return_args["value"] = soln
  if problem.status in ["infeasible", "unbounded"]:
        return print("Problem is " + problem.status)
  else:
      return_args["first_layer_weights"] = []
      return_args["second_layer_weights"] = []
      for k in range(1):
        Z, Z_prime = Z_dict[(k,0,0)], Z_dict[(k,0,1)]
        decomp = ph.neural_decomposition(Z.value)
        decomp_prime = ph.neural_decomposition(Z_prime.value)
        first_layer_weights = np.concatenate((decomp[:-1, :], decomp_prime[:-1, :]), axis=1)
        first_layer_weights = first_layer_weights / np.sqrt(np.sum(first_layer_weights**2, axis=0))
        signs_second_layer = np.sign(np.concatenate((decomp[-1:, :], decomp_prime[-1:, :]), axis=1).T)
        return_args["first_layer_weights"].append(first_layer_weights * signs_second_layer[:,0])
        return_args["second_layer_weights"].append(np.concatenate((decomp[-1:, :]**2, -decomp_prime[-1:, :]**2), axis=1).T)
      return return_args

def solve_ova_always_feas(x,y,which_class,reg_coeff,eps):
    a, b, c = 0.09, 0.5, 0.47
    n, d = x.shape

    y_input = y.copy()
    # change labels for OVA classification
    y_input[y_input != which_class] = -1
    y_input[y_input == which_class] = 1
    
    x_input = ph.scale_x(ph.lift_x(x, d), y, 0.09, 0.5, 0.47, d)
    Z_dict = ph.gen_optimization_vars(1,n,d,robust=False)
    constraints = ph.gen_efficient_constraints(x_input,y_input,1,Z_dict)

    # We have num_constrained * (num_classes - 1) semidefinite constraints, of size (d + 1) x (d + 1)
    I = np.eye(d)
        
    marg = cp.Variable((x.shape[0],1)) #cp.Variable((1,1))#
    Z_dict["lambda"] = cp.Variable((n,1)) #cp.Variable((1,1))#
    constraints += [Z_dict["lambda"] >= 0]
    increment_example = 0
    diff1 = (Z_dict[(0,1,0)] - Z_dict[(0,1,1)])
    diff2 = (Z_dict[(0,2,0)] - Z_dict[(0,2,1)])
    diff4 = (Z_dict[(0,4,0)] - Z_dict[(0,4,1)])

    for i in range(n):
        curl = y_input[i]
        curx = x[i:i+1,:]

        F = (a * diff1) * curl
        
        g = ((b * diff2) + (2*(a * diff1) @ curx.T)) * curl
        h = ((c * diff4) + (a * curx @ diff1 @ curx.T) + (b * diff2.T @ curx.T)) * curl

        lbda = Z_dict["lambda"][increment_example,0]
        
        block = cp.vstack((cp.hstack((lbda * I + F, 0.5*g)), cp.hstack((0.5*g.T, lbda * (-(eps ** 2)) + h - marg[i]))))
        constraints += [block >> 0]

        increment_example += 1

    #x_input, y_input = x_input[masked_x,:], y_input[masked_x]
      
  # vectorized hinge loss
    # obj = 0
    # classes = y_input.astype(int)
    # all_scores = x_input  @ Z_dict["zz0"]
    # errors = 1 - cp.multiply(cp.reshape(all_scores,(n,)), y_input)
    # obj += cp.sum(cp.pos(errors)) / n 
    
    obj = cp.sum(cp.pos(1 - marg)) / n

  # regularization
    obj += reg_coeff * (Z_dict[(0,4,0)] + Z_dict[(0,4,1)])

    problem = cp.Problem(cp.Minimize(obj), constraints)
    soln = problem.solve(solver=cp.SCS)
    Z_dict["marg"] = marg
    
    return_args = {}
    return_args["variables"] = Z_dict
    return_args["value"] = soln
    if problem.status in ["infeasible", "unbounded"]:
            return print("Problem is " + problem.status)
    else:
        return_args["first_layer_weights"] = []
        return_args["second_layer_weights"] = []
        for k in range(1):
            Z, Z_prime = Z_dict[(k,0,0)], Z_dict[(k,0,1)]
            decomp = ph.neural_decomposition(Z.value)
            decomp_prime = ph.neural_decomposition(Z_prime.value)
            first_layer_weights = np.concatenate((decomp[:-1, :], decomp_prime[:-1, :]), axis=1)
            first_layer_weights = first_layer_weights / np.sqrt(np.sum(first_layer_weights**2, axis=0))
            signs_second_layer = np.sign(np.concatenate((decomp[-1:, :], decomp_prime[-1:, :]), axis=1).T)
            return_args["first_layer_weights"].append(first_layer_weights * signs_second_layer[:,0])
            return_args["second_layer_weights"].append(np.concatenate((decomp[-1:, :]**2, -decomp_prime[-1:, :]**2), axis=1).T)
        return return_args

def solve_ova_nonrobust(x,y,which_class,reg_coeff):
    a, b, c = 0.09, 0.5, 0.47
    n, d = x.shape

    y_input = y.copy()
    # change labels for OVA classification
    y_input[y_input != which_class] = -1
    y_input[y_input == which_class] = 1
    
    x_input = ph.scale_x(ph.lift_x(x, d), y, 0.09, 0.5, 0.47, d)
    Z_dict = ph.gen_optimization_vars(1,n,d,robust=False)
    constraints = ph.gen_efficient_constraints(x_input,y_input,1,Z_dict)

    # We have num_constrained * (num_classes - 1) semidefinite constraints, of size (d + 1) x (d + 1)
    I = np.eye(d)
        
    marg = cp.Variable((x.shape[0],1)) #cp.Variable((1,1))#
    Z_dict["lambda"] = cp.Variable((n,1)) #cp.Variable((1,1))#
    constraints += [Z_dict["lambda"] >= 0]
    increment_example = 0
    diff1 = (Z_dict[(0,1,0)] - Z_dict[(0,1,1)])
    diff2 = (Z_dict[(0,2,0)] - Z_dict[(0,2,1)])
    diff4 = (Z_dict[(0,4,0)] - Z_dict[(0,4,1)])
      
  # vectorized hinge loss
    obj = 0
    classes = y_input.astype(int)
    all_scores = x_input  @ Z_dict["zz0"]
    errors = 1 - cp.multiply(cp.reshape(all_scores,(n,)), y_input)
    obj += cp.sum(cp.pos(errors)) / n 

  # regularization
    obj += reg_coeff * (Z_dict[(0,4,0)] + Z_dict[(0,4,1)])

    problem = cp.Problem(cp.Minimize(obj), constraints)
    soln = problem.solve(solver=cp.SCS)

    return_args = {}
    return_args["variables"] = Z_dict
    return_args["value"] = soln
    if problem.status in ["infeasible", "unbounded"]:
            return print("Problem is " + problem.status)
    else:
        return_args["first_layer_weights"] = []
        return_args["second_layer_weights"] = []
        for k in range(1):
            Z, Z_prime = Z_dict[(k,0,0)], Z_dict[(k,0,1)]
            decomp = ph.neural_decomposition(Z.value)
            decomp_prime = ph.neural_decomposition(Z_prime.value)
            first_layer_weights = np.concatenate((decomp[:-1, :], decomp_prime[:-1, :]), axis=1)
            first_layer_weights = first_layer_weights / np.sqrt(np.sum(first_layer_weights**2, axis=0))
            signs_second_layer = np.sign(np.concatenate((decomp[-1:, :], decomp_prime[-1:, :]), axis=1).T)
            return_args["first_layer_weights"].append(first_layer_weights * signs_second_layer[:,0])
            return_args["second_layer_weights"].append(np.concatenate((decomp[-1:, :]**2, -decomp_prime[-1:, :]**2), axis=1).T)
        return return_args