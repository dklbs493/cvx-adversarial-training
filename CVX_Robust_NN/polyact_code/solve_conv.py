import cvxpy as cp
import numpy as np
import torch
import polyact_helpers as ph

def solve_conv(params, x, y, indices=None, margin_curve='constant', always_feas=False):

    a, b, c = .09, .5, .47
    filter_size, stride, which_class = params["filter_size"], params["stride"], params["which_class"]
    eps, reg = params["eps"], params["reg"]

    copy_y = y.copy()
    copy_y[y != which_class] = -1
    copy_y[y == which_class] = 1
    d = filter_size**2
    N = x.shape[0]
    patched_x = ph.patchify_data(x, filter_size, stride)
    scaled_lifted_patched_x = np.concatenate(tuple([ph.scale_x(ph.lift_x(patched_x[i,:,:],d), copy_y, 0.09, 0.5, 0.47, d)[np.newaxis,:] for i in range(N)]), axis=0)
    num_patches = patched_x.shape[1]
    flat_dim = x.shape[1] * x.shape[2]

    # ----------------- Generate the constraints --------------- #

    constraints = []
    Z_dict = ph.gen_conv_vars(params["filter_size"], num_patches)
    for k in range(num_patches):
      constraints += [Z_dict[(1,0,k)] >> 0, Z_dict[(1,1,k)] >> 0]
      constraints += [cp.trace(Z_dict[(1,0,k)]) == Z_dict[(4,0,k)]]
      constraints += [cp.trace(Z_dict[(1,1,k)]) == Z_dict[(4,1,k)]]

    # create matrices which grab patches corresponding to convolution filters
    Is = params["patch_matrices"]

    # generate the quadratic form
    diff1 = np.zeros((x.shape[1] * x.shape[2], x.shape[1] * x.shape[2]))
    diff2 = np.zeros((x.shape[1] * x.shape[2],1))
    diff4 = np.zeros((1,1))

    for k in range(num_patches):
      diff1  = diff1 + Is[k].T @ (Z_dict[(1, 0, k)] - Z_dict[(1, 1, k)]) @ Is[k]
      diff2 = diff2 + Is[k].T @ (Z_dict[(2, 0, k)] - Z_dict[(2, 1, k)])
      diff4 = diff4 + (Z_dict[(4, 0, k)] - Z_dict[(4, 1, k)])

    # Add robustness constraints if indices are provided
    n = N
    if indices is not None:
      Z_dict["lambda"] = cp.Variable(indices.shape)
      constraints += [Z_dict["lambda"] >= 0]

      I = np.eye(flat_dim)

      # set margin to optimization variable if we want to relax constraints
      if always_feas:
        marg = cp.Variable(indices.shape)
      else:
        cur_marg = 0.001

      n -= indices.shape[0]

      # create robust constraints with the convolutional quad form
      for i, idx in enumerate(indices):
        if always_feas:
          cur_marg = marg[i]

        curl = copy_y[idx]
        curx = x[i:i+1, :,:].reshape((1,flat_dim))

        F = a * diff1 * curl
        g = ((b * diff2) + (2*(a * diff1) @ curx.T)) * curl
        h = ((c * diff4) + (a * curx @ diff1 @ curx.T) + (b * diff2.T @ curx.T)) * curl

        marg_term = cur_marg

        if margin_curve == 'quadratic':
          F += cur_marg * I / (eps ** 2) # cur_marg * I
          marg_term = cur_marg #* (eps ** 2)

        lbda = Z_dict["lambda"][i]
        block = cp.vstack((cp.hstack((lbda * I + F, 0.5*g)), cp.hstack((0.5*g.T, lbda * (-(eps ** 2)) + h - marg_term))))
        constraints += [block >> 0]

    # ------------------ Generate the objective ---------------- #

    obj = 0
    classes = copy_y.astype(int)
    zz = Z_dict["zz"]

    # scaled_lifted_patched_x is N x K x (F^4 + F^2 + 1).
    # (F^4 + F^2 + 1) x K should be zz matrix. Then have N x K x K
    # zz is (F^4 + F^2 + 1) x K x C
    if n > 0:
      all_scores = np.zeros((N,1))
      for k in range(num_patches):
        all_scores = all_scores + scaled_lifted_patched_x[:,k,:] @ zz[:,k]
      errors = 1 - cp.multiply(all_scores, classes)
      obj += cp.sum(cp.pos(errors)) / n

    for k in range(num_patches):
      obj += reg * (Z_dict[(4,0,k)] + Z_dict[(4,1,k)])

    problem = cp.Problem(cp.Minimize(obj), constraints)
    soln = problem.solve(solver=cp.SCS)

    lip = np.vstack((np.hstack((0.5*b*diff2.T.value, c*diff4.value)), np.hstack((a * diff1.value, 0.5*b*diff2.value))))

    return_args = {}
    return_args["variables"] = Z_dict
    return_args["value"] = soln
    return_args["lipschitz"] = np.linalg.norm(lip, ord=2)
    if problem.status in ["infeasible", "unbounded"]:
          return print("Problem is " + problem.status)
    else:
      return return_args
