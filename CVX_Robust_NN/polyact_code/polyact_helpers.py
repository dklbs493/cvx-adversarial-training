import cvxpy as cp
import numpy as np
import torch

def lift_x(A, d, verbose=False):
    # Lift data with dimension n x d to n x d^2 + d + 1 dimensions.
    n = A.shape[0]

    X_V = np.zeros((n, d**2+d+1))
    for i in range(n):
        if i % 100 == 0 and verbose:
            print(i, end=", ")
        
        x_i = A[i:i+1,:].T
        X_V[i, 0:d**2] = np.matmul(x_i, x_i.T).reshape((d**2))
        X_V[i, d**2:d**2+d] = x_i.reshape((d))
        X_V[i, d**2+d] = 1
    return X_V

def scale_x(X_K, y, a, b, c, ff): 
    # scale, here, refers to multiplying the columns by a,b,c

    X_K_scaled = X_K.copy()
    X_K_scaled[:, 0:ff**2] = a * X_K_scaled[:, 0:ff**2]
    X_K_scaled[:, ff**2:ff**2+ff] = b * X_K_scaled[:, ff**2:ff**2+ff]
    X_K_scaled[:, ff**2+ff] = c
    return X_K_scaled

def patchify_data(data, filter_size, stride):
    # Take in image data of shape N x D X D and create patched data where original images are 
    # split into smaller images corresponding to each unique filter location. 

    width, height = data.shape[1], data.shape[2]
    num_patches = ((width - filter_size) // stride + 1)**2
    new_data = np.zeros((data.shape[0], num_patches, filter_size ** 2))
    for j in range(data.shape[0]):
        count = 0
        for i in range(0, data.shape[1] - filter_size + 1, stride):
            for k in range(0, data.shape[2] - filter_size + 1, stride):
                cur_patch = data[j,i:i+filter_size,k:k+filter_size]
                new_data[j,count,:] = cur_patch.reshape(filter_size**2)
                count += 1
    return new_data

def neural_decomposition(Z_decomp, tolerance=10**(-6)):
    #IMPLEMENTATION OF NEURAL DECOMPOSITION FROM BHARTAN & PILANCI
    # paper link: https://arxiv.org/pdf/2101.02429.pdf

    # decomposes Z_decomp as a sum of r (where r=rank(Z_decomp)) rank-1 matrices \sum_{j=1}^r y_jy_j^T where
    # y_j^TGy_j = 0 for all j=1,...,r
    # based on the alg given in the proof of lemma 2.4 of the paper 'A Survey of the S-Lemma'
    G = np.identity(Z_decomp.shape[0])
    G[-1,-1] = -1

    # step 0
    evals, evecs = np.linalg.eigh(Z_decomp)
    # some eigvals are negative due to numerical issues, tolerance masking deals with that
    ind_pos_evals = (evals > tolerance)
    p_i_all = evecs[:,ind_pos_evals] * np.sqrt(evals[ind_pos_evals])

    outputs_y = np.zeros(p_i_all.shape)

    for i in range(outputs_y.shape[1]-1):
        # step 1
        p_1 = p_i_all[:,0:1]
        p_1Gp_1 = np.matmul(p_1.T, np.matmul(G, p_1))

        if p_1Gp_1 == 0:
            y = p_1.copy()

            # update
            p_i_all = np.delete(p_i_all, 0, 1) # delete the first column
        else:
            for j in range(1, p_i_all.shape[1]):
                p_j = p_i_all[:,j:j+1]
                p_jGp_j = np.matmul(p_j.T, np.matmul(G, p_j))
                if p_1Gp_1 * p_jGp_j < 0:
                    break

            # step 2
            p_1Gp_j = np.matmul(p_1.T, np.matmul(G, p_j))
            discriminant = 4*p_1Gp_j**2 - 4*p_1Gp_1*p_jGp_j
            alpha = (-2*p_1Gp_j + np.sqrt(discriminant)) / (2*p_jGp_j)
            y = (p_1 + alpha*p_j) / np.sqrt(1+alpha**2)

            # update
            p_i_all = np.delete(p_i_all, j, 1) # delete the jth column
            p_i_all = np.delete(p_i_all, 0, 1) # delete the first column

            u = (p_j - alpha*p_1) / np.sqrt(1+alpha**2)
            p_i_all = np.concatenate((p_i_all, u), axis=1) # insert u to the list of p_i's

        # save y
        outputs_y[:,i:i+1] = y.copy()

    # save the remaining column
    outputs_y[:, -1:] = p_i_all.copy()
    return outputs_y

def gen_optimization_vars(C,n,d,robust=True):
    # generate the optimization variables for the fully connected polynomial
    # activation network training problem 

    Z_dict = {}
    for i in range(C):
        # first entry: class number
        # second entry: 0 means big Z; 1,2,4 means Z1...Z4
        # third entry: 0 / 1 denotes not prime / prime
        Z_dict[(i,1,0)], Z_dict[(i,1,1)] = cp.Variable((d,d), symmetric=True), cp.Variable((d,d),symmetric=True)
        Z_dict[(i,2,0)], Z_dict[(i,2,1)] = cp.Variable((d,1)), cp.Variable((d,1))
        Z_dict[(i,4,0)], Z_dict[(i,4,1)] = cp.Variable((1,1)), cp.Variable((1,1))

        temp_1 = cp.vstack((Z_dict[(i,1,0)], Z_dict[(i,2,0)].T))
        temp_2 = cp.vstack((Z_dict[(i,2,0)],Z_dict[(i,4,0)]))
        Z_dict[(i,0,0)] = cp.hstack((temp_1,temp_2))

        temp_1_prime = cp.vstack((Z_dict[(i,1,1)], Z_dict[(i,2,1)].T))
        temp_2_prime = cp.vstack((Z_dict[(i,2,1)],Z_dict[(i,4,1)]))
        Z_dict[(i,0,1)] = cp.hstack((temp_1_prime,temp_2_prime))

        Z_dict["zz" + str(i)] = cp.vstack((cp.reshape((Z_dict[(i,1,0)] - Z_dict[(i,1,1)]), (d**2,1)), (Z_dict[(i,2,0)]- Z_dict[(i,2,1)]), (Z_dict[(i,4,0)]- Z_dict[(i,4,1)])))

        Zbar_temp1 = cp.hstack((0.09 *(Z_dict[(i,1,0)] - Z_dict[(i,1,1)]), 0.25 * (Z_dict[(i,2,0)]- Z_dict[(i,2,1)])))
        Zbar_temp2 = cp.hstack((0.25 * (Z_dict[(i,2,0)]- Z_dict[(i,2,1)]).T, 0.47*(Z_dict[(i,4,0)]- Z_dict[(i,4,1)])))
        Z_dict["Zbar" + str(i)] = cp.vstack((Zbar_temp1, Zbar_temp2))

    Z_dict["zz"] = cp.hstack(tuple([Z_dict["zz" + str(i)] for i in range(C)]))
    if robust:
      Z_dict["delta"] = cp.Variable((n,C))
    return Z_dict

def gen_efficient_constraints(x,y,C,Z_dict):
    # generate problem constraints for the fully connected polynomial activation
    # training problem

    # x is lifted here, so f = d**2 + d + 1
    constraints = []
    n, f = x.shape

    for k in range(C):
        constraints += [Z_dict[(k,0,0)] >> 0, Z_dict[(k,0,1)] >> 0]
        constraints += [cp.trace(Z_dict[(k,1,0)]) == Z_dict[(k,4,0)]]
        constraints += [cp.trace(Z_dict[(k,1,1)]) == Z_dict[(k,4,1)]]

    return constraints

def gen_conv_vars(filter_size,num_patches):
    # generate optimization variables for the convolutional polynomial activation
    # network training problem.

    d = filter_size**2
    Z_dict = {}
    Z_dict["zz"] = []
    for k in range(num_patches):
        # first entry: class number
        # second entry: 0 means big Z; 1,2,4 means Z1...Z4
        # third entry: 0 / 1 denotes not prime / prime
        # fourth entry: patch number
        Z_dict[(1,0,k)], Z_dict[(1,1,k)] = cp.Variable((d,d), symmetric=True), cp.Variable((d,d),symmetric=True)
        Z_dict[(2,0,k)], Z_dict[(2,1,k)] = cp.Variable((d,1)), cp.Variable((d,1))
        Z_dict[(4,0,k)], Z_dict[(4,1,k)] = cp.Variable((1,1)), cp.Variable((1,1))

        temp_1 = cp.vstack((Z_dict[(1,0,k)], Z_dict[(2,0,k)].T))
        temp_2 = cp.vstack((Z_dict[(2,0,k)],Z_dict[(4,0,k)]))
        Z_dict[(0,0,k)] = cp.hstack((temp_1,temp_2))

        temp_1_prime = cp.vstack((Z_dict[(1,1,k)], Z_dict[(2,1,k)].T))
        temp_2_prime = cp.vstack((Z_dict[(2,1,k)],Z_dict[(4,1,k)]))
        Z_dict[(0,1,k)] = cp.hstack((temp_1_prime,temp_2_prime))

        Z_dict["zz" + str(k)] = cp.vstack((cp.reshape((Z_dict[(1,0,k)] - Z_dict[(1,1,k)]), (d**2,1)), (Z_dict[(2,0,k)]- Z_dict[(2,1,k)]), (Z_dict[(4,0,k)]- Z_dict[(4,1,k)])))
    Z_dict["zz"] = cp.hstack(tuple([Z_dict["zz" + str(k)] for k in range(num_patches)]))
    return Z_dict

def gen_I_k(flat_shape, row_len, row_idx, col_idx, f):
  # generate a matrix 
  
  I_k = np.zeros((f**2, flat_shape))
  out = np.zeros(f**2)
  flat_idxs = row_idx * row_len + col_idx
  I_k[np.arange(f**2), flat_idxs] = 1
  return I_k
  
def make_patchifiers(x, filter_size, stride):
  # for all unique patches, generate the corresponding patch matrix

  patchifiers = []

  width, height = x.shape[0], x.shape[1]
  x_flat_shape = width * height

  block = np.arange(filter_size)
  row_idxs = np.repeat(block, filter_size)
  col_idxs = np.tile(block, filter_size)

  for row in range(0, x.shape[0] - filter_size + 1, stride):
    for col in range(0, x.shape[1] - filter_size + 1, stride):
      cur_col_indices = col + col_idxs
      cur_row_indices = row + row_idxs
      cur_I = gen_I_k(width * height, x.shape[0], cur_row_indices, cur_col_indices, filter_size)
      patchifiers.append(cur_I)

  return patchifiers