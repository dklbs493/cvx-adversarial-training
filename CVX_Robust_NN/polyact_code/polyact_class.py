import cvxpy as cp
import numpy as np
import torch
import polyact_helpers as ph
import solve_conv as sc
import solve_reg as sr

class ConvPolyActNN:
    # Inputs:
    # x - training data
    # y - training labels (classes labeled with integers 0, 1, 2, etc.)
    # filter_size - side length of square filter
    # stride - vertical and horizontal stride
    # which_class - chose integer corresponding to which class label is the "one" in 
    #       one-vs-all classification
    # eps - robustness parameter; gives radius of ball around training data neural network 
    #       gives correct output
    # reg - coefficient of regularization. 
    # indices - specifies which training data will have robustness constraints applied
    # cone - True means we have quadratic margin curve, false means we have a 
    #       constant margin for robustly classified points
    # always_feas - True means the objective is penalized for falling below the margin curves,
    #       False means hard constraints are set to keep the output above the margin curves

    def __init__(self,x,y,filter_size, stride, which_class, eps, reg, indices=None, margin_curve=False, always_feas=False):
        self.params = {}
        self.params["which_class"], self.params["filter_size"] = which_class, filter_size
        self.params["stride"], self.params["reg"], self.params["eps"] = stride, reg, eps
        self.params["patch_matrices"] =  ph.make_patchifiers(x[0], filter_size, stride)
        self.return_args = sc.solve_conv(self.params, x,y,indices=indices, margin_curve=margin_curve, always_feas=always_feas)
        self.Z_dict, self.val, self.lip = self.return_args["variables"], self.return_args["value"], self.return_args["lipschitz"]
        self.Q, self.g, self.h = self.form_mats(x.shape[1] * x.shape[2])
        self.forward_weight = np.vstack((self.Q.reshape(((x.shape[1] * x.shape[2])**2),1), self.g, self.h))
        self.num_filters = self.calc_num_filters()

    def form_mats(self, d):
        Q = np.zeros((d,d))
        g = np.zeros((d,1))
        h = np.zeros((1,1))
        patch_matrices = self.params["patch_matrices"]
        for k, I_k in enumerate(patch_matrices):
            Q += I_k.T @ (self.Z_dict[(1,0,k)].value-self.Z_dict[(1,1,k)].value) @ I_k
            g += I_k.T @ (self.Z_dict[(2,0,k)].value-self.Z_dict[(2,1,k)].value)
            h += (self.Z_dict[(4,0,k)].value-self.Z_dict[(4,1,k)].value)
        return Q, g, h

    def conv_forward(self,x):
        d = (x.shape[1]*x.shape[2])
        N = x.shape[0]
        scaled_lifted_x = ph.scale_x(ph.lift_x(x.reshape(x.shape[0], d), d), None, 0.09, 0.5, 0.47,d)
        all_scores = self.forward_weight.T @ scaled_lifted_x.T
        return all_scores

    def accuracy(self, x, y):
        out = self.conv_forward(x)
        return (np.sign(out).squeeze() == y).sum() / y.shape[0]
    
    def calc_num_filters(self):
        num_filters = 0
        for k in range(len(self.params["patch_matrices"])):
            num_filters += np.linalg.matrix_rank(self.Z_dict[(0,0,k)], tol=1e-3)
            num_filters += np.linalg.matrix_rank(self.Z_dict[(0,1,k)], tol=1e-3)
        return num_filters

    def flip_distance_conv(self, x, label):
        Z_dict = self.return_args["variables"]
        a,b,c = 0.09, 0.5, 0.47

        Is = self.params["patch_matrices"]
        A1 = self.Q #np.zeros((x.shape[0] * x.shape[1], x.shape[0] * x.shape[1]))
        b1 = self.g #np.zeros((x.shape[0] * x.shape[1],1))
        c1 = self.h #np.zeros((1,1))

        # for k in range(len(Is)):
        #     A1  += Is[k].T @ (Z_dict[(1, 0, k)] - Z_dict[(1, 1, k)]).value @ Is[k]
        #     b1 += Is[k].T @ (Z_dict[(2, 0, k)] - Z_dict[(2, 1, k)]).value
        #     c1 += (Z_dict[(4, 0, k)] - Z_dict[(4, 1, k)]).value

        d = x.shape[1]*x.shape[1]

        gamma, X = cp.Variable((d,1)), cp.Variable((d, d), symmetric=True)
        flat_x = x.reshape((1,d))
        A0, b0, c0 = np.eye(d), -flat_x.T, flat_x @ flat_x.T

        A1 *= (label * a)
        b1 *= (label * b)
        c1 *= (label * c)

        block2 = cp.bmat([[A1, b1],[b1.T, c1]])
        constraints = [cp.trace(A1 @ X) + 2*b1.T @ gamma + c1 <= 0]
        constraints += [cp.bmat([[X, gamma],[gamma.T, np.ones((1,1))]]) >> 0]
        obj = cp.Minimize(cp.trace(A0 @ X) + 2 * b0.T @ gamma + c0)
        prob = cp.Problem(obj,constraints)
        dist = prob.solve(solver=cp.SCS)
        return np.sqrt(dist), gamma.value

class PolyActNN:
    # Inputs:
    # x - training data
    # y - training labels (classes labeled with integers 0, 1, 2, etc.)
    # which_class - chose integer corresponding to which class label is the "one" in 
    #       one-vs-all classification
    # eps - robustness parameter; gives radius of ball around training data neural network 
    #       gives correct output
    # reg - coefficient of regularization. 
    # indices - specifies which training data will have robustness constraints applied
    # cone - True means we have quadratic margin curve, false means we have a 
    #       constant margin for robustly classified points
    # always_feas - True means the objective is penalized for falling below the margin curves,
    #       False means hard constraints are set to keep the output above the margin curves
    
    def __init__(self,x,y,eps,reg_coeff,indices=None,marg=1, which_class = None,always_feas=False,margin_curve=None, robust=True):
        if robust:
          self.return_args = sr.solve_ova_always_feas(x,y,which_class,reg_coeff,eps)
        else:
          self.return_args = sr.solve_ova_nonrobust(x,y,which_class,reg_coeff)
        if self.return_args:
          self.weights, self.val = self.return_args["variables"], self.return_args["value"]
          self.num_classes = 1
          self.hidden_size = self.calc_rank()
          self.first_layer_weights = np.hstack(tuple(self.return_args["first_layer_weights"]))
          #self.second_layer_weights = np.hstack(tuple(self.return_args["second_layer_weights"]))
          self.eigs = self.max_eigs()
      
    def forward(self, x):
        n = x.shape[0]
        Z1 = self.weights[(0,1,0)].value - self.weights[(0,1,1)].value  
        Z2 = self.weights[(0,2,0)].value - self.weights[(0,2,1)].value
        Z4 = self.weights[(0,4,0)].value - self.weights[(0,4,1)].value
        out = 0.09 * np.diag(x @ Z1 @ x.T).reshape(n) + 0.5 * (x @ Z2).reshape(n) + 0.47 * Z4
        return out
    
    def calc_rank(self):
        rank = 0
        for k in range(self.num_classes):
          rank += np.linalg.matrix_rank(self.weights[(k,0,0)].value, tol=1e-5)
          rank += np.linalg.matrix_rank(self.weights[(k,0,1)].value, tol=1e-5)
        return rank

    def accuracy(self, x, y):
        pred = self.forward(x)
        return (np.sign(pred).squeeze() == y).sum() / x.shape[0]

    def max_eigs(self):
        max_eigs = np.zeros(self.num_classes)
        for k in range(self.num_classes):
            scale_z = self.weights[(k,0,0)].value - self.weights[(k,0,1)].value
            scale_z[:-1,:-1] *= 0.09
            scale_z[-1,:] *= 0.25
            scale_z[:,-1] *= 0.25
            scale_z[-1,-1] *= 0.47
            max_eigs[k] = np.abs(np.linalg.eig(scale_z)[0]).max()
        return max_eigs

    def truncated_forward(self,x):
        Xu = (x @ self.first_layer_weights)
        return 0.09 * (Xu**2) + 0.5 * Xu + 0.47 

    def flip_distance(self, x, label):
        a,b,c = 0.09, 0.5, 0.47
        d = x.shape[1]
        gamma, X = cp.Variable((d,1)), cp.Variable((d, d), symmetric=True)
        Z_dict = self.return_args["variables"]
        A0, b0, c0 = np.eye(x.shape[1]), -x.T, x @ x.T
     
        A1 = label * a*(Z_dict[(0,1,0)]-Z_dict[(0,1,1)]).value
        b1 = label * (b/2)*(Z_dict[(0,2,0)]-Z_dict[(0,2,1)]).value
        c1 = label * c*(Z_dict[(0,4,0)]-Z_dict[(0,4,1)]).value

        block2 = cp.bmat([[A1, b1],[b1.T, c1]])
        constraints = [cp.trace(A1 @ X) + 2*b1.T @ gamma + c1 <= 0]
        constraints += [cp.bmat([[X, gamma],[gamma.T, np.ones((1,1))]]) >> 0]
        obj = cp.Minimize(cp.trace(A0 @ X) + 2 * b0.T @ gamma + c0)
        prob = cp.Problem(obj,constraints)
        dist = prob.solve(solver=cp.SCS)
        return np.sqrt(dist), gamma.value