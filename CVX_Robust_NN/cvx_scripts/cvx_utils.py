import torch
import numpy as np

def check_if_already_exists(element_list, element):
    # check if element exists in element_list
    # where element is a numpy array
    for i in range(len(element_list)):
        if np.array_equal(element_list[i], element):
            return True
    return False

def generate_sign_patterns(A, P, verbose=False):
    # generate sign patterns
    n, d = A.shape
    sign_pattern_list = []  # sign patterns
    u_vector_list = []             # random vectors used to generate the sign paterns
    umat = np.random.normal(0, 1, (d,P))
    sampled_sign_pattern_mat = (np.matmul(A, umat) >= 0)
    for i in range(P):
        sampled_sign_pattern = sampled_sign_pattern_mat[:,i]
        sign_pattern_list.append(sampled_sign_pattern)
        u_vector_list.append(umat[:,i])
    if verbose:
        print("Number of sign patterns generated: " + str(len(sign_pattern_list)))
    return len(sign_pattern_list),sign_pattern_list, u_vector_list

def one_hot(labels, device, num_classes=10):
    y = torch.eye(num_classes).to(device)
    return y[labels.long()]

def generate_conv_sign_patterns(A2, P, verbose=False):
    # generate convolutional sign patterns given image data A2
    # and desired number of patterns P
    n, c, p1, p2 = A2.shape
    A = A2.reshape(n,int(c*p1*p2))
    # filter size is 3x3xc, c= num_channels
    fsize=9*c

    # convert image data into 2d: N x width x height x channels i.e. (n x c x p1 x p2)
    d=c*p1*p2;
    fs=int(np.sqrt(9))
    unique_sign_pattern_list = []
    u_vector_list = []

    for i in range(P):
        # select a random location to place the filter by choosing ind1, ind2:
        ind1=np.random.randint(0,p1-fs+1)
        ind2=np.random.randint(0,p2-fs+1)

        # create a mask the size of our data where the only nonzero part
        # corresponds to the chosen filter location
        u1p= np.zeros((c,p1,p2))
        u1p[:,ind1:ind1+fs,ind2:ind2+fs]=np.random.normal(0, 1, (fsize,1)).reshape(c,fs,fs)
        u1=u1p.reshape(d,1)

        # sample the sign pattern and append it to the list (as long as we havent seen it before)
        sampled_sign_pattern = (np.matmul(A, u1) >= 0)[:,0]
        unique_sign_pattern_list.append(sampled_sign_pattern)
        u_vector_list.append(u1)
        if i % 250 == 0:
          print(i, "patterns generated...")

    if verbose:
        print("Number of unique sign patterns generated: " + str(len(unique_sign_pattern_list)))
    return len(unique_sign_pattern_list),unique_sign_pattern_list, u_vector_list
