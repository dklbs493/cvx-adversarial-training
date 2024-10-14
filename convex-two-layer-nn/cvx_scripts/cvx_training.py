import torch
from torch.autograd import Variable
from cvx_scripts.losses import *
from cvx_scripts.cvx_utils import *
from cvx_scripts.cvx_nn import custom_cvx_layer
import time
import numpy as np

def validation_cvxproblem(net, model, testloader, u_vectors, beta, rho, device, eps=None, robust=False, norm=2):
    test_loss = 0
    test_correct = 0
    test_noncvx_cost = 0
    t1_tot = 0
    t2_tot = 0
    n_tot = 0

    net.eval()
    with torch.no_grad():
        for ix, (_x, _y) in enumerate(testloader):
            _x = Variable(_x).to(device)
            _y = Variable(_y).to(device)

            _x = net.truncated_forward(_x)
            #_x = _x.view(_x.shape[0], -1)
            
            _z = (torch.matmul(_x, torch.from_numpy(u_vectors).float().to(device)) >= 0)

            output = model.forward(_x, _z)
            yhat = model(_x, _z).float()

            if robust:
              loss, t1, t2, n = robust_loss_func_cvxproblem(yhat, one_hot(_y, device).to(device), model, _x, _z, beta, rho, eps, device, norm=norm)
              t1_tot += t1.item()
              t2_tot += t2.item()
              n_tot += n.item()
            else:
              loss = loss_func_cvxproblem(yhat, one_hot(_y, device).to(device), model, _x, _z, beta, rho, device)

            test_loss += loss.item()
            test_correct += torch.eq(torch.argmax(yhat, dim=1), _y).float().sum()


            test_noncvx_cost += get_nonconvex_cost(one_hot(_y, device).to(device), model, _x, beta, device)
    if robust:
      print("constraint violation", t1_tot, "robust_loss",t2_tot, "norm of weights", n_tot)
    return test_loss, test_correct, test_noncvx_cost

def sgd_solver_cvxproblem(net, ds, ds_test, num_epochs, num_neurons, beta,
                       learning_rate, batch_size, rho, u_vectors,
                          solver_type, LBFGS_param, verbose=False,
                         n=60000, d=3072, num_classes=10, device='cpu', robust=False, eps=None, norm=2):
    device = torch.device(device)

    # create the model
    model = custom_cvx_layer(d, num_neurons, num_classes).to(device)
    net.eval()

    if solver_type == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    elif solver_type == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)#,
    elif solver_type == "adagrad":
        optimizer = torch.optim.Adagrad(model.parameters(), lr=learning_rate)#,
    elif solver_type == "adadelta":
        optimizer = torch.optim.Adadelta(model.parameters(), lr=learning_rate)#,
    elif solver_type == "LBFGS":
        optimizer = torch.optim.LBFGS(model.parameters(), history_size=LBFGS_param[0], max_iter=LBFGS_param[1])#,

    # arrays for saving the loss and accuracy
    losses = np.zeros((int(num_epochs*np.ceil(n / batch_size))))
    accs = np.zeros(losses.shape)
    noncvx_losses = np.zeros(losses.shape)

    losses_test = np.zeros((num_epochs+1))
    accs_test = np.zeros((num_epochs+1))
    noncvx_losses_test = np.zeros((num_epochs+1))

    times = np.zeros((losses.shape[0]+1))
    times[0] = time.time()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                           verbose=verbose,
                                                           factor=0.5,
                                                           eps=1e-12)

    model.eval()
    losses_test[0], accs_test[0], noncvx_losses_test[0] = validation_cvxproblem(net, model, ds_test, u_vectors, beta, rho, device, eps=eps, robust=robust, norm=norm) # loss on the entire test set

    iter_no = 0
    print('starting training')
    for i in range(num_epochs):
        model.train()
        for ix, (_x, _y, _z) in enumerate(ds):
            #=========make input differentiable=======================
            _x = Variable(_x).to(device)
            _y = Variable(_y).to(device)
            _z = Variable(_z).to(device)

            #========forward pass=====================================
            yhat = model(_x, _z).float()
            if robust:
              loss, _, _, _ = robust_loss_func_cvxproblem(yhat, one_hot(_y, device).to(device), model, _x,_z, beta, rho, eps, device, norm=norm)
              loss /= len(_y)
            else:
              loss = loss_func_cvxproblem(yhat, one_hot(_y, device).to(device), model, _x,_z, beta, rho, device)/len(_y)
            correct = torch.eq(torch.argmax(yhat, dim=1), _y).float().sum()/len(_y) # accuracy
            #=======backward pass=====================================
            optimizer.zero_grad() # zero the gradients on each pass before the update
            loss.backward() # backpropagate the loss through the model
            optimizer.step() # update the gradients w.r.t the loss

            losses[iter_no] = loss.item() # loss on the minibatch
            accs[iter_no] = correct
            noncvx_losses[iter_no] = get_nonconvex_cost(one_hot(_y, device).to(device), model, _x, beta, device)/len(_y)

            iter_no += 1
            times[iter_no] = time.time()

        model.eval()
        # get test loss and accuracy
        losses_test[i+1], accs_test[i+1], noncvx_losses_test[i+1] = validation_cvxproblem(net, model, ds_test, u_vectors, beta, rho, device, eps=eps, robust=robust, norm=norm) # loss on the entire test set

        if i % 1 == 0:
            print("Epoch [{}/{}], TRAIN: noncvx/cvx loss: {}, {} acc: {}. TEST: noncvx/cvx loss: {}, {} acc: {}".format(i, num_epochs,
                    np.round(noncvx_losses[iter_no-1], 3), np.round(losses[iter_no-1], 3), np.round(accs[iter_no-1], 3),
                    np.round(noncvx_losses_test[i+1], 3)/10000, np.round(losses_test[i+1], 3)/10000, np.round(accs_test[i+1]/10000, 3)))

        scheduler.step(losses[iter_no-1])

    return noncvx_losses, accs, noncvx_losses_test/10000, accs_test/10000, times, losses, losses_test/10000, model