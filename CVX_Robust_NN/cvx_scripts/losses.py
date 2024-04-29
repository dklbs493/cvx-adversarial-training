import torch

# ORIGINAL HINGE LOSS FUNCTION
def loss_func_cvxproblem(yhat, y, model, _x, sign_patterns, beta, rho, device):
    _x = _x.view(_x.shape[0], -1)

    # term 1
    loss = 0.5 * torch.norm(yhat - y)**2
    # term 2
    loss = loss + beta * torch.sum(torch.norm(model.v, dim=1))
    loss = loss + beta * torch.sum(torch.norm(model.w, dim=1))

    # term 3
    sign_patterns = sign_patterns.unsqueeze(2) # N x P x 1

    # penalize constraint violations on the v's and w's with loss terms for each:
    Xv = torch.matmul(_x, torch.sum(model.v, dim=2, keepdim=True)) # N x d times P x d x 1 -> P x N x 1
    DXv = torch.mul(sign_patterns, Xv.permute(1, 0, 2)) # P x N x 1
    relu_term_v = torch.max(-2*DXv + Xv.permute(1, 0, 2), torch.Tensor([0]).to(device))
    loss = loss + rho * torch.sum(relu_term_v)

    Xw = torch.matmul(_x, torch.sum(model.w, dim=2, keepdim=True))
    DXw = torch.mul(sign_patterns, Xw.permute(1, 0, 2))
    relu_term_w = torch.max(-2*DXw + Xw.permute(1, 0, 2), torch.Tensor([0]).to(device))
    loss = loss + rho * torch.sum(relu_term_w)

    return loss

def new_robust_loss_func_cvxproblem(yhat, y, model, _x, sign_patterns, beta, rho, eps, device, norm=2):
    _x = _x.view(_x.shape[0], -1)

    # term 1
    # sign_patterns are  P x N x 1
    # y_hat is C x 1
    # model.v is P x d x C
    # regularization term
    loss = 0
    norm_term = beta * torch.sum(torch.norm(model.v, dim=1) + torch.norm(model.w, dim=1))
    loss += norm_term

    # ========================== ROBUST LOSS ========================= #
    v_w = model.v - model.w # P x d x C

    # apply sign patterns to v-w
    temp1 = torch.matmul(v_w.permute(1, 2, 0), sign_patterns.permute(1, 0).float()) # d x C x P times P x N gives d x C x N
    temp1 = temp1.permute(2, 0, 1)

    # get class differences and and multiply with x
    indices = y.argmax(dim=1)
    Alpha = temp1[torch.arange(_x.shape[0]),:,indices].unsqueeze(2) - temp1 # N x d x 1 - N x d x C = N x d x C
    _xAlpha = torch.sum(torch.mul(_x.unsqueeze(2), Alpha), dim=1) - eps * torch.norm(Alpha, p = norm, dim=1) # N x C

    #Alpha = torch.matmul(sign_patterns, v_w) # N x P x 1 times P x d x C gives N x d x C
    temp2 = torch.min(_xAlpha, torch.tensor([1]).to(device)) # N x C or min?
    robust_loss = torch.sum(temp2, dim=1) # N x 1
    loss -= torch.mean(robust_loss)

    # ========================ROBUST CONSTRAINTS====================== #
    sign_patterns = sign_patterns.unsqueeze(2) # N x P x 1
    #penalize constraint violations on the v's and w's with loss terms for each:
    Xv = torch.matmul(_x, torch.sum(model.v, dim=2, keepdim=True)) # N x d times P x d x 1 -> P x N x 1
    DXv = torch.mul(sign_patterns, Xv.permute(1, 0, 2)) # P x N x 1
    relu_term_v = torch.max(-2*DXv + Xv.permute(1, 0, 2), torch.Tensor([0]).to(device))
    loss = loss + rho * torch.sum(relu_term_v)

    Xw = torch.matmul(_x, torch.sum(model.w, dim=2, keepdim=True))
    DXw = torch.mul(sign_patterns, Xw.permute(1, 0, 2))
    relu_term_w = torch.max(-2*DXw + Xw.permute(1, 0, 2) + eps*torch.norm(model.w, p = norm, dim=(1,2)).T,torch.Tensor([0]).to(device))
    loss = loss + rho * torch.sum(relu_term_w)

    return loss, torch.mean(rho*(relu_term_v + relu_term_w)), -torch.mean(robust_loss), norm_term

def robust_loss_func_cvxproblem(yhat, y, model, _x, sign_patterns, beta, rho, eps, device, norm=2):
    _x = _x.view(_x.shape[0], -1)

    # term 1
    # sign_patterns are  P x N x 1
    # y_hat is C x 1
    # model.v is P x d x C
    # regularization term
    loss = 0
    norm_term = beta * torch.sum(torch.norm(model.v, dim=1) + torch.norm(model.w, dim=1))
    loss += norm_term

    # ========================== ROBUST LOSS ========================= #
    v_w = model.v - model.w # P x d x C

    # apply sign patterns to v-w
    temp1 = torch.matmul(v_w.permute(1, 2, 0), sign_patterns.permute(1, 0).float()) # d x C x P times P x N gives d x C x N
    temp1 = temp1.permute(2, 0, 1)

    # get class differences and and multiply with x
    indices = y.argmax(dim=1)
    Alpha = temp1[torch.arange(_x.shape[0]),:,indices].unsqueeze(2) - temp1 # N x d x 1 - N x d x C = N x d x C
    _xAlpha = torch.sum(torch.mul(_x.unsqueeze(2), Alpha), dim=1) - eps * torch.norm(Alpha, p = norm, dim=1) # N x C

    #Alpha = torch.matmul(sign_patterns, v_w) # N x P x 1 times P x d x C gives N x d x C
    temp2 = torch.min(_xAlpha, torch.tensor([1]).to(device)) # N x C or min?
    robust_loss = torch.sum(temp2, dim=1) # N x 1
    loss -= torch.mean(robust_loss)

    # ========================ROBUST CONSTRAINTS====================== #
    sign_patterns = (sign_patterns.T).unsqueeze(2)

    Xv = torch.matmul(_x, model.v.permute(0,2,1).unsqueeze(3)).squeeze(3) # P x C x N 
    DXv = torch.mul(sign_patterns, Xv.permute(0,2,1)) # P X N X C

     # N x P x 1 (SHOULD BE P X N X 1)
    #penalize constraint violations on the v's and w's with loss terms for each:
    relu_term_v = torch.max(-2*DXv + Xv.permute(0, 2, 1) + eps * torch.norm(model.v,p=norm,dim=1, keepdim=True), torch.Tensor([0]).to(device))
    loss = loss + rho * torch.sum(relu_term_v)

    Xw = torch.matmul(_x, model.w.permute(0,2,1).unsqueeze(3)).squeeze(3) # P x C x N 
    DXw = torch.mul(sign_patterns, Xw.permute(0,2,1)) # P X N X C
    relu_term_w = torch.max(-2*DXw + Xw.permute(0, 2, 1) + eps * torch.norm(model.w, p=norm, dim=1,keepdim=True), torch.Tensor([0]).to(device))
    loss = loss + rho * torch.sum(relu_term_w)

    return loss, torch.mean(rho*(relu_term_v + relu_term_w)), -torch.mean(robust_loss), norm_term

def get_nonconvex_cost(y, model, _x, beta, device):
    _x = _x.view(_x.shape[0], -1)
    Xv = torch.matmul(_x, model.v)
    Xw = torch.matmul(_x, model.w)
    Xv_relu = torch.max(Xv, torch.Tensor([0]).to(device))
    Xw_relu = torch.max(Xw, torch.Tensor([0]).to(device))

    prediction_w_relu = torch.sum(Xv_relu - Xw_relu, dim=0, keepdim=False)
    prediction_cost = 0.5 * torch.norm(prediction_w_relu - y)**2

    regularization_cost = beta * (torch.sum(torch.norm(model.v, dim=1)**2) + torch.sum(torch.norm(model.w, p=1, dim=1)**2))

    return prediction_cost + regularization_cost