r"""
Classes <AbsWeighting, MoCo> are taken from https://github.com/median-research-group/LibMTL repository.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class AbsWeighting(nn.Module):
    r"""An abstract class for weighting strategies.
    """
    def __init__(self):
        super(AbsWeighting, self).__init__()
        
    def init_param(self):
        r"""Define and initialize some trainable parameters required by specific weighting methods. 
        """
        pass

    def _compute_grad_dim(self):
        self.grad_index = []
        for param in self.get_share_params():
            self.grad_index.append(param.data.numel())
        self.grad_dim = sum(self.grad_index)

    def _grad2vec(self):
        grad = torch.zeros(self.grad_dim)
        count = 0
        for param in self.get_share_params():
            if param.grad is not None:
                beg = 0 if count == 0 else sum(self.grad_index[:count])
                end = sum(self.grad_index[:(count+1)])
                grad[beg:end] = param.grad.data.view(-1)
            count += 1
        return grad

    def _compute_grad(self, losses, mode, rep_grad=False):
        '''
        mode: backward, autograd
        '''
        if not rep_grad:
            grads = torch.zeros(self.task_num, self.grad_dim).to(self.device)
            for tn in range(self.task_num):
                if mode == 'backward':
                    losses[tn].backward(retain_graph=True) if (tn+1)!=self.task_num else losses[tn].backward()
                    grads[tn] = self._grad2vec()
                elif mode == 'autograd':
                    grad = list(torch.autograd.grad(losses[tn], self.get_share_params(), retain_graph=True))
                    grads[tn] = torch.cat([g.view(-1) for g in grad])
                else:
                    raise ValueError('No support {} mode for gradient computation')
                self.zero_grad_share_params()
        else:
            if not isinstance(self.rep, dict):
                grads = torch.zeros(self.task_num, *self.rep.size()).to(self.device)
            else:
                grads = [torch.zeros(*self.rep[task].size()) for task in self.task_name]
            for tn, task in enumerate(self.task_name):
                if mode == 'backward':
                    losses[tn].backward(retain_graph=True) if (tn+1)!=self.task_num else losses[tn].backward()
                    grads[tn] = self.rep_tasks[task].grad.data.clone()
        return grads

    def _reset_grad(self, new_grads):
        count = 0
        for param in self.get_share_params():
            if param.grad is not None:
                beg = 0 if count == 0 else sum(self.grad_index[:count])
                end = sum(self.grad_index[:(count+1)])
                param.grad.data = new_grads[beg:end].contiguous().view(param.data.size()).data.clone()
            count += 1
            
    def _get_grads(self, losses, mode='backward'):
        r"""This function is used to return the gradients of representations or shared parameters.

        If ``rep_grad`` is ``True``, it returns a list with two elements. The first element is \
        the gradients of the representations with the size of [task_num, batch_size, rep_size]. \
        The second element is the resized gradients with size of [task_num, -1], which means \
        the gradient of each task is resized as a vector.

        If ``rep_grad`` is ``False``, it returns the gradients of the shared parameters with size \
        of [task_num, -1], which means the gradient of each task is resized as a vector.
        """
        if self.rep_grad:
            per_grads = self._compute_grad(losses, mode, rep_grad=True)
            if not isinstance(self.rep, dict):
                grads = per_grads.reshape(self.task_num, self.rep.size()[0], -1).sum(1)
            else:
                try:
                    grads = torch.stack(per_grads).sum(1).view(self.task_num, -1)
                except:
                    raise ValueError('The representation dimensions of different tasks must be consistent')
            return [per_grads, grads]
        else:
            self._compute_grad_dim()
            grads = self._compute_grad(losses, mode)
            return grads
        
    def _backward_new_grads(self, batch_weight, per_grads=None, grads=None):
        r"""This function is used to reset the gradients and make a backward.

        Args:
            batch_weight (torch.Tensor): A tensor with size of [task_num].
            per_grad (torch.Tensor): It is needed if ``rep_grad`` is True. The gradients of the representations.
            grads (torch.Tensor): It is needed if ``rep_grad`` is False. The gradients of the shared parameters. 
        """
        if self.rep_grad:
            if not isinstance(self.rep, dict):
                # transformed_grad = torch.einsum('i, i... -> ...', batch_weight, per_grads)
                transformed_grad = sum([batch_weight[i] * per_grads[i] for i in range(self.task_num)])
                self.rep.backward(transformed_grad)
            else:
                for tn, task in enumerate(self.task_name):
                    rg = True if (tn+1)!=self.task_num else False
                    self.rep[task].backward(batch_weight[tn]*per_grads[tn], retain_graph=rg)
        else:
            # new_grads = torch.einsum('i, i... -> ...', batch_weight, grads)
            new_grads = sum([batch_weight[i] * grads[i] for i in range(self.task_num)])
            self._reset_grad(new_grads)
    
    @property
    def backward(self, losses, **kwargs):
        r"""
        Args:
            losses (list): A list of losses of each task.
            kwargs (dict): A dictionary of hyperparameters of weighting methods.
        """
        pass

class MoCo(AbsWeighting):
    r"""MoCo.
    
    This method is proposed in `Mitigating Gradient Bias in Multi-objective Learning: A Provably Convergent Approach (ICLR 2023) <https://openreview.net/forum?id=dLAYGdKTi2>`_ \
    and implemented based on the author' sharing code (Heshan Fernando: fernah@rpi.edu). 

    Args:
        MoCo_beta (float, default=0.5): The learning rate of y.
        MoCo_beta_sigma (float, default=0.5): The decay rate of MoCo_beta.
        MoCo_gamma (float, default=0.1): The learning rate of lambd.
        MoCo_gamma_sigma (float, default=0.5): The decay rate of MoCo_gamma.
        MoCo_rho (float, default=0): The \ell_2 regularization parameter of lambda's update.

    .. warning::
            MoCo is not supported by representation gradients, i.e., ``rep_grad`` must be ``False``.

    """
    def __init__(self):
        super(MoCo, self).__init__()

    def init_param(self):
        self._compute_grad_dim()
        self.step = 0
        self.y = torch.zeros(self.task_num, self.grad_dim).to(self.device)
        self.lambd = (torch.ones([self.task_num, ]) / self.task_num).to(self.device)
        
    def backward(self, losses, **kwargs):
        self.step += 1
        beta, beta_sigma = kwargs['MoCo_beta'], kwargs['MoCo_beta_sigma']
        gamma, gamma_sigma = kwargs['MoCo_gamma'], kwargs['MoCo_gamma_sigma']
        rho = kwargs['MoCo_rho']

        if self.rep_grad:
            raise ValueError('No support method MoCo with representation gradients (rep_grad=True)')
        else:
            self._compute_grad_dim()
            grads = self._compute_grad(losses, mode='backward')

        with torch.no_grad():
            for tn in range(self.task_num):
                grads[tn] = grads[tn]/(grads[tn].norm()+1e-8)*losses[tn]
        self.y = self.y - (beta/self.step**beta_sigma) * (self.y - grads)
        self.lambd = F.softmax(self.lambd - (gamma/self.step**gamma_sigma) * (self.y@self.y.t()+rho*torch.eye(self.task_num).to(self.device))@self.lambd, -1)
        new_grads = self.y.t()@self.lambd

        self._reset_grad(new_grads)
        return self.lambd.detach().cpu().numpy()

class MoCoLoss(MoCo):
    def __init__(self, model, task_num, device, beta, beta_sigma, gamma, gamma_sigma, rho):
        super().__init__()
        self.model = model
        self.task_num = task_num
        self.device = device
        self.beta = beta
        self.beta_sigma = beta_sigma
        self.gamma = gamma
        self.gamma_sigma = gamma_sigma
        self.rho = rho
        self.rep_grad = False
        self.init_param()
    
    def get_share_params(self):
        if hasattr(self.model, 'get_shared_parameters'):
            raise ValueError('The model has no shared parameters')
        return self.model.get_shared_params()

    def zero_grad_share_params(self):
        r"""Set gradients of the shared parameters to zero.
        """
        if hasattr(self.model, 'zero_grad_shared_parameters'):
            raise ValueError('The model has no shared parameters')
        self.model.zero_grad_shared_params()
    
    def loss(self, losses):
        return self.backward(
            losses,
            MoCo_beta=self.beta, 
            MoCo_beta_sigma=self.beta_sigma,
            MoCo_gamma=self.gamma,
            MoCo_gamma_sigma=self.gamma_sigma,
            MoCo_rho=self.rho
        )