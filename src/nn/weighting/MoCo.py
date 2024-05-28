from LibMTL.weighting import MoCo

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
        return self.model.parameters()

    def zero_grad_share_params(self):
        r"""Set gradients of the shared parameters to zero.
        """
        #shared_params = self.get_share_params()
        self.model.zero_grad(set_to_none=False)
    
    def loss(self, losses):
        return self.backward(
            losses,
            MoCo_beta=self.beta, 
            MoCo_beta_sigma=self.beta_sigma,
            MoCo_gamma=self.gamma,
            MoCo_gamma_sigma=self.gamma_sigma,
            MoCo_rho=self.rho
        )