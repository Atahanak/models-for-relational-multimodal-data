import torch
import torch.nn.functional as F
from torch.special import expit


class SSLoss:
    def __init__(self, device, num_numerical=None):
        self.num_numerical = num_numerical
        self.device = device

    @staticmethod
    def lp_loss(pos_pred, neg_pred):
        return -torch.log(pos_pred + 1e-12).mean() - torch.log(1 - neg_pred + 1e-12).mean()

    def mcm_loss(self, cat_out, num_out, y):
        accum_n = accum_c = t_n = t_c = 0
        for i, ans in enumerate(y):
            # ans --> [val, idx]
            # pred --> feature_type_num X type_num X batch_size
            if ans[1] > (self.num_numerical - 1):
                t_c += 1
                a = torch.tensor(int(ans[0])).to(self.device)
                accum_c += F.cross_entropy(cat_out[int(ans[1]) - self.num_numerical][i], a)
                del a
            else:
                t_n += 1
                accum_n += torch.square(num_out[i][int(ans[1])] - ans[0])  # mse
        return (accum_n / t_n) + torch.sqrt(accum_c / t_c), (accum_c, t_c), (accum_n, t_n)

    def mv_loss(self, mv_out, y):
        """
        Computes the mask prediction loss, as described in "VIME: Extending the Success of Self- and Semi-supervised Learning to Tabular Domain".
        """
        return F.cross_entropy(mv_out, y[:, 1].long())
