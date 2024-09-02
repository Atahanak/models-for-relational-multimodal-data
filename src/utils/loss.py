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

    # def mcm_loss(self, cat_out, num_out, y):
    #     # accum_n = torch.tensor(0.0, dtype=torch.float32, device=self.device) 
    #     # accum_c = torch.tensor(0.0, dtype=torch.float32, device=self.device)
    #     accum_n = torch.tensor(0.0, dtype=torch.float32) 
    #     accum_c = torch.tensor(0.0, dtype=torch.float32)
    #     t_n = t_c = 0
    #     for i, ans in enumerate(y):
    #         # ans --> [val, idx]
    #         # pred --> feature_type_num X type_num X batch_size
    #         if ans[1] > (self.num_numerical - 1):
    #             t_c += 1
    #             # a = torch.tensor(int(ans[0]), device=self.device)
    #             a = torch.tensor(int(ans[0]))
    #             accum_c += F.cross_entropy(cat_out[int(ans[1]) - self.num_numerical][i], a)
    #         else:
    #             t_n += 1
    #             accum_n += torch.square(num_out[i][int(ans[1])] - ans[0])  # mse
    #             # if torch.isnan(accum_n):
    #             #     print('accum_n is nan')
    #             #     print(num_out[i][int(ans[1])], ans[0])
    #             #     exit()
    #     if t_c == 0:
    #         return torch.sqrt(accum_n / t_n), (accum_c, t_c), (accum_n, t_n)
    #     elif t_n == 0:
    #         return (accum_c / t_c), (accum_c, t_c), (accum_n, t_n)
    #     return (accum_c / t_c) + torch.sqrt(accum_n / t_n), (accum_c, t_c), (accum_n, t_n)
    
    def mcm_loss(self, cat_out, num_out, y):
        y_val, y_idx = y[:, 0], y[:, 1].long()

        # Separate categorical and numerical indices
        cat_mask = y_idx >= self.num_numerical
        num_mask = ~cat_mask

        # Categorical loss
        cat_indices = y_idx[cat_mask] - self.num_numerical
        cat_targets = y_val[cat_mask].long()
        
        cat_loss = torch.tensor(0.0, device=y.device)
        acc = torch.tensor(0.0, device=y.device)
        cat_sample_indices = torch.where(cat_mask)[0]
        for i, (idx, target) in enumerate(zip(cat_indices, cat_targets)):
            cat_loss += F.cross_entropy(cat_out[idx][cat_sample_indices[i]], target)
            acc += torch.argmax(cat_out[idx][cat_sample_indices[i]], dim=0) == target
        
        # Numerical loss
        num_preds = num_out[num_mask, y_idx[num_mask]]
        num_targets = y_val[num_mask]
        num_loss = torch.sum((num_preds - num_targets) ** 2)
        
        t_c = cat_mask.sum().item()
        t_n = num_mask.sum().item()
        
        if t_c == 0:
            return torch.sqrt(num_loss / t_n), (cat_loss, t_c, acc), (num_loss, t_n)
        elif t_n == 0:
            return cat_loss / t_c, (cat_loss, t_c, acc), (num_loss, t_n)
        
        return (cat_loss / t_c) + torch.sqrt(num_loss / t_n), (cat_loss, t_c, acc), (num_loss, t_n)

    def mv_loss(self, mv_out, y):
        """
        Computes the mask prediction loss, as described in "VIME: Extending the Success of Self- and Semi-supervised Learning to Tabular Domain".
        """
        return F.cross_entropy(mv_out, y[:, 1].long())
