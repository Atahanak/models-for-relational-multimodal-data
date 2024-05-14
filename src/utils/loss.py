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

    def mv_loss(self, pred, y):
        """
        Computes the mask prediction loss, as described in "VIME: Extending the Success of Self- and Semi-supervised Learning to Tabular Domain".
        Returns:
            Sum of all losses,
            Cross-entropy loss of the mask vector
            Cross-entropy loss of the categorical columns,
            RMSE of the numerical columns.
        """
        num_out, cat_out, mv_out = pred
        accum_n = accum_c = t_n = t_c = vector_loss = 0
        for i, (values, mask_vector) in enumerate(y):
            # 1. Cross entropy between the predicted mask vector and the target mask vector
            normalized_mv_out = expit(mv_out[i])
            for p, r in zip(normalized_mv_out, mask_vector):
                vector_loss += F.binary_cross_entropy(p, r)
            # 2. Reconstruction loss of the (masked) numerical and categorical columns
            for j, val in enumerate(values):
                if val.isnan():  # only compute loss for masked values
                    continue
                if j > (self.num_numerical - 1):
                    t_c += 1
                    a = torch.tensor(int(val)).to(self.device)
                    accum_c += F.cross_entropy(cat_out[int(j) - self.num_numerical][j], a)
                    del a
                else:
                    t_n += 1
                    accum_n += torch.square(num_out[i][int(j)] - val)  # mse
        vector_loss = torch.sqrt(vector_loss / len(y))
        # return torch.sqrt(vector_loss / len(y)) + (accum_n / t_n) + torch.sqrt(accum_c / t_c), vector_loss, (accum_c, t_c), (accum_n, t_n)
        return vector_loss + (accum_c / t_c) + (accum_n / t_n), vector_loss, (accum_c, t_c), (accum_n, t_n)








        # vector_loss = 0
        # accum_n = accum_c = t_n = t_c = 0
        # for j, ans in enumerate(y):
        #     vector_loss += torch.nn.functional.cross_entropy(mv_out[i], ans[1])   # TODO: is this formulation correct?
        #
        #     # 2. Reconstruction loss of the (masked) numerical and categorical columns
        #     print(ans[0])
        #     for i, ans in enumerate(y):
        #         # ans --> [val, idx]
        #         # pred --> feature_type_num X type_num X batch_size
        #         if ans[1] > (num_numerical - 1):
        #             t_c += 1
        #             a = torch.tensor(int(ans[0])).to(device)
        #             accum_c += F.cross_entropy(pred[1][int(ans[1]) - num_numerical][i], a)
        #             del a
        #         else:
        #             t_n += 1
        #             accum_n += torch.square(pred[0][i][int(ans[1])] - ans[0])  # mse
        #     return (accum_n / t_n) + torch.sqrt(accum_c / t_c), (accum_c, t_c), (accum_n, t_n)
