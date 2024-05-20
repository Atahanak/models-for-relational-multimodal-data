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

    def mcm_loss(self, num_out, cat_out, y):
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
        accum_mv_loss = t = 0
        # Prepare the target mask vectors
        a = torch.zeros_like(mv_out).to(self.device)
        mask_vector = torch.tensor([int(mask_vector) for _, mask_vector in y]).to(self.device)
        for i, m in enumerate(mask_vector):
            a[i][m] = 1

        # Sum of cross entropy losses for each mask vector prediction
        for i in range(len(y)):
            # Cross entropy between the predicted mask vector and the target mask vector
            accum_mv_loss += F.cross_entropy(mv_out[i], a[i])
            t += 1
        del a
        return accum_mv_loss / t


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
