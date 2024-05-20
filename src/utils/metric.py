import numpy as np
import torch.nn.functional as F
import torch


class SSMetric:
    def __init__(self, num_numerical):
        self.num_numerical = num_numerical

    @staticmethod
    def mrr(pos_pred, neg_pred, ks, num_neg_samples) -> (float, dict[str, float]):
        """Compute mean reciprocal rank (MRR) and Hits@k.

        Returns
        -------
        float, dict[str, float]
            MRR and dictionary of Hits@k metrics.
        """
        pos_pred = pos_pred.detach().clone().cpu().numpy().flatten()
        neg_pred = neg_pred.detach().clone().cpu().numpy().flatten()

        num_positives = len(pos_pred)
        neg_pred_reshaped = neg_pred.reshape(num_positives, num_neg_samples)

        mrr_scores = []
        keys = [f'hits@{k}' for k in ks]
        hits_dict = {key: 0 for key in keys}
        count = 0

        for pos, neg in zip(pos_pred, neg_pred_reshaped):
            # Combine positive and negative predictions
            combined = np.concatenate([neg, [pos]])  # Add positive prediction to the end

            # Rank predictions (argsort twice gives the ranks)
            ranks = (-combined).argsort().argsort() + 1  # Add 1 because ranks start from 1
            for k, key in zip(ks, keys):
                if ranks[-1] <= k:
                    hits_dict[key] += 1

            count += 1
            # Reciprocal rank of positive prediction (which is the last one in combined)
            reciprocal_rank = 1 / ranks[-1]
            mrr_scores.append(reciprocal_rank)

        # Calculate Hits@k
        for key in keys:
            hits_dict[key] /= count

        # Calculate Mean Reciprocal Rank
        mrr = np.mean(mrr_scores)

        return mrr, hits_dict

    def mv_accuracy(self, mv_out, y, tol=0.1):
        mask_acc = total = 0
        for i, (values, mask_vector) in enumerate(y):
            # Did mv_out predict the correct mask vector?
            mask_acc += 1 if torch.argmax(mv_out[i]) == mask_vector else 0
            total += 1
            # for j, p in enumerate(mv_out[i]):
            #     r = 1 if j == mask_vector else 0
            #     mask_acc += 1 if ((r == 1 and p > 1 - tol) or (r == 0 and p < tol)) else 0
            #     total += 1
        return mask_acc / total


# def mrr(pos_pred, neg_pred, ks, num_neg_samples) -> (float, dict[str, float]):
#     """Compute mean reciprocal rank (MRR) and Hits@k for link prediction.
    
#     Returns
#     -------
#     float, dict[str, float]
#         MRR and dictionnary with Hits@k metrics. 
#     """
#     pos_pred = pos_pred.detach().clone().cpu().numpy().flatten()
#     neg_pred = neg_pred.detach().clone().cpu().numpy().flatten()

#     num_positives = len(pos_pred)
#     neg_pred_reshaped = neg_pred.reshape(num_positives, num_neg_samples)

#     mrr_scores = []
#     keys = [f'hits@{k}' for k in ks]
#     hits_dict = {key: 0 for key in keys}
#     count = 0

#     for pos, neg in zip(pos_pred, neg_pred_reshaped):
#         # Combine positive and negative predictions
#         combined = np.concatenate([neg, [pos]])  # Add positive prediction to the end
        
#         # Rank predictions (argsort twice gives the ranks)
#         ranks = (-combined).argsort().argsort() + 1  # Add 1 because ranks start from 1
#         # from icecream import ic
#         # ic(ranks[-1])
#         for k, key in zip(ks, keys):
#             if ranks[-1] <= k:
#                 hits_dict[key] += 1
        
#         count += 1
#         # Reciprocal rank of positive prediction (which is the last one in combined)
#         reciprocal_rank = 1 / ranks[-1]
#         mrr_scores.append(reciprocal_rank)
    
#     # Calculate Hits@k
#     # for key in keys:
#     #     hits_dict[key] /= count

#     # Calculate Mean Reciprocal Rank
#     mrr = np.mean(mrr_scores)
    
#     return mrr, hits_dict