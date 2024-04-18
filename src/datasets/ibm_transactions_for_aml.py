import torch
import torch_frame
import pandas as pd
import numpy as np
import itertools

import sys
from icecream import ic

class IBMTransactionsAML(torch_frame.data.Dataset):
        r"""`"Realistic Synthetic Financial Transactions for Anti-Money Laundering Models" https://arxiv.org/pdf/2306.16424.pdf`_.
        
        IBM Transactions for Anti-Money Laundering (AML) dataset.
        The dataset contains 10 columns:
        - Timestamp: The timestamp of the transaction.
        - From Bank: The bank from which the transaction is made.
        - From ID: The ID of the sender.
        - To Bank: The bank to which the transaction is made.
        - To ID: The ID of the receiver.
        - Amount Received: The amount received by the receiver.
        - Receiving Currency: The currency in which the amount is received.
        - Amount Paid: The amount paid by the sender.
        - Payment Currency: The currency in which the amount is paid.
        - Payment Format: The format of the payment.
        - Is Laundering: The label indicating whether the transaction is a money laundering transaction.

        Args:
            root (str): Root directory of the dataset.
            preetrain (bool): Whether to use the pretrain split or not (default: False).
        """
        def __init__(self, root, pretrain=False, split_type='temporal', splits=[0.6, 0.2, 0.2]):
            self.root = root
            self.split_type = split_type
            self.splits = splits
            names = [
                'Timestamp',
                'From Bank',
                'From ID',
                'To Bank',
                'To ID',
                'Amount Received',
                'Receiving Currency',
                'Amount Paid',
                'Payment Currency',
                'Payment Format',
                'Is Laundering',
            ]
            dtypes = {
                'From Bank': 'category',
                'From ID': 'category',
                'To Bank': 'category',
                'To ID': 'category',
                'Amount Received': 'float64',
                'Receiving Currency': 'category',
                'Amount Paid': 'float64',
                'Payment Currency': 'category',
                'Payment Format': 'category',
                'Is Laundering': 'category',
            }

            self.df = pd.read_csv(root, names=names, dtype=dtypes, header=0)         
            col_to_stype = {
                'From ID': torch_frame.categorical,
                'To ID': torch_frame.categorical,
                'From Bank': torch_frame.categorical,
                'To Bank': torch_frame.categorical,
                'Payment Currency': torch_frame.categorical,
                'Receiving Currency': torch_frame.categorical,
                'Payment Format': torch_frame.categorical,
                'Timestamp': torch_frame.timestamp,
                'Amount Paid': torch_frame.numerical,
                'Amount Received': torch_frame.numerical
            }

            if pretrain:
                self.target_col = 'MASK'
                maskable_columns = ['Amount Received', 'Receiving Currency', 'Amount Paid', 'Payment Currency', 'Payment Format']
                self.df['MASK'] = None
                self.df = self.df.apply(self.mask_column, args=(maskable_columns,), axis=1)
                #ic(self.df.head(5))
                col_to_stype['MASK'] = torch_frame.mask
            else:
                col_to_stype['Is Laundering'] = torch_frame.categorical
                self.target_col = 'Is Laundering'
            
            if self.split_type == 'temporal':
                self.temporal_balanced_split()
            else:
                self.random_split()

            super().__init__(self.df, col_to_stype, split_col='split', target_col=self.target_col)
        
        def random_split(self, seed=42):
            self.df['split'] = torch_frame.utils.generate_random_split(self.df.shape[0], seed)

        def temporal_split(self):
            self.df = self.df.sort_values(by='Timestamp')
            train_size = int(self.df.shape[0] * 0.3)
            validation_size = int(self.df.shape[0] * 0.1)
            test_size = self.df.shape[0] - train_size - validation_size

            #add split column, use 0 for train, 1 for validation, 2 for test
            self.df['split'] = [0] * train_size + [1] * validation_size + [2] * test_size 

        def temporal_balanced_split(self):
            assert 'Timestamp' in self.df.columns, \
            '"transaction" split is only available for datasets with a "Timestamp" column'
            self.df['Timestamp'] = self.df['Timestamp'] - self.df['Timestamp'].min()

            timestamps = torch.Tensor(self.df['Timestamp'].to_numpy())
            n_days = int(timestamps.max() / (3600 * 24) + 1)

            daily_inds, daily_trans = [], [] #irs = illicit ratios, inds = indices, trans = transactions
            for day in range(n_days):
                l = day * 24 * 3600
                r = (day + 1) * 24 * 3600
                day_inds = torch.where((timestamps >= l) & (timestamps < r))[0]
                daily_inds.append(day_inds)
                daily_trans.append(day_inds.shape[0])
            
            split_per = self.splits
            #split_per = [0.8, 0.2]
            daily_totals = np.array(daily_trans)
            d_ts = daily_totals
            I = list(range(len(d_ts)))
            split_scores = dict()
            for i,j in itertools.combinations(I, 2):
                if j >= i:
                    split_totals = [d_ts[:i].sum(), d_ts[i:j].sum(), d_ts[j:].sum()]
                    #split_totals = [d_ts[:i].sum(), d_ts[i:].sum()]
                    split_totals_sum = np.sum(split_totals)
                    split_props = [v/split_totals_sum for v in split_totals]
                    split_error = [abs(v-t)/t for v,t in zip(split_props, split_per)]
                    score = max(split_error) #- (split_totals_sum/total) + 1
                    split_scores[(i,j)] = score
                else:
                    continue
            
            i,j = min(split_scores, key=split_scores.get)
            #split contains a list for each split (train, validation and test) and each list contains the days that are part of the respective split
            split = [list(range(i)), list(range(i, j)), list(range(j, len(daily_totals)))]

            #Now, we seperate the transactions based on their indices in the timestamp array
            split_inds = {k: [] for k in range(3)}
            for i in range(3):
                for day in split[i]:
                    split_inds[i].append(daily_inds[day]) #split_inds contains a list for each split (tr,val,te) which contains the indices of each day seperately

            #tr_inds = torch.cat(split_inds[0])
            val_inds = torch.cat(split_inds[1])
            te_inds = torch.cat(split_inds[2])
            
            #add a new split column to df
            self.df['split'] = 0

            # Set values for val_inds and te_inds
            self.df.loc[val_inds, 'split'] = 1
            self.df.loc[te_inds, 'split'] = 2
        
        # Randomly mask a column of each row and store original value and max index
        def mask_column(self, row, maskable_cols):
            col_to_mask = np.random.choice(maskable_cols)  # Choose a column randomly
            original_value = row[col_to_mask]
            row['MASK'] = [original_value, col_to_mask]  # Store original value and max index in 'MASK' column
            row[col_to_mask] = np.nan  # Mask the value
            return row