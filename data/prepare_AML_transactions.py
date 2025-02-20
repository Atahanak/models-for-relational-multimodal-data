"""Script to pre-process AML transaction data to be used in training and inference."""
import os
import argparse
import logging
from datetime import datetime

import pandas as pd
import numpy as np
from icecream import ic

# Setup logging
logger = logging.getLogger(__name__)
logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)


def main(in_path: str, out_path: str):

    df = pd.read_csv(in_path)
    logger.info(f"Number of transactions: {len(df)}")

    df.rename(columns={'Account': 'From ID', 'Account.1': 'To ID'}, inplace=True)
    # create ids
    df['From ID'] = df['From ID'].astype(str) + df['From Bank'].astype(str)
    df['To ID'] = df['To ID'].astype(str) + df['To Bank'].astype(str)
    # get all unique items in columns From ID and To ID
    unique_ids = pd.concat([df['From ID'], df['To ID']]).unique()
    logger.info(f'Number of Unique IDs: {len(unique_ids)}')

    # create a mapping from the original ID to a new ID
    id_map = {old_id: new_id for new_id, old_id in enumerate(unique_ids)}
    # replace the original IDs with the new IDs
    df['From ID'] = df['From ID'].apply(lambda x: id_map[x])
    df['To ID'] = df['To ID'].apply(lambda x: id_map[x])

    # Convert the datetime object to Unix time (POSIX time)
    timestamp_format = "%Y/%m/%d %H:%M"
    format_fn = lambda x: int(datetime.strptime(x, timestamp_format).timestamp())
    df['Timestamp'] = df['Timestamp'].apply(format_fn)

    df['From Bank'] = df['From Bank'].apply(lambda b: f'B_{b}')
    df['To Bank'] = df['To Bank'].apply(lambda b: f'B_{b}')
    df.dropna()

    # normalize numerical features
    for col in df.columns:
        if df[col].dtype == 'float64':
            df[col] = np.log1p(df[col])
            # take z-normalization
            # df[col] = (df[col] - df[col].mean()) / df[col].std()
            # ic(df[col].min(), df[col].max(), df[col].mean())
            df[col] = (df[col] - df[col].min()) / (df[col].max() - df[col].min())

    logger.info(f'Edge data:\n {str(df)}')
    logger.info(f'Saving edge data:\n {out_path}')
    df.to_csv(out_path, index=False)
    logger.info(f'Saved edge data in {out_path}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("-i", required=True, type=str, help="Input transactions CSV file")
    parser.add_argument("-o", required=True, type=str, help="Output formatted transactions CSV file")

    args = parser.parse_args()

    main(args.i, args.o)