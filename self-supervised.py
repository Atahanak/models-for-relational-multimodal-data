import fire
import torch
import os
import re
import torch.nn.functional as F
from torch_frame import stype
from torch_frame.data import DataLoader
from torch_frame.nn import (
    EmbeddingEncoder,
    LinearEncoder,
    TimestampEncoder,
)
from tqdm import tqdm
from src.datasets.util.mask import PretrainType
from transformers import get_inverse_sqrt_schedule

from icecream import ic
import wandb

torch.set_float32_matmul_precision('high')
torch.autograd.set_detect_anomaly(False)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
num_numerical, num_categorical, num_columns = 0, 0, 0

def parse_checkpoint(checkpoint):
    if not checkpoint:
        raise ValueError('No checkpoint file provided')

    # Construct the full file path
    full_path = os.path.join(os.getcwd(), checkpoint)
    # Check if the file exists
    if not os.path.isfile(full_path):
        raise ValueError('Checkpoint file does not exist')

    # Extract information from the file name
    pattern = r"^run_(?P<run_id>[a-zA-Z0-9]+)_epoch_(?P<epoch>\d+)\.pth$"
    match = re.match(pattern, os.path.basename(checkpoint))
    if match:
        run_id = match.group("run_id")
        epoch = match.group("epoch")
        print(f'Continuing run_{run_id} using checkpoint file: {checkpoint} from epoch {epoch}')
        return run_id, int(epoch)
    else:
        raise ValueError('Checkpoint file has invalid format')


def init_wandb(args, run_name: str, wandb_dir: str, run_id: str|None):
    wandb.login()
    if run_id is not None:
        run = wandb.init(
            id=run_id,
            entity="cse3000",
            dir= wandb_dir,
            mode="disabled" if args['testing'] else "online",
            project="rel-mm",
            name=run_name,
            config=args,
            resume="must"
        )
    else:
        run = wandb.init(
            entity="cse3000",
            dir= wandb_dir,
            mode="disabled" if args['testing'] else "online",
            project="rel-mm",
            name=run_name,
            config=args
        )
    wandb.log({"device": str(device)})
    return run

def parse_pretrain_args(pretrain) -> set[PretrainType]:

    pretrain_dict = {
        "mask": PretrainType.MASK,
        "mv": PretrainType.MASK_VECTOR,
        "lp": PretrainType.LINK_PRED,
    }

    pretrain_set = set()
    for pretrain_type in pretrain:
        pretrain_set.add(pretrain_dict[pretrain_type])

    return pretrain_set



def prepare_dataset(dataset_path, pretrain_set):
    from src.datasets import IBMTransactionsAML
    dataset = IBMTransactionsAML(root=os.getcwd() + dataset_path, pretrain=pretrain_set)
    ic(dataset)
    dataset.materialize()
    global num_numerical, num_categorical, num_columns
    num_numerical = len(dataset.tensor_frame.col_names_dict[stype.numerical])
    num_categorical = len(dataset.tensor_frame.col_names_dict[stype.categorical])
    num_columns = num_numerical + num_categorical
    dataset.df.head(5)

    ic(num_numerical, num_categorical, num_columns)
    return dataset

# Set up data loaders
def setup_data_loaders(dataset, batch_size):
    train_dataset, val_dataset, test_dataset = dataset.split()
    train_loader = DataLoader(train_dataset.tensor_frame, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset.tensor_frame, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset.tensor_frame, batch_size=batch_size, shuffle=False)
    ic(len(train_loader), len(val_loader), len(test_loader))
    wandb.log({
        "train_loader size": len(train_loader),
        "val_loader size": len(val_loader),
        "test_loader size": len(test_loader)
    })
    return train_loader, val_loader, test_loader

def initialize_model(dataset, device, channels, num_layers, pretrain_set, is_compile, checkpoint):
    from src.nn.models.ft_transformer import FTTransformer

    model = FTTransformer(
        channels=channels,
        out_channels=None,
        num_layers=num_layers,
        col_stats=dataset.col_stats,
        col_names_dict=dataset.tensor_frame.col_names_dict,
        stype_encoder_dict={
            stype.categorical: EmbeddingEncoder(),
            stype.numerical: LinearEncoder(),
            stype.timestamp: TimestampEncoder(),
        },
        pretrain=pretrain_set
    ).to(device)

    if checkpoint:
        # get epoch from checkpoint file name
        checkpoint.split(".pth")
        pattern = r"^saved_models/self-supervised/run_(?P<identifier>[a-zA-Z0-9]+)_epoch_(?P<epoch>\d+)\.pth$"
        match = re.match(pattern, checkpoint)

        if match:
            identifier = match.group("identifier")
            epoch = match.group("epoch")
            print(f'Continuing run_{identifier} using checkpoint file: {checkpoint} from epoch {epoch}')
            model.load_state_dict(torch.load(checkpoint, map_location=device))
        else:
            raise ValueError('Checkpoint file has invalid format')
    else:
        model = torch.compile(model, dynamic=True) if is_compile else model

    return model



# Set up optimizer and scheduler
def setup_optimizer(model, lr, eps, weight_decay):
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=lr, eps=eps)
    # scheduler = get_inverse_sqrt_schedule(optimizer, num_warmup_steps=0, timescale=1000)
    learnable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    ic(learnable_params)
    wandb.log({"learnable_params": learnable_params})
    return optimizer

def train(model, train_loader, optimizer, epoch: int) -> float:
    model.train()
    loss_accum = loss_c_accum = loss_n_accum = total_count = t_c = t_n = 0

    with tqdm(train_loader, desc=f'Epoch {epoch}') as t:
        for tf in t:
            tf = tf.to(device)
            pred = model(tf)
            loss, loss_c, loss_n = calc_loss(pred, tf.y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_accum += loss.item() * len(tf.y)
            loss_c_accum += loss_c[0].item()
            loss_n_accum += loss_n[0].item()
            total_count += len(tf.y)
            t_c += loss_c[1]
            t_n += loss_n[1]
            t.set_postfix(loss=f'{loss_accum / total_count:.4f}',
                          loss_c=f'{loss_c_accum / t_c:.4f}',
                          loss_n=f'{loss_n_accum / t_n:.4f}')
            del loss
            del loss_c
            del loss_n
            del pred
            del tf
            wandb.log({"train_loss_mcm": loss_accum / total_count,
                       "train_loss_c": loss_c_accum / t_c,
                       "train_loss_n": loss_n_accum / t_n,
                       "epoch": epoch})
    return ((loss_c_accum / t_c) * (num_categorical / num_columns)) + (
            (loss_n_accum / t_n) * (num_numerical / num_columns))

@torch.no_grad()
def test(model, test_loader: DataLoader, dataset_name: str, epoch: int) -> float:
    model.eval()
    accum_acc = accum_l2 = 0
    loss_c_accum = loss_n_accum = 0
    t_n = t_c = 0
    with tqdm(test_loader, desc=f'Evaluating') as t:
        for tf in t:
            tf = tf.to(device)
            pred = model(tf)
            _, loss_c, loss_n = calc_loss(pred, tf.y)
            loss_c_accum += loss_c[0].item()
            loss_n_accum += loss_n[0].item()
            t_c += loss_c[1]
            t_n += loss_n[1]
            for i, ans in enumerate(tf.y):
                # ans --> [val, idx]
                # pred --> feature_type_num X type_num X batch_size
                if ans[1] > (num_numerical - 1):
                    accum_acc += (pred[1][int(ans[1]) - num_numerical][i].argmax() == int(ans[0]))
                else:
                    accum_l2 += torch.square(ans[0] - pred[0][i][int(ans[1])])  #rmse

            # loss numerical
            loss_c_mcm = ((loss_c_accum / t_c) * (num_categorical / num_columns)) + ((loss_n_accum / t_n) * (num_numerical / num_columns))
            loss_c = loss_c_accum / t_c
            loss_n = loss_n_accum / t_n
            wandb.log({f"{dataset_name}_loss_mcm": loss_c_mcm,
                       f"{dataset_name}_loss_c": loss_c,
                       f"{dataset_name}_loss_n": loss_n,
                       "epoch": epoch})

            acc = accum_acc/t_c
            rmse = torch.sqrt(accum_l2/t_n)
            loss = (loss_c_accum / t_c) + (loss_n_accum / t_n)
            t.set_postfix(accuracy=f'{acc:.4f}',
                          rmse=f'{rmse:.4f}',
                          loss=f'{loss:.4f}',
                          loss_c_mcm=f'{loss_c_mcm:.4f}',
                          loss_c=f'{loss_c:.4f}',
                          loss_n=f'{loss_n:.4f}')


        wandb.log({f"{dataset_name}_accuracy": accum_acc / t_c,
                   f"{dataset_name}_rmse": rmse,
                   f"{dataset_name}_loss": loss,
                   f"{dataset_name}_loss_c_mcm": loss_c_mcm,
                   f"{dataset_name}_loss_c": loss_c_accum / t_c,
                   f"{dataset_name}_loss_n": loss_n_accum / t_n,
                   "epoch": epoch})
        del tf
        del pred
        return [rmse, acc]


def calc_loss(pred, y):
    accum_n = accum_c = t_n = t_c = 0
    for i, ans in enumerate(y):
        # ans --> [val, idx]
        # pred --> feature_type_num X type_num X batch_size
        if ans[1] > (num_numerical - 1):
            t_c += 1
            a = torch.tensor(int(ans[0])).to(device)
            accum_c += F.cross_entropy(pred[1][int(ans[1]) - num_numerical][i], a)
            del a
        else:
            t_n += 1
            accum_n += torch.square(pred[0][i][int(ans[1])] - ans[0])  #mse
    return (accum_n / t_n) + torch.sqrt(accum_c / t_c), (accum_c, t_c), (accum_n, t_n)


def main(dataset="/data/Over-Sampled_Tiny_Trans-c.csv", run_name="self-supervised", checkpoint="saved_models/self-supervised/run_o8rmsna8_epoch_11.pth",
         testing=False, seed=42,batch_size=200, channels=128, num_layers=3, data_split=[0.6, 0.2, 0.2], split_type="temporal",
         pretrain=["mask"], is_compile=False, lr=2e-4, eps=1e-8, weight_decay=1e-3, epochs=10, wand_dir="/mnt/data/"):

    args = {
        "testing": testing,
        "seed": seed,
        "batch_size": batch_size,
        "channels": channels,
        "num_layers": num_layers,
        "pretrain": pretrain,
        "compile": is_compile,
        "lr": lr,
        "eps": eps,
        "epochs": epochs,
        "data_split": data_split,
        "split_type": split_type,
        "weight_decay": weight_decay,
        "device": device
    }
    torch.manual_seed(args['seed'])

    if checkpoint:
        run_id, checkpoint_epoch = parse_checkpoint(checkpoint)
    else:
        run_id, checkpoint_epoch = None, None

    init_wandb(args, run_name, wand_dir, run_id)
    pretrain_set = parse_pretrain_args(pretrain)
    dataset = prepare_dataset(dataset, pretrain_set)
    train_loader, val_loader, test_loader = setup_data_loaders(dataset, batch_size)

    model = initialize_model(dataset, device, channels, num_layers, pretrain_set, is_compile, checkpoint)
    optimizer = setup_optimizer(model, lr, eps, weight_decay)

    save_dir = 'saved_models/self-supervised'
    run_id = wandb.run.id
    os.makedirs(save_dir, exist_ok=True)

    if checkpoint_epoch is not None:
        start_epoch = checkpoint_epoch + 1
        end_epoch = checkpoint_epoch + epochs + 1
    else:
        start_epoch = 1
        end_epoch = epochs + 1

    for epoch in range(start_epoch, end_epoch):
        train_loss = train(model, train_loader, optimizer, epoch)
        train_metric = test(model, train_loader, "tr", epoch)
        val_metric = test(model, val_loader, "val", epoch)
        test_metric = test(model, test_loader, "test", epoch)
        ic(
            train_loss,
            train_metric,
            val_metric,
            test_metric
        )
        model_save_path = os.path.join(save_dir, f'run_{run_id}_epoch_{epoch}.pth')
        torch.save(model.state_dict(), model_save_path)

    wandb.finish()


if __name__ == "__main__":
    fire.Fire(main)
