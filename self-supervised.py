import fire
import torch
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
import os
from icecream import ic
import wandb

torch.set_float32_matmul_precision('high')


def main(dataset="", run_name="self-supervised", checkpoint=None, testing=False, seed=42, batch_size=200, channels=128, num_layers=3,
         data_split=[0.6, 0.2, 0.2], split_type="temporal", pretrain=["mask"], is_compile=False,
         lr=2e-4, eps=1e-8, weight_decay=1e-3, epochs=3, wand_dir="/mnt/data/"):

    pretrain_dict = {
        "mask": PretrainType.MASK,
        "mv": PretrainType.MASK_VECTOR,
        "lp": PretrainType.LINK_PRED,
    }

    pretrain_set = set()
    for pretrain_type in pretrain:
        pretrain_set.add(pretrain_dict[pretrain_type])

    args = {
        "testing": testing,
        "seed": seed,
        "batch_size": batch_size,
        "channels": channels,
        "num_layers": num_layers,
        "pretrain": pretrain_set,
        "compile": is_compile,
        "lr": lr,
        "eps": eps,
        "epochs": epochs,
        "data_split": data_split,
        "split_type": split_type,
        'weight_decay': weight_decay
    }

    wandb.login()
    run = wandb.init(
        dir=wand_dir,
        mode="disabled" if args['testing'] else "online",
        project="rel-mm-2",
        name=run_name,
        config=args
    )

    from src.datasets import IBMTransactionsAML
    dataset = IBMTransactionsAML(root=dataset, pretrain=pretrain_set)
    ic(dataset)
    dataset.materialize()
    num_numerical = len(dataset.tensor_frame.col_names_dict[stype.numerical])
    num_categorical = len(dataset.tensor_frame.col_names_dict[stype.categorical])
    dataset.df.head(5)

    num_columns = num_numerical + num_categorical
    ic(num_numerical, num_categorical, num_columns)

    torch.manual_seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    wandb.log({"device": str(device)})

    train_dataset, val_dataset, test_dataset = dataset.split()

    train_tensor_frame = train_dataset.tensor_frame
    val_tensor_frame = val_dataset.tensor_frame
    test_tensor_frame = test_dataset.tensor_frame
    train_loader = DataLoader(train_tensor_frame, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_tensor_frame, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_tensor_frame, batch_size=batch_size, shuffle=False)
    ic(len(train_loader), len(val_loader), len(test_loader))
    wandb.log({
        "train_loader size": len(train_loader),
        "val_loader size": len(val_loader),
        "test_loader size": len(test_loader)
    })

    stype_encoder_dict = {
        stype.categorical: EmbeddingEncoder(),
        stype.numerical: LinearEncoder(),
        stype.timestamp: TimestampEncoder(),
    }

    from src.nn.models.ft_transformer import FTTransformer
    model = FTTransformer(
        channels=channels,
        out_channels=None,
        num_layers=num_layers,
        col_stats=dataset.col_stats,
        col_names_dict=train_tensor_frame.col_names_dict,
        stype_encoder_dict=stype_encoder_dict,
        pretrain=pretrain_set
    ).to(device)

    if checkpoint:
        print(f'running from checkpoint file: {checkpoint}')
        model.load_state_dict(torch.load(checkpoint, map_location=device))
    else:
        model = torch.compile(model, dynamic=True) if is_compile else model

    learnable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    ic(learnable_params)
    wandb.log({"learnable_params": learnable_params})

    # Prepare optimizer and lr scheduler
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    #optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=lr, eps=eps)
    scheduler = get_inverse_sqrt_schedule(optimizer, num_warmup_steps=0, timescale=1000)

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

    def train(epoc: int) -> float:
        model.train()
        loss_accum = loss_c_accum = loss_n_accum = total_count = t_c = t_n = 0

        with tqdm(train_loader, desc=f'Epoch {epoc}') as t:
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
                t.set_postfix(loss=f'{loss_accum / total_count:.4f}', loss_c=f'{loss_c_accum / t_c:.4f}',
                              loss_n=f'{loss_n_accum / t_n:.4f}')
                del loss
                del loss_c
                del loss_n
                del pred
                del tf
                wandb.log({"train_loss_mcm": loss_accum / total_count, "train_loss_c": loss_c_accum / t_c,
                           "train_loss_n": loss_n_accum / t_n})
        return ((loss_c_accum / t_c) * (num_categorical / num_columns)) + (
                    (loss_n_accum / t_n) * (num_numerical / num_columns))

    @torch.no_grad()
    def test(loader: DataLoader, dataset_name) -> float:
        model.eval()
        accum_acc = accum_l2 = 0
        loss_c_accum = loss_n_accum = 0
        t_n = t_c = 0
        with tqdm(loader, desc=f'Evaluating') as t:
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

                wandb.log({f"{dataset_name}_loss_mcm": ((loss_c_accum / t_c) * (num_categorical / num_columns)) + (
                            (loss_n_accum / t_n) * (num_numerical / num_columns)),
                           f"{dataset_name}_loss_c": loss_c_accum / t_c, f"{dataset_name}_loss_n": loss_n_accum / t_n})
                t.set_postfix(accuracy=f'{accum_acc / t_c:.4f}', rmse=f'{torch.sqrt(accum_l2 / t_n):.4f}',
                              loss=f'{(loss_c_accum / t_c) + (loss_n_accum / t_n):.4f}',
                              loss_c=f'{loss_c_accum / t_c:.4f}', loss_n=f'{loss_n_accum / t_n:.4f}')
            wandb.log({f"{dataset_name}_accuracy": accum_acc / t_c, f"{dataset_name}_rmse": torch.sqrt(accum_l2 / t_n),
                       f"{dataset_name}_loss": ((loss_c_accum / t_c) * (num_categorical / num_columns)) + (
                                   (loss_n_accum / t_n) * (num_numerical / num_columns)),
                       f"{dataset_name}_loss_c": loss_c_accum / t_c, f"{dataset_name}_loss_n": loss_n_accum / t_n})
            del tf
            del pred
            accuracy = accum_acc / t_c
            rmse = torch.sqrt(accum_l2 / t_n)
            return [rmse, accuracy]

    # %%
    torch.autograd.set_detect_anomaly(False)

    # Create a directory to save models
    save_dir = 'saved_models/self-supervised'
    run_id = wandb.run.id
    os.makedirs(save_dir, exist_ok=True)

    for epoch in range(1, epochs + 1):
        train_loss = train(epoch)
        train_metric = test(train_loader, "tr")
        val_metric = test(val_loader, "val")
        test_metric = test(test_loader, "test")
        ic(
            train_loss,
            train_metric,
            val_metric,
            test_metric
        )
        model_save_path = os.path.join(save_dir, f'run_{run_id}_epoch_{epoch}.pth')
        torch.save(model.state_dict(), model_save_path)

    # [Include rest of training, testing, and model saving logic here]
    # Call to Fire at the end of the script:
    wandb.finish()


if __name__ == "__main__":
    fire.Fire(main)
