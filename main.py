import os
import pandas as pd
import csv
import time
import numpy as np

from datasets.dataset_survival import Generic_MIL_Survival_Dataset
from utils.options import parse_args
from utils.util import get_split_loader, set_seed
import torch
from utils.loss import define_loss
from utils.optimizer import define_optimizer
from utils.scheduler import define_scheduler


def main(args):

    # set random seed for reproduction
    set_seed(args.seed)
    device = torch.device(args.gpu if torch.cuda.is_available() else "cpu")
    # create results directory
    results_dir = "./results/{dataset}/[{model}]-[{alpha}]-[{beta}]-[{time}]".format(
        dataset=args.dataset,
        model=args.model,
        alpha=args.alpha,
        beta=args.beta,
        time=time.strftime("%Y-%m-%d]-[%H-%M-%S"),
    )
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    # 5-fold cross validation
    header = ["folds", "fold 0", "fold 1", "fold 2", "fold 3", "fold 4", "mean", "std"]
    best_epoch = ["best epoch"]
    best_score = ["best cindex"]
    data_dir = args.data_root_dir
    # start 5-fold CV evaluation.
    for fold in range(5):
        # build dataset
        dataset = Generic_MIL_Survival_Dataset(
            csv_path="./csv/%s_all_clean.csv" % (args.dataset),
            modal=args.modal,
            OOM=args.OOM,
            apply_sig=True,
            data_dir=data_dir,
            shuffle=False,
            seed=args.seed,
            patient_strat=False,
            n_bins=4,
            label_col="survival_months",
        )
        split_dir = os.path.join("./splits", args.which_splits, args.dataset)
        train_dataset, val_dataset = dataset.return_splits(
            from_id=False, csv_path="{}/splits_{}.csv".format(split_dir, fold)
        )
        train_loader = get_split_loader(
            train_dataset,
            training=True,
            weighted=args.weighted_sample,
            modal=args.modal,
            batch_size=args.batch_size,
        )
        val_loader = get_split_loader(
            val_dataset, modal=args.modal, batch_size=args.batch_size
        )
        print("fold:{}-----------------------------------------".format(fold))
        print("data_dir:{}".format(data_dir))
        print(
            "training: {}, validation: {}".format(len(train_dataset), len(val_dataset))
        )
        print("model:{}".format(args.model))
        print("alpha:{}".format(args.alpha))
        print("beta:{}".format(args.beta))
        # build model, criterion, optimizer, schedular
        if args.model == "MDCS_MoAME":
            from models.MDCS_MoAME.MDCS_MoAME import MDCS_MoAME
            print(train_dataset.omic_sizes)
            model_dict = {
                "omic_sizes": train_dataset.omic_sizes,
                "n_classes": 4,
                "model_size": args.model_size,
            }
            model = MDCS_MoAME(**model_dict)
        else:
            raise NotImplementedError(
                "Model [{}] is not implemented".format(args.model)
            )

        from models.MDCS_MoAME.engine import Engine
        criterion = define_loss(args)
        optimizer = define_optimizer(args, model)
        scheduler = define_scheduler(args, optimizer)
        engine = Engine(args, results_dir, fold)
        # start training

        score, epoch = engine.learning(
            model, train_loader, val_loader, criterion, optimizer, scheduler,device
        )
        # save best score and epoch for each fold
        best_epoch.append(epoch)
        best_score.append(score)

    # finish training
    # mean and std
    best_epoch.append("~")
    best_epoch.append("~")
    best_score.append(np.mean(best_score[1:6]))
    best_score.append(np.std(best_score[1:6]))

    csv_path = os.path.join(results_dir, "results.csv")
    print("############", csv_path)
    with open(csv_path, "w", encoding="utf-8", newline="") as fp:
        writer = csv.writer(fp)
        writer.writerow(header)
        writer.writerow(best_epoch)
        writer.writerow(best_score)

    df = pd.read_csv(csv_path)
    for index, row in df.iterrows():
        print(row)
    mean_value = float(df['mean'][1])
    std_value = float(df['std'][1])
    print(f"result: {mean_value:.4f} ± {std_value:.4f}")
if __name__ == "__main__":
    args = parse_args()
    results = main(args)
    print("finished!")
