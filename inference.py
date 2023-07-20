"""Infer rankings with trained model"""
import argparse
import json
import logging
import os

import torch
import torch.optim

import models
import optimizers.regularizers as regularizers
from datasets.kg_dataset import KGDataset
from models import all_models
from optimizers.kg_optimizer import KGOptimizer
from utils.train import get_savedir, avg_both, format_metrics, count_params


parser = argparse.ArgumentParser(
    description="Predict tails by heads and relations."
)
parser.add_argument(
    "--model", default="RotE", choices=all_models, help="Knowledge Graph embedding model"
)
parser.add_argument(
    "--dataset", default="1710", help="Knowledge Graph dataset"
)
parser.add_argument(
    "--save_dir", default="./logs/", help="Model Checkpoint Path"
)
parser.add_argument(
    "--regularizer", choices=["N3", "F2"], default="N3", help="Regularizer"
)
parser.add_argument(
    "--reg", default=0, type=float, help="Regularization weight"
)
parser.add_argument(
    "--rank", default=1000, type=int, help="Embedding dimension"
)
parser.add_argument(
    "--batch_size", default=1000, type=int, help="Batch size"
)
parser.add_argument(
    "--neg_sample_size", default=50, type=int, help="Negative sample size, -1 to not use negative sampling"
)
parser.add_argument(
    "--dropout", default=0, type=float, help="Dropout rate"
)
parser.add_argument(
    "--init_size", default=1e-3, type=float, help="Initial embeddings' scale"
)
parser.add_argument(
    "--learning_rate", default=1e-1, type=float, help="Learning rate"
)
parser.add_argument(
    "--gamma", default=0, type=float, help="Margin for distance-based losses"
)
parser.add_argument(
    "--bias", default="constant", type=str, choices=["constant", "learn", "none"], help="Bias type (none for no bias)"
)
parser.add_argument(
    "--dtype", default="double", type=str, choices=["single", "double"], help="Machine precision"
)
parser.add_argument(
    "--double_neg", action="store_true",
    help="Whether to negative sample both head and tail entities"
)
parser.add_argument(
    "--multi_c", action="store_true", help="Multiple curvatures per relation"
)


def predict(args):
    save_dir = args.save_dir
    
    # create dataset
    dataset_path = os.path.join(os.environ["DATA_PATH"], args.dataset)
    dataset = KGDataset(dataset_path, False)
    train_examples = dataset.get_examples("train")
    filters = dataset.get_filters()
    args.sizes = dataset.get_shape()

    # create model
    model = getattr(models, args.model)(args)
    logging.info("\t Start inference")
    model.load_state_dict(torch.load(os.path.join(save_dir, "model.pt")))
    model.eval()
    model.to('cuda:0')
    # print(train_examples[:4,])
    # print(model.get_queries(train_examples[:4,].to('cuda:0')[0]))
    model.compute_metrics(train_examples[:4,].to('cuda:0'), filters)
    # print(model.entity(train_examples))

if __name__ == "__main__":
    predict(parser.parse_args())