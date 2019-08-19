import argparse
import models as models
import torch.nn as nn


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', action='store_true')
    parser.add_argument('-o', type=str, required=True)
    parser.add_argument('--data_path', type=str, required=True)
    return parser.parse_args()

def get_data_loader():
    return None

def main(args):
    print("learning stuff.")
    trainer = models.Trainer.create_new_trainer(models.TwoLayerNet, 100, nn.MSELoss)

    if args.f:
        trainer.train_epoch(get_data_loader())


if __name__ == '__main__':
    args = get_args()
    main(args)