from argparse import ArgumentParser
from models.trainer import *

print(torch.cuda.is_available())

"""
the main function for training the CD networks
"""

import random
def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
set_random_seed(42)


def train(args):
    dataloaders = utils.get_loaders(args)
    model = CDTrainer(args=args, dataloaders=dataloaders)
    model.train_models()


def test(args):
    from models.evaluator import CDEvaluator
    dataloader = utils.get_loader(args.data_name, img_size=args.img_size,
                                  batch_size=args.batch_size, is_train=False,
                                  split='test')
    model = CDEvaluator(args=args, dataloader=dataloader)

    model.eval_models()


if __name__ == '__main__':
    # ------------
    # args
    # ------------
    parser = ArgumentParser()
    parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
    parser.add_argument('--project_name', default='MEIT_LEVIR', type=str)
    parser.add_argument('--checkpoint_root', default='checkpoints', type=str)

    # data
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--dataset', default='CDDataset', type=str)
    parser.add_argument('--data_name', default='train', type=str)

    parser.add_argument('--batch_size', default=4, type=int)
    parser.add_argument('--split', default="train", type=str)
    parser.add_argument('--split_val', default="val", type=str)

    parser.add_argument('--img_size', default=512, type=int)

    # model
    parser.add_argument('--n_class', default=2, type=int)
    parser.add_argument('--net_G', default='MEIT', type=str,
                        help='base_resnet18 | base_transformer_pos_s4 | '
                             'base_transformer_pos_s4_dd8 | '
                             'base_transformer_pos_s4_dd8_dedim8|')
    parser.add_argument('--loss', default='ce', type=str)

    # optimizer
    parser.add_argument('--optimizer', default='SGD', type=str)
    parser.add_argument('--lr', default=0.01, type=float)
    parser.add_argument('--max_epochs', default=1, type=int)
    parser.add_argument('--lr_policy', default='linear', type=str,
                        help='linear | step')
    parser.add_argument('--lr_decay_iters', default=100, type=int)

    args = parser.parse_args()
    utils.get_device(args)
    print(args.gpu_ids)

    #  checkpoints dir
    args.checkpoint_dir = os.path.join(args.checkpoint_root, args.project_name)
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    #  visualize dir
    args.vis_dir = os.path.join('vis', args.project_name)
    os.makedirs(args.vis_dir, exist_ok=True)

    train(args)

    # test(args)
