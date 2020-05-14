import argparse


def args_for_train_tl(argv=None):
    """
    get all the input arguemnts from commandline
    :param argv:
    :return: args that have all the arguments
    """
    parser = argparse.ArgumentParser(description='PyTorch ')

    # Datasets
    parser.add_argument('-d', '--dataset', default='cvpr_birds', type=str,
                        help='name of dataset {cvpr_birds}')
    parser.add_argument('-j', '--workers', default=2, type=int, metavar='N',
                        help='number of data loading workers')
    # Architecture
    parser.add_argument('--model', default='resnet', type=str,
                        help='model type {resnet, vgg}')
    parser.add_argument('--depth', default=18, type=int,
                        help='model depth')
    parser.add_argument('--norm', default='bn', type=str,
                        help='normalization type; rbn: retrained bn {bn, rbn}')
    parser.add_argument('--basicblock', action='store_true', default=False,
                        help='force to use basicblock')

    # Transfer Learning
    parser.add_argument('--pre_trained', default='True', type=str,
                        help='model type {True, False}')
    parser.add_argument('--retrain_classifier_only', default='True', type=str,
                        help='model type {True, False}')

    # Optimization options
    parser.add_argument('--epochs', default=100, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('--train-batch', default=128, type=int, metavar='N',
                        help='train batchsize')
    parser.add_argument('--test-batch', default=128, type=int, metavar='N',
                        help='test batchsize')
    parser.add_argument('--lr', '--learning-rate', default=0.1, type=float, metavar='LR',
                        help='initial learning rate')
    parser.add_argument('--schedule', type=int, nargs='+', default=[30, 60, 90],
                        help='Decrease learning rate at these epochs.')
    parser.add_argument('--gamma', type=float, default=0.1,
                        help='LR is multiplied by gamma on schedule.')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float, metavar='W',
                        help='weight decay')
    parser.add_argument('--bin-lr', default=10.0, type=float, metavar='M',
                        help='lr mutiplier for BIN gates')
    # Checkpoints
    parser.add_argument('-c', '--checkpoint', default='checkpoints', type=str, metavar='PATH',
                        help='path to save checkpoint')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint')
    parser.add_argument('--submission_file', default='results/submission/1.csv', type=str, metavar='PATH',
                        help='path to latest checkpoint')

    # Device options
    parser.add_argument('-g', '--gpu-id', default='0', type=str,
                        help='id(s) for CUDA_VISIBLE_DEVICES')
    args = parser.parse_args()
    return args


def args_for_train_noisy(argv=None):
    """
    get all the input arguemnts from commandline
    :param argv:
    :return: args that have all the arguments
    """
    parser = argparse.ArgumentParser(description='PyTorch ')

    # Datasets
    parser.add_argument('-d', '--dataset', default='cvpr_birds', type=str,
                        help='name of dataset {cvpr_birds}')
    parser.add_argument('-j', '--workers', default=2, type=int, metavar='N',
                        help='number of data loading workers')
    # Architecture
    parser.add_argument('--base-model', default='resnet', type=str,
                        help='model type {resnet, vgg}')

    parser.add_argument('--target-model', default='resnet', type=str,
                        help='model type {resnet, vgg}')
    parser.add_argument('--b-depth', default=18, type=int,
                        help='model depth')
    parser.add_argument('--t-depth', default=32, type=int,
                        help='model depth')
    parser.add_argument('--norm', default='bn', type=str,
                        help='normalization type; rbn: retrained bn {bn, rbn}')
    parser.add_argument('--basicblock', action='store_true', default=False,
                        help='force to use basicblock')

    # Transfer Learning
    parser.add_argument('--pre_trained', default='True', type=str,
                        help='model type {True, False}')
    parser.add_argument('--retrain_classifier_only', default='True', type=str,
                        help='model type {True, False}')

    # Optimization options
    parser.add_argument('--epochs', default=100, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('--train-batch', default=128, type=int, metavar='N',
                        help='train batchsize')
    parser.add_argument('--test-batch', default=128, type=int, metavar='N',
                        help='test batchsize')
    parser.add_argument('--lr', '--learning-rate', default=0.1, type=float, metavar='LR',
                        help='initial learning rate')
    parser.add_argument('--schedule', type=int, nargs='+', default=[30, 60, 90],
                        help='Decrease learning rate at these epochs.')
    parser.add_argument('--gamma', type=float, default=0.1,
                        help='LR is multiplied by gamma on schedule.')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float, metavar='W',
                        help='weight decay')
    parser.add_argument('--bin-lr', default=10.0, type=float, metavar='M',
                        help='lr mutiplier for BIN gates')
    # Checkpoints
    parser.add_argument('-c', '--checkpoint', default='checkpoints', type=str, metavar='PATH',
                        help='path to save checkpoint')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint')
    parser.add_argument('--submission_file', default='results/submission/1.csv', type=str, metavar='PATH',
                        help='path to latest checkpoint')

    # Device options
    parser.add_argument('-g', '--gpu-id', default='0', type=str,
                        help='id(s) for CUDA_VISIBLE_DEVICES')
    args = parser.parse_args()
    return args



