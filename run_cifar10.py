from __future__ import print_function
import argparse
from main import main


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='cifar10', help='Dataset')
parser.add_argument('--network', type=str, default='KCNN', help='Deep network')
parser.add_argument('--nBatch', type=int, default=64, help='Batch size for training')
parser.add_argument('--nEpoch', type=int, default=150, help='Epochs for training')
parser.add_argument('--nValid', type=int, default=5000, help='Number of validation dataset')
parser.add_argument('--nInit', type=int, default=200, help='Number of initial labeled dataset')
parser.add_argument('--nPool', type=int, default=4000, help='Number of pool dataset')
parser.add_argument('--nQuery', type=int, default=400, help='Number of queries at each step')
parser.add_argument('--nStep', type=int, default=24, help='Number of acquisition steps')
parser.add_argument('--nSample', type=int, default=4000, help='Number of samples for approximating rho')

args = parser.parse_args()

nRep = 5
for idx_rep in range(nRep):
    main(args, idx_rep)
