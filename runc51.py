import gym
import time
import numpy as np
import argparse

from accord.c51.workerC51 import worker
if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='Run C51 for K-steps with and without resume')
    parser.add_argument('-env', dest='env_str', help="Environment string.")
    parser.add_argument('-id',
                        dest='wid',
                        type=int,
                        help="Designated worker id.")
    parser.add_argument('-k',
                        dest='k',
                        type=int,
                        help="Training steps in thousands.")
    parser.add_argument('-resume',
                        dest='rflag',
                        action='store_true',
                        help="To resume or not to resume")

    args = parser.parse_args()

    env_str = args.env_str
    worker(wid=args.wid,
           env_str=env_str,
           k_steps=args.k,
           resumeflag=args.rflag)
