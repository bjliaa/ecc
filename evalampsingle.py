import gym
import time
import numpy as np
import argparse

from accord.amp.evaluatorsingle import evaluator

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Evaluate Single Amp model')
    parser.add_argument('-env', dest='env_str', help="Environment string.")
    parser.add_argument('-id',
                        dest='wid',
                        type=int,
                        help="Designated model id.")
    parser.add_argument('-start',
                        dest='start',
                        type=int,
                        help="Eval start in 100K, e.g., start = 10 for 1M")
    parser.add_argument('-end',
                        dest='end',
                        type=int,
                        help="Eval end in 100K, e.g., end = 10 for 1M")
    args = parser.parse_args()

    env_str = args.env_str
    evaluator(wid=args.wid, env_str=env_str, start=args.start, end=args.end)
