import argparse
from accord.c51.evaluatorjoint import evaluator

if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='Evaluate an AvgJointPolicy model')
    parser.add_argument('-env', dest='env_str', help="Environment string.")
    parser.add_argument('-start',
                        dest='start',
                        type=int,
                        help="Eval start in 100K, e.g., start = 10 for 1M")
    parser.add_argument('-end',
                        dest='end',
                        type=int,
                        help="Eval end in 100K, e.g., end = 10 for 1M")
    parser.add_argument(
        '-ids',
        dest='wids',
        type=int,
        nargs=argparse.REMAINDER,
        help="Designated model ids to use for the joint policy.")
    args = parser.parse_args()

    env_str = args.env_str
    evaluator(env_str=env_str, start=args.start, end=args.end, wids=args.wids)
