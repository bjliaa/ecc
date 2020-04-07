import argparse
from eval import lunarevaluator

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Eval Lunar')
    parser.add_argument('-id',
                        dest='wid',
                        type=int,
                        help="Designated worker id.")

    args = parser.parse_args()
    lunarevaluator(wid=args.wid)
