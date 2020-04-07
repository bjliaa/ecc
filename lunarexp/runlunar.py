import argparse
from train import lunarworker

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run Lunar')
    parser.add_argument('-id',
                        dest='wid',
                        type=int,
                        help="Designated worker id.")

    args = parser.parse_args()
    lunarworker(wid=args.wid)
