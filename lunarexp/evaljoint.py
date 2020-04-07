import argparse
from joint import lunarjointevaluator

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Eval Joint Lunar')
    parser.add_argument(
        '-ids',
        dest='wids',
        type=int,
        nargs=argparse.REMAINDER,
        help="Designated model ids to use for the joint policy.")

    args = parser.parse_args()
    lunarjointevaluator(wids=args.wids)
