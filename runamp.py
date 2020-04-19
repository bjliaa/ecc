import multiprocessing as mp
import time
import argparse

from accord.amp.workeramp import worker, dispatcher
from accord.amp.hyperparameters import paramdict

if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='Run Amp for K-steps without resume')
    parser.add_argument('-env', dest='env_str', help="Environment string.")
    parser.add_argument('-k',
                        dest='k',
                        type=int,
                        help="Training steps in thousands.")
    parser.add_argument(
        '-ids',
        dest='wids',
        type=int,
        nargs=argparse.REMAINDER,
        help="Designated model ids to use for the amplified policy.")

    args = parser.parse_args()

    params = paramdict()

    # Environment variables
    env_str = args.env_str
    wids = args.wids
    vsize = len(wids)
    N = args.k * 1000
    targetfreq = params["target update freq"]

    # Multiprocessing
    sqs = [mp.Queue() for wid in wids]
    aqs = [mp.Queue() for wid in wids]
    workers = []
    for i in range(vsize):
        args = [wids[i], env_str, N, vsize, sqs[i], aqs[i]]
        workers.append(mp.Process(target=worker, args=args))
        workers[i].start()

    disp = mp.Process(target=dispatcher, args=(sqs,aqs,N,vsize, targetfreq))
    disp.start()

    for i in range(vsize):
        workers[i].join()
    disp.join()