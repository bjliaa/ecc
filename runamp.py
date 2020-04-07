import multiprocessing as mp
import time
import argparse

from accord.amp.workeramp import worker
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
    updatefreq = params["target update freq"]
    groupfreq = params["group"]

    # Multiprocessing
    sqs = [mp.Queue() for wid in wids]
    aqs = [mp.Queue() for wid in wids]
    workers = []
    for i in range(vsize):
        args = [wids[i], env_str, N, vsize, sqs[i], aqs[i]]
        workers.append(mp.Process(target=worker, args=args))
        workers[i].start()

    # Initial dispatch
    while True:
        if all([not sqs[i].empty() for i in range(vsize)]):
            wlst = []
            for i in range(vsize):
                wlst.append(sqs[i].get())
            for i in range(vsize):
                aqs[i].put(wlst)
            break
        else:
            time.sleep(1)

    # Loop
    group = True
    for t in range(1, N + 1):

        if t % groupfreq == 0:
            group = not group

        if t % updatefreq == 0 and group:
            while True:
                if all([not sqs[i].empty() for i in range(vsize)]):
                    wlst = []
                    for i in range(vsize):
                        wlst.append(sqs[i].get())
                    for i in range(vsize):
                        aqs[i].put(wlst)
                    break
                else:
                    time.sleep(1)
