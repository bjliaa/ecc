# Distribution DQN Ensemble Selection Process
def paramdict():
    d = {
        "explore steps": int(250e3),
        "start epsilon": 1,
        "final epsilon": 0.01,
        "update freq": 4,
        "target update freq": int(10e3),
        "print freq": int(10e3),
        "save freq": int(100e3),
        "eval epsilon": 0.001,
        "eval steps": int(125e3),
        "gamma": 0.99,
        "learning rate": 0.00025,
        "adams epsilon": 3.125e-4,
        "mem size": int(0.9e6),
        "prefill size": int(50e3),
        "gamma": 0.99,
        "batch size": 32,
    }
    return d
