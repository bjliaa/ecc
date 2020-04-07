def lunarjointevaluator(wids):
    import tensorflow as tf
    import numpy as np
    import gym
    import time
    import glob
    import os
    import pandas as pd

    from ensemble import JointEnsemble
    from memory import ReplayBuffer
    from util import RewMonitor, SkipEnv

    gpus = tf.config.experimental.get_visible_devices("GPU")

    # Restricts mem to allow multiple tf sessions on one GPU
    tf.config.experimental.set_memory_growth(gpus[0], True)

    # Eval parameters
    savefreq = 80000
    evaleps = 0.001

    # Weights
    filenames = []
    for step in np.array([i for i in range(0, 100 + 1)]) * savefreq:
        if step == 0:
            continue
        names = []
        for wid in wids:
            names.append(glob.glob(f"lunarmodels/step{step}/model-id-{wid}.h5")[0])
        print(names)
        filenames.append(names)

    ensemblesize = len(wids)

    # Setup
    env = gym.make("LunarLander-v2")
    env = RewMonitor(env)
    env = SkipEnv(env, skip=4)
    action_len = env.action_space.n
    ensemble = JointEnsemble(action_len,
                             size=ensemblesize,
                             dense=16,
                             supportsize=29,
                             vmin=-7.0,
                             vmax=7.0)

    # Warm up
    state = env.reset()
    mem = ReplayBuffer(32, 32)
    for _ in range(32):
        action = env.action_space.sample()
        endstate, rew, done, _ = env.step(action)
        data = (state, action, rew, 0.99, endstate, float(done))
        mem.add(data)
        if done:
            state = env.reset()
        else:
            state = endstate
    states, _, _, _, _, _, = mem.sample()

    for i in range(ensemblesize):
        ensemble.agents[i].probvalues(states)

    tid = time.time()
    # Eval loop
    tf.print(f"Joint ids: {wids}")
    t_eps = tf.constant(evaleps, dtype=tf.float32)
    off = 1
    mrew = []
    its = []
    for pt in range(0, 100 + 1):
        if pt > 0:
            tf.print(f"Loading {filenames[pt-off]}.")
            ensemble.load(filenames[pt - off])
        tf.print(f"Evaluating@{pt*savefreq}")
        state = env.reset()
        evalrews = []

        while len(evalrews) < 100:
            action = ensemble.avgQ_action(
                state=np.reshape(state, [1, 8]).astype(np.float32),
                epsval=t_eps,
            )[0].numpy()
            endstate, _, done, info = env.step(action)
            if info["Game Over"]:
                evalrews.append(info["Episode Score"])
                state = env.reset()
            else:
                state = endstate
        tmptid = (time.time() - tid) / (pt + 1)
        mrew.append(np.mean(evalrews))
        its.append(pt * savefreq)

        print(f"Iteration: {pt*savefreq}, " +
              f"Mean Score: {np.mean(evalrews):7.2f}, " +
              f"AvgTimePt: {tmptid:4.2f}")

    env.close()
    C = {"Iteration": its, "Average Reward": mrew}
    data = pd.DataFrame(C)
    data["Method"] = f"{len(wids)}-Agents"
    data["Model"] = f"{wids}"

    dir_str = f"lunardata/"
    os.makedirs(dir_str, exist_ok=True)
    file_str = dir_str + "model-id-" + f"{wids}" + ".csv"
    data.to_csv(file_str)
    tf.print("Done.")
