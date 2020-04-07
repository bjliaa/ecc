def evaluator(env_str, start, end, wids):
    import tensorflow as tf
    import numpy as np
    import gym
    import time
    import glob
    import pandas as pd
    import os

    from accord.misc.atariwrappers import make_atari_eval
    from accord.agents.ensembles import AmpJointEnsemble
    from accord.memory.memory import ReplayBuffer
    from accord.amp.hyperparameters import paramdict

    # Hyperparameter dict
    params = paramdict()

    env = make_atari_eval(env_str)
    print(env.spec.id)
    action_len = env.action_space.n

    # Hyperparameter dict
    params = paramdict()

    # Eval parameters
    savefreq = params["save freq"]
    evalsteps = params["eval steps"]  # Rainbow-style
    printfreq = params["print freq"]
    evaleps = params["eval epsilon"]

    # Weights
    filenames = []
    for step in np.array([i for i in range(start, end + 1)]) * savefreq:
        if step == 0:
            continue
        names = []
        for wid in wids:
            names.append(
                glob.glob(f"models/{env_str}Amp/step{step}/model-id-{wid}.h5")
                [0])
        print(names)
        filenames.append(names)

    ensemblesize = len(wids)

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

    ensemble = AmpJointEnsemble(action_len, ensemblesize)
    ensemble.avgQ_action(states, evaleps)

    # Initial dispatch
    tottime = time.time()
    dispatchtime = tottime

    # Eval loop
    tf.print(f"Amp ids: {wids}")
    t_eps = tf.constant(evaleps, dtype=tf.float32)
    off = max(start, 1)
    data = pd.DataFrame()
    for pt in range(start, end + 1):
        if pt > 0:
            print(f"Loading {filenames[pt-off]}.")
            ensemble.load(filenames[pt - off])
        print(f"Evaluating@{pt*savefreq}")
        state = env.reset()
        evalrews = []
        for t in range(1, evalsteps + 1):
            action = ensemble.avgQ_action(
                state=np.reshape(state, [1, 84, 84, 4]).astype(np.float32),
                epsval=t_eps,
            )[0].numpy()
            endstate, _, done, info = env.step(action)
            # env.render()
            if info["Game Over"]:
                evalrews.append(info["Episode Score"])
            if done:
                state = env.reset()
            else:
                state = endstate
            if t % printfreq == 0:
                tmptime = time.time()
                msit = (tmptime - dispatchtime) / printfreq * 1000
                dispatchtime = tmptime
                if len(evalrews) > 0:
                    print(
                        f"Step: {t}, "
                        f"Mean Score: {np.mean(evalrews):7.2f}, "
                        f"Speed: {msit:4.2f} ms/it, EpFrame:{env.frame_count}")

        print(f"Mean Score: {np.mean(evalrews):7.2f}, ",
              f"Speed: {msit:4.2f} ms/it, EpFrame:{env.frame_count}")
        C = {"Frame": 4 * pt * savefreq, "Reward": evalrews}
        df = pd.DataFrame(C)
        data = pd.concat([data, df])

    env.close()

    tf.print(f"Saving scores for amp models {wids}...")

    data.reset_index(drop=True, inplace=True)
    data["Alg"] = "Amp"
    data["Model"] = f"{wids}"

    env_dir = env_str + "Amp"
    dir_str = f"data/{env_dir}/steps{start}-{end}/"
    os.makedirs(dir_str, exist_ok=True)
    file_str = dir_str + "model-id-" + f"{wids}" + ".csv"

    data.to_csv(file_str)

    tf.print("Done.")
