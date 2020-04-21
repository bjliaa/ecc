def evaluator(wid, env_str, start, end):
    import tensorflow as tf
    import numpy as np
    import gym
    import time
    import glob
    import os
    import pandas as pd

    from accord.misc.atariwrappers import make_atari_eval
    from accord.agents.distributional import DistAgent
    from accord.memory.memory import ReplayBuffer
    from accord.amp.hyperparameters import paramdict

    gpus = tf.config.experimental.get_visible_devices("GPU")
    # Select single gpu depending on wid
    total_gpus = 2
    gpu_nr = wid % total_gpus
    tf.config.set_visible_devices(gpus[gpu_nr], 'GPU')

    # Restricts mem to allow multiple tf sessions on one GPU
    tf.config.experimental.set_memory_growth(gpus[gpu_nr], True)

    # Setup environment
    env = make_atari_eval(env_str)
    action_len = env.action_space.n
    agent = DistAgent(action_len)

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
        names = glob.glob(f"models/{env_str}Amp/step{step}/model-id-{wid}.h5")
        print(names)
        filenames.append(names[0])

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
    agent.probvalues(states)

    tottime = time.time()
    dispatchtime = tottime

    # Eval loop (5 ms/step Björn@home)
    tf.print(f"Amp Model id: {wid}")
    t_eps = tf.constant(evaleps, dtype=tf.float32)
    off = max(start, 1)
    data = pd.DataFrame()
    for pt in range(start, end + 1):
        if pt > 0:
            tf.print(f"Loading {filenames[pt-off]}.")
            agent.load(filenames[pt - off])
        tf.print(f"Evaluating@{pt*savefreq}")
        state = env.reset()
        evalrews = []
        for t in range(1, evalsteps + 1):
            action = agent.eps_greedy_action(
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

    tf.print(f"Saving scores for model {wid}...")

    data.reset_index(drop=True, inplace=True)
    data["Alg"] = "Amp Single"
    data["Model"] = wid

    env_dir = env_str + "AmpSingle"
    dir_str = f"data/{env_dir}/steps{start}-{end}/"
    os.makedirs(dir_str, exist_ok=True)
    file_str = dir_str + "model-id-" + f"{wid}" + ".csv"

    data.to_csv(file_str)

    tf.print("Done.")
