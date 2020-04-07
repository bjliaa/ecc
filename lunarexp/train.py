def lunarworker(wid):
    import tensorflow as tf
    import numpy as np
    import gym
    import time
    import os

    from distagent import DistAgent
    from memory import ReplayBuffer
    from util import Linear, scale, RewMonitor, SkipEnv, StackEnv

    gpus = tf.config.experimental.get_visible_devices("GPU")

    # Select single gpu depending on wid
    total_gpus = 2
    gpu_nr = wid % total_gpus
    tf.config.set_visible_devices(gpus[gpu_nr], 'GPU')

    # Restricts mem to allow multiple tf sessions on one GPU
    tf.config.experimental.set_memory_growth(gpus[gpu_nr], True)

    # Train parameters
    N = int(8e6)
    eps = Linear(startval=0.1, endval=0.01, exploresteps=int(200e3))
    gamma = 0.99
    updatefreq = 4
    targetfreq = 1000
    savefreq = 80000

    # Setup
    env = gym.make("LunarLander-v2")
    env = RewMonitor(env)
    env = SkipEnv(env, skip=4)
    # env = StackEnv(env, n_frames=4)
    action_len = env.action_space.n
    agent = DistAgent(action_len,
                      dense=16,
                      supportsize=29,
                      vmin=-7.0,
                      vmax=7.0)
    mem = ReplayBuffer(size=int(20e3), batchsize=32)

    # Prefill
    tf.print("Collecting history...")
    prefill_end = int(10e3)
    state = env.reset()
    buff = []
    for t in range(1, prefill_end + 1):
        action = env.action_space.sample()
        endstate, rew, done, _ = env.step(action)
        data = (state, action, scale(rew), gamma, endstate, float(done))
        buff.append(data)
        if done:
            state = env.reset()
        else:
            state = endstate
        if t % 10000 == 0:
            tf.print(f"Collected {t} samples.")
    tf.print("Done.")

    tf.print("Storing history...")
    for data in buff:
        mem.add(data)
    tf.print("Done.")

    # Warm up
    states, _, _, _, _, _, = mem.sample()
    agent.probvalues(states)
    agent.t_probvalues(states)
    agent.update_target()

    # Initial dispatch
    tottime = time.time()

    # Training loop
    tf.print(f"Worker {wid} learning...")
    state = env.reset()
    episode_rewards = []
    buff = []
    for t in range(1, N + 1):
        t_eps = tf.constant(eps(t), dtype=tf.float32)
        action = agent.eps_greedy_action(
            state=np.reshape(state, [1, 8]).astype(np.float32),
            epsval=t_eps,
        )[0].numpy()
        endstate, rew, done, info = env.step(action)
        data = (state, action, scale(rew), gamma, endstate, float(done))
        buff.append(data)
        if info["Game Over"]:
            score = info["Episode Score"]
            episode_rewards.append(score)
            state = env.reset()
            if len(episode_rewards) % 100 == 0:
                tmptime = time.time()
                msit = (tmptime - tottime) / t * 1000
                ma100 = np.mean(episode_rewards[-111:-1])
                epstr = (f"Epsiode: {len(episode_rewards)}, " +
                         f"Step: {t}, " + f"MA100: {ma100}, " +
                         f"AvgSpeed: {msit:4.2f} ms/it")
                tf.print(epstr)
        else:
            state = endstate

        if t % updatefreq == 0:
            for data in buff:
                mem.add(data)
            buff = []
            (states, actions, drews, gexps, endstates, dones) = mem.sample()
            agent.train(states, actions, drews, gexps, endstates, dones)

        if t % targetfreq == 0:
            agent.update_target()

        if t % savefreq == 0:
            dir_str = f"lunarmodels/step{t}/"
            os.makedirs(dir_str, exist_ok=True)
            file_str = dir_str + "model-id-" + f"{wid}" + ".h5"
            agent.save(file_str)

    env.close()
    tmptime = time.time()
    tottime = tmptime - tottime
    msit = tottime / N * 1000
    tf.print(f"Learning done in {tottime:6.0f}s using {msit:4.2f} ms/it.")
    tf.print("Done.")
