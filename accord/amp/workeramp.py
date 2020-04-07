def worker(wid, env_str, N, vsize, sq, aq):
    import tensorflow as tf
    import numpy as np
    import gym
    import time

    from accord.agents.ampdist import AmpDistAgent
    from accord.misc.atariwrappers import make_atari_train
    from accord.misc.util import Linear, rclip, modelsave
    from accord.memory.memory import ReplayBuffer
    from accord.amp.hyperparameters import paramdict

    gpus = tf.config.experimental.get_visible_devices("GPU")

    # Select gpu depending on wid
    total_gpus = 1
    gpu_nr = wid % total_gpus
    tf.config.set_visible_devices(gpus[gpu_nr], 'GPU')

    # Restricts mem to allow multiple tf sessions on one GPU
    tf.config.experimental.set_memory_growth(gpus[gpu_nr], True)

    # Setup environment
    env = make_atari_train(env_str)
    action_len = env.action_space.n

    # Hyperparameter dict
    params = paramdict()

    # Memory
    mem = ReplayBuffer(size=int(params["mem size"]),
                       batchsize=params["batch size"])

    # Train parameters
    prefill_end = int(params["prefill size"])
    eps = Linear(params["start epsilon"], params["final epsilon"],
                 params["explore steps"])
    gamma = params["gamma"]
    updatefreq = params["update freq"]
    targetfreq = params["target update freq"]

    # Eval and logging parameters
    printfreq = params["print freq"]
    savefreq = params["save freq"]

    # Prefill
    print(f"Worker{wid} collecting history...")
    state = env.reset()
    buff = []
    for t in range(1, prefill_end + 1):
        action = env.action_space.sample()
        endstate, rew, done, _ = env.step(action)
        data = (state, action, rclip(rew), gamma, endstate, float(done))
        buff.append(data)
        if done:
            state = env.reset()
        else:
            state = endstate
        if t > 0 and t % 10000 == 0:
            print(f"Worker{wid} collected {t} samples.")

    print(f"Worker{wid} storing history...")
    for data in buff:
        mem.add(data)

    # Warm up
    states, _, _, _, _, _, = mem.sample()
    agent = AmpDistAgent(action_len, vsize=vsize, dense=int(512))
    agent.probvalues(states)
    agent.t_probvalues(states)
    agent.s_probvalues(states)
    agent.update_self()

    # Initial dispatch
    # sq.put(agent.probnet.trainable_variables)
    sq.put(agent.probnet.get_weights())
    while True:
        if not aq.empty():
            wlst = aq.get()
            agent.update_target(wlst)
            break
        time.sleep(0.01)
    print(f"Worker{wid} initial dispatch.")
    tottime = time.time()
    dispatchtime = tottime

    # Training loop
    print(f"Worker{wid} learning...")
    state = env.reset()
    episode_rewards = [0.0]
    buff = []
    for t in range(1, N + 1):
        t_eps = tf.constant(eps(t), dtype=tf.float32)
        action = agent.eps_greedy_action(
            state=np.reshape(state, [1, 84, 84, 4]).astype(np.float32),
            epsval=t_eps,
        )[0].numpy()
        endstate, rew, done, info = env.step(action)
        data = (state, action, rclip(rew), gamma, endstate, float(done))
        buff.append(data)

        if info["Game Over"]:
            episode_rewards.append(info["Episode Score"])
        if done:
            state = env.reset()
        else:
            state = endstate

        if t % updatefreq == 0:
            for data in buff:
                mem.add(data)
            buff = []
            (states, actions, drews, gexps, endstates, dones) = mem.sample()

        if t % targetfreq == 0:
            sq.put(agent.probnet.get_weights())
            while True:
                if not aq.empty():
                    wlst = aq.get()
                    agent.update_target(wlst)
                    break
                time.sleep(0.01)

        if t % printfreq == 0:
            tmptime = time.time()
            msit = (tmptime - dispatchtime) / printfreq * 1000
            ma10 = np.mean(episode_rewards[-11:-1])
            dispatchtime = tmptime
            print(
                f"Worker{wid}, Step: {t}, MA10: {ma10:6.2f}, Speed: {msit:4.2f} ms/it"
            )

        if t % savefreq == 0:
            modelsave(agent, env_str + "Amp", t, wid)

    env.close()
    tmptime = time.time()
    tottime = tmptime - tottime
    msit = tottime / N * 1000
    print(
        f"Worker{wid}: Learning done in {tottime:6.0f}s using {msit:4.2f} ms/it."
    )
