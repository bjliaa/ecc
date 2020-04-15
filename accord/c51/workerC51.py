def worker(wid, env_str, k_steps, resumeflag=False):
    import tensorflow as tf
    import numpy as np
    import gym
    import time
    import pickle

    from accord.misc.atariwrappers import make_atari_train
    from accord.agents.distributional import DistAgent
    from accord.memory.memory import ReplayBuffer
    from accord.misc.util import Linear, rclip, datasave, modelsave
    from accord.c51.hyperparameters import paramdict

    gpus = tf.config.experimental.get_visible_devices("GPU")

    # Select single gpu depending on wid
    total_gpus = 2
    gpu_nr = wid % total_gpus
    tf.config.set_visible_devices(gpus[gpu_nr], "GPU")

    # Restricts mem to allow multiple tf sessions on one GPU
    tf.config.experimental.set_memory_growth(gpus[gpu_nr], True)

    # Setup environment
    env = make_atari_train(env_str)
    action_len = env.action_space.n
    agent = DistAgent(action_len)
    chkpt_dir = f"chkpoints/chkpoints-C51-{wid}"

    # Hyperparameter dict
    params = paramdict()

    # Memory
    if resumeflag:
        t0 = pickle.load(open(chkpt_dir + "/currstep.pkl", "rb"))
        tf.print(
            f"Resuming training on {env_str} at step {t0} for worker {wid}."
        )
        tf.print(f"Loading memory...")
        mem = pickle.load(open(chkpt_dir + "/mem.pkl", "rb"))
        tf.print(f"Done.")
    else:
        mem = ReplayBuffer(
            size=params["mem size"], batchsize=params["batch size"]
        )
        t0 = 0

    # Train parameters
    N = int(k_steps * 1e3)
    eps = Linear(
        params["start epsilon"],
        params["final epsilon"],
        params["explore steps"],
    )
    gamma = params["gamma"]
    updatefreq = params["update freq"]
    targetfreq = params["target update freq"]

    # Eval and logging parameters
    printfreq = params["print freq"]
    savefreq = params["save freq"]
    checkpoint = tf.train.Checkpoint(agent=agent)
    manager = tf.train.CheckpointManager(
        checkpoint, directory=chkpt_dir, max_to_keep=1
    )
    if resumeflag:
        checkpoint.restore(manager.latest_checkpoint)

    if not resumeflag:
        # Prefill
        tf.print("Collecting history...")
        prefill_end = params["prefill size"]
        state = env.reset()
        buff = []
        for t in range(prefill_end):
            action = env.action_space.sample()
            endstate, rew, done, _ = env.step(action)
            data = (state, action, rclip(rew), gamma, endstate, float(done))
            buff.append(data)
            if done:
                state = env.reset()
            else:
                state = endstate
            if t > 0 and t % 10000 == 0:
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
    dispatchtime = tottime

    # Training loop (8-9 ms/step Björn@home, 7.3ms/step HPCC)
    tf.print(f"Worker {wid} learning...")
    state = env.reset()
    episode_rewards = [0.0]
    buff = []
    for t in range(t0 + 1, t0 + N + 1):
        t_eps = tf.constant(eps(t), dtype=tf.float32)
        action = agent.eps_greedy_action(
            state=np.reshape(state, [1, 84, 84, 4]).astype(np.float32),
            epsval=t_eps,
        )[0].numpy()
        endstate, rew, done, info = env.step(action)
        data = (state, action, rclip(rew), gamma, endstate, float(done))
        buff.append(data)
        # env.render()
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
            agent.train(states, actions, drews, gexps, endstates, dones)

        if t % targetfreq == 0:
            agent.update_target()

        if t % printfreq == 0:
            tmptime = time.time()
            msit = (tmptime - dispatchtime) / printfreq * 1000
            ma100 = np.nan
            h100 = np.nan
            ma10 = np.nan
            h10 = np.nan
            if len(episode_rewards) >= 10:
                ma10 = np.mean(episode_rewards[-10:])
                h10 = np.max(episode_rewards[-10:])
            if len(episode_rewards) >= 100:
                ma100 = np.mean(episode_rewards[-100:])
                h100 = np.max(episode_rewards[-100:])
            dispatchtime = tmptime
            tf.print(f"Step: {t}, " + f"MA100: {ma100:6.2f}, " +
                     f"H100: {h100:4.1f}, " + f"MA10: {ma10:6.2f}, " +
                     f"H10: {h10:4.1f}, " + f"Speed: {msit:4.2f} ms/sample")

        if t % savefreq == 0:
            modelsave(agent, env_str + "C51", t, wid)

    env.close()
    tmptime = time.time()
    tottime = tmptime - tottime
    msit = tottime / N * 1000
    tf.print(f"Learning done in {tottime:6.0f}s using {msit:4.2f} ms/it.")
    tf.print(f"Saving checkpoint for worker {wid}...")
    # manager.save()
    # t0 = t0 + N
    # pickle.dump(t0,
    #             open(chkpt_dir + "/currstep.pkl", "wb"),
    #             protocol=pickle.HIGHEST_PROTOCOL)
    # pickle.dump(mem,
    #             open(chkpt_dir + "/mem.pkl", "wb"),
    #             protocol=pickle.HIGHEST_PROTOCOL)
    tf.print("Done.")
