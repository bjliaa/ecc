from gym.envs.registration import registry, register, make, spec

for game in [
        'adventure', 'air_raid', 'alien', 'amidar', 'assault', 'asterix',
        'asteroids', 'atlantis', 'bank_heist', 'battle_zone', 'beam_rider',
        'berzerk', 'bowling', 'boxing', 'breakout', 'carnival', 'centipede',
        'chopper_command', 'crazy_climber', 'defender', 'demon_attack',
        'double_dunk', 'elevator_action', 'enduro', 'fishing_derby', 'freeway',
        'frostbite', 'gopher', 'gravitar', 'hero', 'ice_hockey', 'jamesbond',
        'journey_escape', 'kangaroo', 'krull', 'kung_fu_master',
        'montezuma_revenge', 'ms_pacman', 'name_this_game', 'phoenix',
        'pitfall', 'pong', 'pooyan', 'private_eye', 'qbert', 'riverraid',
        'road_runner', 'robotank', 'seaquest', 'skiing', 'solaris',
        'space_invaders', 'star_gunner', 'tennis', 'time_pilot', 'tutankham',
        'up_n_down', 'venture', 'video_pinball', 'wizard_of_wor',
        'yars_revenge', 'zaxxon'
]:
    for obs_type in ['image', 'ram']:
        # space_invaders should yield SpaceInvaders-v0 and SpaceInvaders-ram-v0
        name = ''.join([g.capitalize() for g in game.split('_')])
        if obs_type == 'ram':
            name = '{}-ram'.format(name)

        nondeterministic = False
        if game == 'elevator_action' and obs_type == 'ram':
            # ElevatorAction-ram-v0 seems to yield slightly
            # non-deterministic observations about 10% of the time. We
            # should track this down eventually, but for now we just
            # mark it as nondeterministic.
            nondeterministic = True

        register(
            id='{}NoFrameskip-v5'.format(name),
            entry_point='gym.envs.atari:AtariEnv',
            kwargs={
                'game': game,
                'obs_type': obs_type,
                'frameskip': 1
            },
            max_episode_steps=108000,
            nondeterministic=nondeterministic,
        )
