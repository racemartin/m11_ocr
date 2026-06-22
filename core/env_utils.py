# env_utils.py
# Utilitaires environnements. Reutilisable Ex.1-3 + Eagle-1

import gymnasium as gym
import numpy     as np


def make_env_V1(env_id, render_mode=None, seed=42):
    # Cree une instance Gymnasium avec graine pour reproductibilite
    env = gym.make(env_id, render_mode=render_mode)
    env.reset(seed=seed)
    return env

def make_env_V2(env_id, render_mode=None, is_slippery=True, seed=42):
    # Cree une instance Gymnasium avec graine pour reproductibilite
    env = gym.make(env_id, render_mode=render_mode, is_slippery=is_slippery)
    env.reset(seed=seed)
    return env

def make_env(env_id, render_mode=None, seed=42, **kwargs):
    env = gym.make(env_id, render_mode=render_mode, **kwargs)
    env.reset(seed=seed)
    return env

def inspect_env(env):
    # Affiche un rapport complet sur les espaces d observation et d action
    print('=' * 60)
    print('RAPPORT D INSPECTION DE L ENVIRONNEMENT')
    print('=' * 60)
    obs_space = env.observation_space
    print('\n-- ESPACE D OBSERVATION -------------------------')
    print(f'  Type................: {type(obs_space).__name__}')
    if isinstance(obs_space, gym.spaces.Box):
        print(f'  Shape...............: {obs_space.shape}')
        print(f'  dtype...............: {obs_space.dtype}')
        print(f'  Borne inferieure....: {obs_space.low}')
        print(f'  Borne superieure....: {obs_space.high}')
    elif isinstance(obs_space, gym.spaces.Discrete):
        print(f'  N etats.............: {obs_space.n}')
    print(f'  Exemple aleatoire...: {obs_space.sample()}')
    
    act_space = env.action_space
    print('\n-- ESPACE D ACTION --------------------------------')
    print(f'  Type................: {type(act_space).__name__}')
    if isinstance(act_space, gym.spaces.Discrete):
        print(f'  N actions...........: {act_space.n}')
        print(f'  Actions dispo.......: {list(range(act_space.n))}')
    elif isinstance(act_space, gym.spaces.Box):
        print(f'  Shape...............: {act_space.shape}')
        print(f'  Borne inferieure....: {act_space.low}')
        print(f'  Borne superieure....: {act_space.high}')
    print(f'  Exemple aleatoire...: {act_space.sample()}')
    print('=' * 60)


def run_random_episode(env):
    # Execute un episode complet avec politique aleatoire
    # Retourne total_reward (float) et n_steps (int)
    obs, info    = env.reset()
    total_reward = 0.0
    n_steps      = 0
    terminated   = False
    truncated    = False
    while not (terminated or truncated):
        # Selectionner une action uniformement aleatoire
        action                              = env.action_space.sample()
        # Appliquer l action (API Gymnasium : 5 valeurs)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        n_steps      += 1
    return total_reward, n_steps


def run_n_random_episodes(env_id, n_episodes=10, seed=42, verbose=True):
    # Execute n episodes aleatoires et collecte les metriques
    env     = make_env(env_id, seed=seed)
    rewards = []
    steps   = []
    if verbose:
        print(f'\nPolitique ALEATOIRE -- {n_episodes} episodes -- {env_id}')
        print('-' * 50)
    for ep in range(n_episodes):
        total_reward, n_steps = run_random_episode(env)
        rewards.append(total_reward)
        steps.append(n_steps)
        if verbose:
            print(f'  Episode {ep+1:2d}.........: recompense = {total_reward:7.1f}'
                  f'  |  pas = {n_steps:4d}')
    env.close()
    if verbose:
        print('-' * 50)
        print(f'  Moyenne recompense..: {np.mean(rewards):.2f}'
              f'  +-  {np.std(rewards):.2f}')
        print(f'  Moyenne pas.........: {np.mean(steps):.1f}')
    return rewards, steps
