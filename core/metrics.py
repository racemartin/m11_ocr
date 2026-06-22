# metrics.py
# Statistiques et metriques. Reutilisable Ex.1-3 + Eagle-1

import numpy as np


def compute_stats(rewards, steps=None):
    # Statistiques descriptives d une liste de recompenses
    rewards = np.array(rewards)
    stats   = {
        'n_episodes' : len(rewards),
        'mean'       : float(np.mean(rewards)),
        'std'        : float(np.std(rewards)),
        'min'        : float(np.min(rewards)),
        'max'        : float(np.max(rewards)),
        'median'     : float(np.median(rewards)),
        'q25'        : float(np.percentile(rewards, 25)),
        'q75'        : float(np.percentile(rewards, 75)),
    }
    if steps is not None:
        steps               = np.array(steps)
        stats['mean_steps'] = float(np.mean(steps))
        stats['std_steps']  = float(np.std(steps))
        stats['max_steps']  = float(np.max(steps))
    return stats


def print_stats(stats, label='Statistiques'):
    # Rapport formate des statistiques d episodes
    print(f'\n{"=" * 50}')
    print(f'  {label}')
    print(f'{"=" * 50}')
    print(f'  Episodes............: {stats["n_episodes"]}')
    print(f'  Recompense moyenne..: {stats["mean"]:7.2f}  +-  {stats["std"]:.2f}')
    print(f'  Recompense min/max..: {stats["min"]:7.2f}  /   {stats["max"]:.2f}')
    print(f'  Mediane.............: {stats["median"]:7.2f}')
    print(f'  Q25 / Q75...........: {stats["q25"]:7.2f}  /   {stats["q75"]:.2f}')
    if 'mean_steps' in stats:
        print(f'  Pas moyens..........: {stats["mean_steps"]:7.1f}  +-  {stats["std_steps"]:.1f}')
        print(f'  Pas maximum.........: {stats["max_steps"]:7.0f}')
    print(f'{"=" * 50}')


def moving_average(values, window=50):
    # Moyenne glissante pour lisser les courbes de recompense
    values = np.array(values)
    return np.convolve(values, np.ones(window) / window, mode='valid')


def success_rate(rewards, threshold=1.0):
    # Taux d episodes au-dessus d un seuil
    # FrozenLake : threshold=1.0 / Eagle-1 : threshold=200
    rewards = np.array(rewards)
    return float(np.mean(rewards >= threshold))
