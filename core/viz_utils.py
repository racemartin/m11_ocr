# =============================================================================
# core/viz_utils.py
# Port de sortie : utilitaires de visualisation.
# Reutilisable dans tous les exercices et dans la mission Eagle-1.
# =============================================================================

import numpy             as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.gridspec as gridspec
import matplotlib.patches  as mpatches

# from matplotlib.colors import Normalize
# from matplotlib.cm     import ScalarMappable, get_cmap

from matplotlib.cm import ScalarMappable

from matplotlib.colors import Normalize 
from matplotlib.cm import ScalarMappable
import matplotlib as _mpl
def get_cmap(name):
    if hasattr(_mpl, 'colormaps'):
        return _mpl.colormaps[name]
    return _mpl.cm.get_cmap(name)
    
from   IPython.display   import HTML
from   collections       import deque


# =============================================================================
# GRAPHIQUES STATIQUES
# =============================================================================

def plot_episode_metrics(rewards, steps, title="Politique Aleatoire"):
    """
    Deux graphiques cote a cote :
      - Recompense totale par episode (avec moyenne glissante)
      - Nombre de pas par episode

    Parametres
    ----------
    rewards : list[float]
    steps   : list[int]
    title   : str
    """
    episodes = list(range(1, len(rewards) + 1))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle(title, fontsize=14, fontweight='bold')

    ax1 = axes[0]
    ax1.plot(episodes, rewards,
             color='steelblue', alpha=0.6, linewidth=1.5,
             label='Recompense par episode')
    window = min(10, len(rewards))
    if len(rewards) >= window:
        rolling = np.convolve(rewards, np.ones(window)/window, mode='valid')
        ax1.plot(range(window, len(rewards)+1), rolling,
                 color='darkblue', linewidth=2,
                 label=f'Moyenne glissante ({window} ep.)')
    ax1.axhline(0, color='gray', linestyle='--', linewidth=0.8)
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Recompense totale')
    ax1.set_title('Recompense par episode')
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    ax2 = axes[1]
    ax2.plot(episodes, steps,
             color='darkorange', alpha=0.6, linewidth=1.5,
             label='Pas par episode')
    if len(steps) >= window:
        rolling_steps = np.convolve(steps, np.ones(window)/window, mode='valid')
        ax2.plot(range(window, len(steps)+1), rolling_steps,
                 color='saddlebrown', linewidth=2,
                 label=f'Moyenne glissante ({window} ep.)')
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Nombre de pas')
    ax2.set_title('Duree des episodes (pas)')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


# -----------------------------------------------------------------------------

def plot_reward_distribution(rewards, env_id=""):
    """
    Histogramme de distribution des recompenses.

    Parametres
    ----------
    rewards : list[float]
    env_id  : str
    """
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(rewards, bins=20, color='steelblue',
            edgecolor='white', alpha=0.8)
    ax.axvline(np.mean(rewards), color='red',
               linestyle='--', linewidth=1.5,
               label=f'Moyenne : {np.mean(rewards):.1f}')
    ax.axvline(np.median(rewards), color='orange',
               linestyle='--', linewidth=1.5,
               label=f'Mediane : {np.median(rewards):.1f}')
    ax.set_xlabel('Recompense totale')
    ax.set_ylabel('Frequence')
    ax.set_title(f'Distribution des recompenses -- {env_id}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


# =============================================================================
# AFFICHAGE D'UN FRAME STATIQUE
# =============================================================================

def show_env_frame(env_id, seed=42):
    """
    Affiche un unique frame de l'etat initial de l'environnement.

    Parametres
    ----------
    env_id : str
    seed   : int
    """
    import gymnasium as gym
    env = gym.make(env_id, render_mode="rgb_array")
    env.reset(seed=seed)
    frame = env.render()
    env.close()
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.imshow(frame)
    ax.axis('off')
    ax.set_title(f'Etat initial -- {env_id}', fontsize=12)
    plt.tight_layout()
    plt.show()


# =============================================================================
# ANIMATION : ENVIRONNEMENT + GRAPHIQUES EN TEMPS REEL
# =============================================================================

def animate_episode_with_metrics(env_id, policy_fn=None,
                                   seed=42, max_steps=300,
                                   title="Episode en cours"):
    """
    Anime un episode avec deux panneaux :
      - Gauche : rendu visuel (frames RGB)
      - Droite haut : recompense par pas
      - Droite bas  : nombre de pas ecoules

    Parametres
    ----------
    env_id    : str
    policy_fn : callable(obs) -> action  |  None = aleatoire
    seed      : int
    max_steps : int
    title     : str

    Retourne
    --------
    HTML — animation affichable dans Colab via display(...)
    """
    import gymnasium as gym

    env        = gym.make(env_id, render_mode="rgb_array")
    obs, _     = env.reset(seed=seed)
    frames     = []
    rewards_live = []
    steps_live   = []
    terminated = truncated = False
    step = 0

    while not (terminated or truncated) and step < max_steps:
        frame = env.render()
        frames.append(frame)
        action = policy_fn(obs) if policy_fn is not None \
                 else env.action_space.sample()
        obs, reward, terminated, truncated, _ = env.step(action)
        step += 1
        rewards_live.append(reward)
        steps_live.append(step)

    env.close()

    fig = plt.figure(figsize=(14, 5))
    fig.suptitle(title, fontsize=13, fontweight='bold')
    gs  = gridspec.GridSpec(2, 2, width_ratios=[1, 1],
                             height_ratios=[1, 1], figure=fig)
    ax_env = fig.add_subplot(gs[:, 0])
    ax_r   = fig.add_subplot(gs[0, 1])
    ax_s   = fig.add_subplot(gs[1, 1])

    ax_env.axis('off')
    img     = ax_env.imshow(frames[0])
    line_r, = ax_r.plot([], [], color='steelblue', linewidth=1.5,
                         label='Recompense / pas')
    point_r,= ax_r.plot([], [], 'o', color='steelblue', markersize=5)
    ax_r.set_xlim(0, len(frames)+1)
    ax_r.set_ylim(min(rewards_live)-0.2, max(rewards_live)+0.2)
    ax_r.axhline(0, color='gray', linestyle='--', linewidth=0.8)
    ax_r.set_xlabel('Pas de temps', fontsize=9)
    ax_r.set_ylabel('Recompense', fontsize=9)
    ax_r.set_title('Recompense par pas', fontsize=10)
    ax_r.legend(fontsize=8); ax_r.grid(True, alpha=0.3)

    line_s, = ax_s.plot([], [], color='darkorange', linewidth=1.5,
                         label='Pas ecoules')
    point_s,= ax_s.plot([], [], 'o', color='darkorange', markersize=5)
    ax_s.set_xlim(0, len(frames)+1)
    ax_s.set_ylim(0, len(frames)+5)
    ax_s.set_xlabel('Pas de temps', fontsize=9)
    ax_s.set_ylabel('Pas total', fontsize=9)
    ax_s.set_title('Nombre de pas ecoules', fontsize=10)
    ax_s.legend(fontsize=8); ax_s.grid(True, alpha=0.3)

    txt = ax_env.text(0.02, 0.04, '', transform=ax_env.transAxes,
                      fontsize=9, color='white',
                      bbox=dict(facecolor='black', alpha=0.5, pad=3),
                      verticalalignment='bottom')
    plt.tight_layout()

    def update(i):
        img.set_data(frames[i])
        xs = list(range(1, i+2))
        line_r.set_data(xs, rewards_live[:i+1])
        point_r.set_data([xs[-1]], [rewards_live[i]])
        line_s.set_data(xs, steps_live[:i+1])
        point_s.set_data([xs[-1]], [steps_live[i]])
        txt.set_text(f'Pas : {steps_live[i]}  |  R : {rewards_live[i]:.1f}')
        return img, line_r, point_r, line_s, point_s, txt

    anim = animation.FuncAnimation(fig, update, frames=len(frames), interval=50, blit=True)
    plt.close()
    # return HTML(anim.to_html5_video())
    return anim 


# =============================================================================
# Q-VALUE DIRECTIONAL POLICY MAP  (Exercice 2 — FrozenLake)
# =============================================================================

def plot_q_directional_map(q_table,
                            grid_desc=None,
                            title="Q-value Directional Policy Map",
                            dark=True):
    """
    Visualisation de la Q-table sous forme de grille directionnelle.

    Pour chaque etat, une croix a 4 bras est dessinee :
      - bras nord  = Q(s, UP)
      - bras sud   = Q(s, DOWN)
      - bras ouest = Q(s, LEFT)
      - bras est   = Q(s, RIGHT)
    La couleur de chaque bras va du rouge (valeur basse) au vert
    (valeur haute). Le centre blanc indique l'action optimale (fleche).

    Parametres
    ----------
    q_table   : np.ndarray  shape (n_states, 4)
                Q-table apprise par Q-Learning
    grid_desc : list[str] | None
                Description de la grille, ex. ['SFFF','FHFH','FFFH','HFFG']
                Si None : FrozenLake 4x4 par defaut
    title     : str
                Titre de la figure
    dark      : bool
                True = fond sombre (style notebook),  False = fond clair

    Retourne
    --------
    None — affiche la figure directement

    Exemple
    -------
    plot_q_directional_map(q_table)
    plot_q_directional_map(q_table, title="Apres 5000 episodes")
    """



    # ── Layout de la grille ───────────────────────────────────────
    if grid_desc is None:
        grid_desc = ['SFFF', 'FHFH', 'FFFH', 'HFFG']

    ROWS = len(grid_desc)
    COLS = len(grid_desc[0])
    GRID = list(''.join(grid_desc))   # liste plate des tuiles

    assert len(GRID) == q_table.shape[0], \
        f"grid_desc ({len(GRID)} cases) != q_table ({q_table.shape[0]} etats)"
    assert q_table.shape[1] == 4, \
        "q_table doit avoir 4 colonnes (LEFT, DOWN, RIGHT, UP)"

    # ── Couleurs par type de tuile ────────────────────────────────
    if dark:
        BG       = '#0D0D1A'
        TFC      = {'S':'#9FE1CB','F':'#1C1C30','H':'#3C3489','G':'#FAC775'}
        TEC      = {'S':'#0F6E56','F':'#2E2E50','H':'#26215C','G':'#BA7517'}
        TTC      = {'S':'#085041','F':'#9999BB','H':'#EEEDFE','G':'#412402'}
        SEP_COL  = '#252540'
        LBL_COL  = '#9999BB'
        TTL_COL  = '#DDDDEE'
        SUB_COL  = '#888899'
        BAR_COL  = '#AAAACC'
    else:
        BG       = '#F8F7F5'
        TFC      = {'S':'#9FE1CB','F':'#EEECEA','H':'#534AB7','G':'#FAC775'}
        TEC      = {'S':'#0F6E56','F':'#AAAAAA','H':'#3C3489','G':'#BA7517'}
        TTC      = {'S':'#085041','F':'#444441','H':'#EEEDFE','G':'#412402'}
        SEP_COL  = '#CCCCCC'
        LBL_COL  = '#888888'
        TTL_COL  = '#1A1A1A'
        SUB_COL  = '#666666'
        BAR_COL  = '#666666'

    # ── Colormap Q-values ─────────────────────────────────────────
    q_max = q_table.max()
    norm  = Normalize(vmin=0, vmax=max(q_max, 1e-6))
    cmap  = get_cmap('RdYlGn')
    def qcol(v): return cmap(norm(float(v)))

    # Actions : 0=LEFT 1=DOWN 2=RIGHT 3=UP
    ARR = {0:'←', 1:'↓', 2:'→', 3:'↑'}

    # ── Figure ────────────────────────────────────────────────────
    fig = plt.figure(figsize=(COLS*1.9, ROWS*1.9 + 0.8), dpi=130)
    fig.patch.set_facecolor(BG)
    ax = fig.add_axes([0.08, 0.10, 0.84, 0.80])
    ax.set_facecolor(BG)
    ax.set_xlim(0, COLS)
    ax.set_ylim(0, ROWS)
    ax.set_aspect('equal')
    ax.axis('off')

    ARM = 0.28   # longueur demi-bras
    THK = 0.24   # epaisseur bras

    for state in range(len(GRID)):
        row  = state // COLS
        col  = state  % COLS
        tile = GRID[state]
        ox   = col + 0.03
        oy   = (ROWS-1-row) + 0.03
        sz   = 0.94
        cx   = ox + sz/2
        cy   = oy + sz/2
        hw   = ARM/2
        ht   = THK/2

        # Fond de la case
        ax.add_patch(mpatches.FancyBboxPatch(
            (ox, oy), sz, sz,
            boxstyle='round,pad=0.03',
            facecolor=TFC[tile], edgecolor=TEC[tile],
            lw=1.0, zorder=1))

        # Cases terminales : pas de croix
        if tile in ('H', 'G'):
            lbl = '✕' if tile == 'H' else '★'
            ax.text(cx, cy, lbl, ha='center', va='center',
                    fontsize=24, color=TTC[tile],
                    fontweight='bold', zorder=4)
            ax.text(ox+0.07, oy+0.07, str(state),
                    ha='left', va='bottom',
                    fontsize=7.5, color=TTC[tile], zorder=4)
            continue

        ql, qd, qr, qu = q_table[state]   # LEFT DOWN RIGHT UP

        # ── Dessiner les 4 bras de la croix ───────────────────
        def bras(x, y, w, h, v):
            ax.add_patch(mpatches.FancyBboxPatch(
                (x, y), w, h,
                boxstyle='round,pad=0.015',
                facecolor=qcol(v), edgecolor='none', zorder=2))

        bras(cx-ht,    cy,        THK,    hw+ht, qu)  # UP   (nord)
        bras(cx-ht,    cy-hw-ht,  THK,    hw+ht, qd)  # DOWN (sud)
        bras(cx-hw-ht, cy-ht,     hw+ht,  THK,   ql)  # LEFT (ouest)
        bras(cx,       cy-ht,     hw+ht,  THK,   qr)  # RIGHT (est)

        # Centre blanc + fleche action optimale
        best = np.argmax([ql, qd, qr, qu])
        ax.add_patch(mpatches.FancyBboxPatch(
            (cx-ht, cy-ht), THK, THK,
            boxstyle='round,pad=0.01',
            facecolor='#FFFFFF', edgecolor='none',
            alpha=0.90, zorder=3))
        ax.text(cx, cy, ARR[best],
                ha='center', va='center',
                fontsize=11, color='#111111',
                fontweight='bold', zorder=5)

        # Valeurs Q sur chaque bras
        # Les coefficients eloignent les labels du centre pour eviter
        # qu'ils se superposent avec la fleche et entre eux
        fs  = 8.5
        cv  = '#EEEEEE' if dark else '#222222'
        off = 0.19    # decalage supplementaire depuis le centre du bras
        ax.text(cx,          cy + hw + off,   f'{qu:.3f}',
                ha='center', va='center', fontsize=fs, color=cv, zorder=5)
        ax.text(cx,          cy - hw - off,   f'{qd:.3f}',
                ha='center', va='center', fontsize=fs, color=cv, zorder=5)
        ax.text(cx - hw - off, cy,            f'{ql:.3f}',
                ha='center', va='center', fontsize=fs, color=cv, zorder=5)
        ax.text(cx + hw + off, cy,            f'{qr:.3f}',
                ha='center', va='center', fontsize=fs, color=cv, zorder=5)

        # Numero d'etat et type de tuile
        ax.text(ox+0.07, oy+0.07, str(state),
                ha='left', va='bottom',
                fontsize=7.5, color='#7777AA' if dark else '#888888',
                zorder=5)
        ax.text(ox+sz-0.07, oy+0.07, tile,
                ha='right', va='bottom',
                fontsize=7.5, color=TTC[tile],
                fontweight='bold', zorder=5)

    # ── Separateurs de grille ─────────────────────────────────────
    for i in range(COLS+1):
        ax.plot([i, i], [0, ROWS], color=SEP_COL, lw=1.0, zorder=0)
    for i in range(ROWS+1):
        ax.plot([0, COLS], [i, i], color=SEP_COL, lw=1.0, zorder=0)

    # ── Labels col / row ──────────────────────────────────────────
    for c in range(COLS):
        ax.text(c+0.5, ROWS+0.08, str(c),
                ha='center', va='bottom', fontsize=11, color=LBL_COL)
    for r in range(ROWS):
        ax.text(-0.10, (ROWS-1-r)+0.5, str(r),
                ha='right', va='center', fontsize=11, color=LBL_COL)
    ax.text(COLS/2, ROWS+0.25, 'col',
            ha='center', va='bottom', fontsize=10, color=SUB_COL)
    ax.text(-0.30, ROWS/2, 'row',
            ha='center', va='center', fontsize=10,
            color=SUB_COL, rotation=90)

    # ── Titre ─────────────────────────────────────────────────────
    fig.text(0.50, 0.975, title,
             ha='center', va='top',
             fontsize=13, fontweight='bold', color=TTL_COL)
    fig.text(0.50, 0.955,
             'Chaque bras = Q(s, action)  '
             '. vert=haute valeur  '
             '. rouge=basse valeur  '
             '. fleche=action optimale',
             ha='center', va='top',
             fontsize=8.5, color=SUB_COL)

    # ── Colorbar ──────────────────────────────────────────────────
    cbar_ax = fig.add_axes([0.08, 0.030, 0.84, 0.022])
    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cb = fig.colorbar(sm, cax=cbar_ax, orientation='horizontal')
    cb.set_label('Valeur Q(s,a)', color=BAR_COL, fontsize=9, labelpad=3)
    cb.ax.xaxis.set_tick_params(color=BAR_COL)
    plt.setp(cb.ax.xaxis.get_ticklabels(), color=BAR_COL, fontsize=8)

    # ── Legende tiles ─────────────────────────────────────────────
    leg = [
        mpatches.Patch(facecolor=TFC['S'], edgecolor=TEC['S'], label='S  Start'),
        mpatches.Patch(facecolor=TFC['F'], edgecolor=TEC['F'], label='F  Frozen'),
        mpatches.Patch(facecolor=TFC['H'], edgecolor=TEC['H'], label='H  Hole  ✕'),
        mpatches.Patch(facecolor=TFC['G'], edgecolor=TEC['G'], label='G  Goal  ★'),
    ]
    fig.legend(handles=leg, loc='lower center',
               bbox_to_anchor=(0.50, 0.055), ncol=4,
               fontsize=9,
               facecolor='#1A1A2E' if dark else '#EEEEEE',
               edgecolor='#333355' if dark else '#CCCCCC',
               labelcolor=TTL_COL)

    # plt.show()
    return fig
