# ============================================================
# gui.py — Interface graphique Eagle-1 / AstroDynamics
# GUI + Dashboard + Info modèle — Streamlit
# Lancement : streamlit run src/gui.py
# ============================================================

# — Bibliothèques standard
import base64                        # décodage frames base64
import io                            # flux mémoire
import sys                           # path Python
import time                          # délai animation
from pathlib import Path             # chemins portables

# — Interface Streamlit
import streamlit as st               # framework GUI

# — Calcul et visualisation
import numpy             as np       # calcul vectoriel
import pandas            as pd       # tableaux de données
import matplotlib.pyplot as plt      # graphiques
import matplotlib.patches as mpatches  # formes décoratives
from PIL                 import Image  # traitement images

# — Requêtes HTTP vers l'API
import requests                      # client HTTP

# — Outil de journalisation
sys.path.insert(0, str(Path(__file__).resolve().parent))
from tools.rafael.log_tool import LogTool


# ----------------------------------------------------------
# Instance globale du logger GUI
# ----------------------------------------------------------
log = LogTool(origin="gui")

# ============================================================
# Configuration de la page
# ============================================================
st.set_page_config(
    page_title = "Eagle-1 — AstroDynamics",
    page_icon  = "🚀",
    layout     = "wide",
)

API_URL_DEFAUT = "http://localhost:8000"

NOMS_ACTIONS = {
    0: "do nothing",
    1: "moteur gauche",
    2: "moteur principal",
    3: "moteur droit",
}

COULEURS_ACTIONS = {
    0: "#888780",   # gris    — do nothing
    1: "#EF9F27",   # ambre   — moteur gauche
    2: "#E84B3A",   # rouge   — moteur principal
    3: "#534AB7",   # violet  — moteur droit
}

NOMS_OBS = [
    "x   position horiz.",
    "y   position vert. ",
    "vx  vitesse horiz. ",
    "vy  vitesse vert.  ",
    "θ   angle          ",
    "dθ  vit. angulaire ",
    "cG  contact pied G ",
    "cD  contact pied D ",
]

# ============================================================
# Fonctions utilitaires
# ============================================================

def _api_get(endpoint: str, api_url: str) -> dict | None:
    """GET vers l'API — retourne None en cas d'erreur."""
    log.START_CALL_CONTROLLER_FUNCTION("GUI", "_api_get", endpoint)
    try:
        r = requests.get(f"{api_url}{endpoint}", timeout=5)
        r.raise_for_status()
        resultat = r.json()
        log.FINISH_CALL_CONTROLLER_FUNCTION("GUI", "_api_get", "OK")
        return resultat
    except Exception as e:
        log.LEVEL_4_ERROR("GUI", f"GET {endpoint} échoué : {e}")
        log.FINISH_CALL_CONTROLLER_FUNCTION("GUI", "_api_get", "ERREUR")
        return None


def _api_post(endpoint: str, payload: dict, api_url: str) -> dict | None:
    """POST vers l'API — retourne None en cas d'erreur."""
    log.START_CALL_CONTROLLER_FUNCTION("GUI", "_api_post", endpoint)
    log.PARAMETER_VALUE("payload_keys", list(payload.keys()))
    try:
        r = requests.post(
            f"{api_url}{endpoint}",
            json    = payload,
            timeout = 120,
        )
        r.raise_for_status()
        resultat = r.json()
        log.FINISH_CALL_CONTROLLER_FUNCTION("GUI", "_api_post", "OK")
        return resultat
    except Exception as e:
        log.LEVEL_4_ERROR("GUI", f"POST {endpoint} échoué : {e}")
        st.error(f"Erreur API : {e}")
        log.FINISH_CALL_CONTROLLER_FUNCTION("GUI", "_api_post", "ERREUR")
        return None


def _decoder_frame(b64: str) -> Image.Image:
    """Décode une frame base64 PNG en objet PIL Image."""
    return Image.open(io.BytesIO(base64.b64decode(b64)))


def _assembler_video(frames_b64: list) -> bytes | None:
    """Assemble les frames base64 en vidéo MP4 via imageio-ffmpeg."""
    log.START_CALL_CONTROLLER_FUNCTION(
        "GUI", "_assembler_video", f"{len(frames_b64)} frames"
    )
    try:
        import imageio.v3    as iio
        import imageio_ffmpeg
        import tempfile, os

        ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
        frames_np  = [np.array(_decoder_frame(f)) for f in frames_b64]

        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as tmp:
            tmp_path = tmp.name

        writer = imageio_ffmpeg.write_frames(
            tmp_path,
            frames_np[0].shape[:2][::-1],  # (width, height)
            fps        = 30,
            ffmpeg_log_level = "quiet",
        )
        writer.send(None)                   # initialiser le générateur
        for frame in frames_np:
            writer.send(frame)
        writer.close()

        with open(tmp_path, 'rb') as f:
            video_bytes = f.read()
        os.unlink(tmp_path)

        log.PARAMETER_VALUE("taille_mp4", f"{len(video_bytes):,} bytes")
        log.FINISH_CALL_CONTROLLER_FUNCTION(
            "GUI", "_assembler_video", "MP4 généré"
        )
        return video_bytes

    except Exception as e:
        log.LEVEL_5_WARNING("GUI", f"assemblage MP4 échoué : {e}")
        log.FINISH_CALL_CONTROLLER_FUNCTION(
            "GUI", "_assembler_video", "ERREUR"
        )
        return None


# ####################################################################
# Graphiques
# ####################################################################

def _fig_actions(actions: list) -> plt.Figure:
    """Histogramme des actions utilisées dans un épisode."""
    log.START_CALL_CONTROLLER_FUNCTION(
        "GUI", "_fig_actions", f"{len(actions)} actions"
    )
    comptes = [actions.count(i) for i in range(4)]
    total   = len(actions) or 1
    fig, ax = plt.subplots(figsize=(5, 2.5))
    fig.patch.set_facecolor('none')
    ax.set_facecolor('none')

    couleurs = [COULEURS_ACTIONS[i] for i in range(4)]
    bars     = ax.bar(
        [NOMS_ACTIONS[i] for i in range(4)],
        [c / total * 100 for c in comptes],
        color=couleurs, edgecolor='white', linewidth=0.5,
    )
    for bar, c in zip(bars, comptes):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.5,
            f"{c / total:.0%}",
            ha='center', fontsize=9, color='white',
        )
    ax.set_ylabel("% de l'épisode", fontsize=9, color='white')
    ax.tick_params(colors='white', labelsize=8)
    for spine in ax.spines.values():
        spine.set_edgecolor('#444')
    plt.tight_layout()
    log.FINISH_CALL_CONTROLLER_FUNCTION("GUI", "_fig_actions", "OK")
    return fig


def _fig_reward(rewards: list) -> plt.Figure:
    """Courbe de récompense cumulée pas à pas."""
    log.START_CALL_CONTROLLER_FUNCTION(
        "GUI", "_fig_reward", f"{len(rewards)} points"
    )
    fig, ax = plt.subplots(figsize=(5, 2.2))
    fig.patch.set_facecolor('none')
    ax.set_facecolor('none')
    ax.plot(rewards, color='#7F77DD', linewidth=1.5)
    ax.axhline(200, color='#1D9E75', linewidth=1, linestyle='--',
               label='seuil 200')
    ax.axhline(0, color='#888', linewidth=0.5, linestyle=':')
    ax.set_xlabel("pas", fontsize=8, color='white')
    ax.set_ylabel("récompense cumulée", fontsize=8, color='white')
    ax.tick_params(colors='white', labelsize=7)
    ax.legend(fontsize=7, facecolor='#222', labelcolor='white')
    for spine in ax.spines.values():
        spine.set_edgecolor('#444')
    plt.tight_layout()
    log.FINISH_CALL_CONTROLLER_FUNCTION("GUI", "_fig_reward", "OK")
    return fig

def _afficher_telemetrie(act, obs_c, rew_c):
    """Affiche action + récompense + tableau obs."""
    st.markdown(
        f"**Action :** "
        f"<span style='background:#EEEDFE;color:#3C3489;"
        f"padding:2px 10px;border-radius:12px;font-weight:500'>"
        f"{act} — {NOMS_ACTIONS[act]}</span>",
        unsafe_allow_html=True
    )
    st.metric("Récompense cumulée", f"{rew_c:+.2f}")
    st.markdown("**Observation :**")
    st.dataframe(
        pd.DataFrame({
            "Variable": NOMS_OBS,
            "Valeur"  : [f"{v:+.5f}" for v in obs_c],
        }),
        hide_index=True, use_container_width=True, height=240
    )



@st.fragment
def _bloque_simulacion(frames, rewards, actions, obs_all, n_steps, vitesse):
    """Fragment isolé — re-exécuté indépendamment du reste de la page."""
    col_sim, col_tele = st.columns([3, 2])

    with col_sim:
        st.subheader("Simulation")
        idx = st.slider("Pas", 0, n_steps - 1, 0, key="slider_frame_frag")
        st.session_state['frame_idx'] = idx
        placeholder_frame = st.empty()
        placeholder_prog  = st.empty()
        placeholder_frame.image(_decoder_frame(frames[idx]), width='stretch')
        placeholder_prog.progress(
            idx / max(n_steps - 1, 1),
            text=f"Pas {idx + 1} / {n_steps}"
        )
        btn_anim = st.button("▶ Animer l'épisode complet",
                             key="btn_anim_frag")

    with col_tele:
        st.subheader("Télémétrie")
        placeholder_tele = st.empty()

    # Affichage initial
    with placeholder_tele.container():
        _afficher_telemetrie(actions[idx], obs_all[idx], rewards[idx])

    # Animation
    if btn_anim:
        for i, f64 in enumerate(frames):
            placeholder_frame.image(_decoder_frame(f64), width='stretch')
            placeholder_prog.progress(
                (i + 1) / n_steps,
                text=f"Pas {i + 1} / {n_steps}"
            )
            with placeholder_tele.container():
                _afficher_telemetrie(actions[i], obs_all[i], rewards[i])
            time.sleep(1.0 / vitesse)

# ============================================================
# Sidebar — navigation et configuration
# ============================================================
log.START_ACTION("GUI", "rendu", "initialisation sidebar")
with st.sidebar:

    # ── Logo AstroDynamics ────────────────────────────────
    logo_path = Path(__file__).resolve().parent.parent / 'docs' / 'images' / 'astrodynamics_logo.png'
    if logo_path.exists():
        col1, col2, col3 = st.columns([1, 3, 1])
        with col2:
            st.image(str(logo_path), use_container_width=True)

    st.markdown(
        "<div style='text-align:center;font-size:1.2em;font-weight:600'>"
        "🚀 Eagle-1 GUI"
        "</div>",
        unsafe_allow_html=True
    )
    st.markdown("---")

    # ── URL et connexion API ──────────────────────────────
    st.markdown("**Service API**")
    api_url = st.text_input("URL de l'API", value=API_URL_DEFAUT)

    log.START_CALL_CONTROLLER_FUNCTION("GUI", "health_check", api_url)
    health = _api_get("/health", api_url)
    api_ok = health and health.get("status") == "ok"
    if api_ok:
        st.success("● API connectée")
        st.caption(f"{health.get('modele', '—')}")
        log.PARAMETER_VALUE("api_status", "ok")
    else:
        st.error("● API non disponible")
        st.caption("Lancer : uvicorn src.api:app")
        log.LEVEL_5_WARNING("GUI", "API non disponible")
    log.FINISH_CALL_CONTROLLER_FUNCTION("GUI", "health_check", "terminé")

    st.markdown("---")

    # ── Sélecteur de modèle ───────────────────────────────
    # Chargé avant le menu — les pages ne s affichent que si
    # un modele est sélectionné
    modele_selectionne = None

    if api_ok:
        log.START_CALL_CONTROLLER_FUNCTION(
            "GUI", "models_list", "GET /models/list"
        )
        resp_models = _api_get("/models/list", api_url)
        log.FINISH_CALL_CONTROLLER_FUNCTION(
            "GUI", "models_list", "terminé"
        )

        if resp_models and resp_models.get("modeles"):
            liste_modeles = resp_models["modeles"]

            # Construire les labels du selectbox
            def _label(m):
                score = f"+{m['mean_reward']:.1f}" if m['mean_reward'] else "?"
                ok    = "✓" if m['critere_valide'] else "✗"
                return f"{m['algorithme']} — {m['id']}  {score} pts  {ok}"

            labels  = [_label(m) for m in liste_modeles]
            ids     = [m['id']   for m in liste_modeles]

            # Indice du modèle actif par défaut
            idx_actif = next(
                (i for i, m in enumerate(liste_modeles) if m['actif']),
                0
            )

            # Récupérer la sélection précédente pour détecter un changement
            ancien_id = st.session_state.get('modele_id_actif')

            st.markdown("**Modèle actif**")
            choix_idx = st.selectbox(
                "Sélectionner un modèle",
                options          = range(len(labels)),
                format_func     = lambda i: labels[i],
                index           = idx_actif,
                key             = "selectbox_modele",
                label_visibility = "collapsed",
            )
            modele_selectionne = liste_modeles[choix_idx]
            nouvel_id          = ids[choix_idx]

            # Si changement de modèle → charger + reset session_state
            if nouvel_id != ancien_id:
                log.START_CALL_CONTROLLER_FUNCTION(
                    "GUI", "load_modele", f"POST /models/load id={nouvel_id}"
                )
                r = _api_post(
                    "/models/load", {"id": nouvel_id}, api_url
                )
                if r and r.get("status") == "ok":
                    # Reset complet — données du modèle précédent invalides
                    for k in ['episode_data', 'frame_idx',
                               'historique', 'video_bytes']:
                        st.session_state.pop(k, None)
                    st.session_state['modele_id_actif'] = nouvel_id
                    log.PARAMETER_VALUE("modele chargé", nouvel_id)
                    log.FINISH_CALL_CONTROLLER_FUNCTION(
                        "GUI", "load_modele", "OK — session reset"
                    )
                    st.rerun()
                else:
                    st.error(f"Impossible de charger {nouvel_id}")
                    log.LEVEL_4_ERROR(
                        "GUI", f"load_modele échoué pour {nouvel_id}"
                    )
                    log.FINISH_CALL_CONTROLLER_FUNCTION(
                        "GUI", "load_modele", "ERREUR"
                    )

            # Afficher score du modèle sélectionné
            m = modele_selectionne
            if m['mean_reward']:
                badge = "🟢" if m['critere_valide'] else "🔴"
                st.caption(
                    f"{badge} {m['algorithme']}  "
                    f"**{m['mean_reward']:+.1f} pts**"
                )
        else:
            st.warning("Aucun modèle disponible")
            st.caption("Entraîner un agent d abord (notebook)")

    st.markdown("---")
    st.markdown("**Menu**")

    # ── Navigation — désactivée si pas de modèle ─────────
    # pages_disponibles = ["GUI Épisode", "Dashboard", "Modèle", "🕹️ Interactif"]
    pages_disponibles = ["GUI Épisode", "Dashboard", "Modèle"]

    if modele_selectionne is None:
        # Aucun modèle — menu grisé
        st.markdown(
            "<div style='color:#888;font-size:13px'>"
            "Sélectionner un modèle pour accéder aux pages."
            "</div>",
            unsafe_allow_html=True
        )
        page = None
    else:
        page = st.radio(
            "Navigation",
            pages_disponibles,
            label_visibility="collapsed",
        )

    st.markdown("---")
    st.caption("AstroDynamics · Eagle-1")
    st.caption("Rafael CEREZO MARTIN")

log.FINISH_ACTION("GUI", "rendu", f"page={page}")


# ============================================================
# Garde — aucun modèle sélectionné
# ============================================================
if modele_selectionne is None:
    if not api_ok:
        st.info("🔌 API non disponible — lancer uvicorn src.api:app")
    # Pas de st.stop() ici — la page Épisode gère la sélection

# ============================================================
# PAGE : ÉPISODE
# ============================================================
if page == "GUI Épisode":

    log.START_ACTION("GUI", "page_episode", "rendu page épisode")

    # ***************************************************************
    # SÉLECTION DU MODÈLE — avant tout affichage de la page
    # Liste tous les modèles disponibles via /models/list
    # Une fois sélectionné et chargé, le reste de la page s'affiche
    # ***************************************************************

    # ── Récupérer la liste des modèles disponibles ────────────────
    modele_id_courant = st.session_state.get('modele_id_actif')

    log.START_CALL_CONTROLLER_FUNCTION(
        "GUI", "models_list_episode", "GET /models/list"
    )
    resp_models = _api_get("/models/list", api_url)
    log.FINISH_CALL_CONTROLLER_FUNCTION(
        "GUI", "models_list_episode", "terminé"
    )

    if not api_ok:
        st.error("🔌 API non disponible — lancer uvicorn src.api:app")
        st.stop()

    liste_modeles = (resp_models or {}).get("modeles", [])

    if not liste_modeles:
        # ── Aucun modèle disponible ───────────────────────────────
        st.warning(
            "🤖 Aucun modèle disponible dans `models/`.  "
            "Entraîner un agent PPO ou DQN dans le notebook."
        )
        st.stop()

    # ── Bouton "Sélectionner un modèle" — affiche la liste ────────
    if not modele_id_courant:
        st.info("Sélectionner un modèle pour commencer.")
        if st.button("🤖 Sélectionner un modèle", type="primary",
                     key="btn_ouvrir_liste"):
            st.session_state['liste_ouverte'] = True
            st.rerun()

    # ── Liste des modèles — affichée après clic ───────────────────
    if not modele_id_courant or st.session_state.get('liste_ouverte'):

        st.markdown("### Modèles disponibles")
        st.caption("Cliquer sur un modèle pour le charger et démarrer.")

        # ── Une ligne par modèle ──────────────────────────────────
        for m in liste_modeles:
            score_txt = f"{m['mean_reward']:+.1f} pts" if m['mean_reward'] else "? pts"
            badge     = "🟢" if m['critere_valide'] else "🔴"
            actif_txt = " ✦ actif" if m['actif'] else ""
            label_btn = (
                f"{badge}  {m['algorithme']}  ·  {m['id']}"
                f"  ·  {score_txt}{actif_txt}"
            )
            if st.button(label_btn, key=f"btn_modele_{m['id']}",
                         use_container_width=True):
                # ── Charger le modèle sélectionné ─────────────────
                log.START_CALL_CONTROLLER_FUNCTION(
                    "GUI", "load_modele",
                    f"POST /models/load id={m['id']}"
                )
                r = _api_post("/models/load", {"id": m['id']}, api_url)
                if r and r.get("status") == "ok":
                    for k in ['episode_data', 'frame_idx',
                               'historique', 'video_bytes']:
                        st.session_state.pop(k, None)
                    st.session_state['modele_id_actif'] = m['id']
                    st.session_state['liste_ouverte']   = False
                    log.PARAMETER_VALUE("modele chargé", m['id'])
                    log.FINISH_CALL_CONTROLLER_FUNCTION(
                        "GUI", "load_modele", "OK"
                    )
                    st.rerun()
                else:
                    st.error(f"Impossible de charger {m['id']}")
                    log.LEVEL_4_ERROR(
                        "GUI", f"load_modele échoué pour {m['id']}"
                    )

        # ── Arrêter ici si pas encore de modèle chargé ───────────
        if not modele_id_courant:
            st.stop()

    # ── Modèle actif — affichage compact + bouton changer ─────────
    # ---------------------------------------------------------------
    health_r  = _api_get("/health", api_url) or {}
    score_act = health_r.get('mean_reward')
    algo_act  = health_r.get('algorithme', '?')
    critere   = health_r.get('critere_valide', False)
    badge_act = "🟢" if critere else "🔴"

    col_modele, col_changer = st.columns([5, 1])
    with col_modele:
        st.markdown(
            f"{badge_act} **Modèle actif :** `{modele_id_courant}`  "
            f"· {algo_act}"
            + (f" · **{score_act:+.1f} pts**" if score_act else ""),
        )
    with col_changer:
        if st.button("⇄ Changer", key="btn_changer_modele"):
            st.session_state['liste_ouverte'] = True
            st.rerun()

    # ---------------------------------------------------------------

    st.title("Visualisation d'épisode")

    # ----------------------------------------------------------
    # Contrôles
    # ----------------------------------------------------------
    col_seed, col_speed, col_btn, col_reset = st.columns([1, 2, 1, 1])
    with col_seed:
        seed = st.number_input("Seed", value=42, step=1)
    with col_speed:
        vitesse = st.slider("Vitesse animation", 1, 20, 15, step=1,
                            help="Frames par seconde affichées")
    with col_btn:
        lancer = st.button("▶ Jouer", type="primary",
                           use_container_width=True)
    with col_reset:
        reset = st.button("↺ Reset", use_container_width=True)

    if reset:
        log.LEVEL_7_INFO("GUI", "reset session_state épisode")
        for key in ['episode_data', 'frame_idx']:
            st.session_state.pop(key, None)
        st.rerun()

    # ----------------------------------------------------------
    # Lancement d'un épisode
    # ----------------------------------------------------------
    if lancer:
        log.START_CALL_CONTROLLER_FUNCTION(
            "GUI", "lancer_episode", f"seed={seed}"
        )
        if health is None or health.get("status") != "ok":
            st.error("API non disponible — vérifier le serveur.")
            log.LEVEL_4_ERROR("GUI", "tentative épisode sans API")
        else:
            with st.spinner("Exécution de l'épisode en cours…"):
                data = _api_post(
                    "/episode",
                    {"seed": int(seed), "render": True},
                    api_url
                )
            if data:
                st.session_state['episode_data'] = data
                st.session_state['frame_idx']    = 0
                log.PARAMETER_VALUE("total_reward", data['total_reward'])
                log.PARAMETER_VALUE("n_steps",      data['n_steps'])
                log.PARAMETER_VALUE("success",      data['success'])
        log.FINISH_CALL_CONTROLLER_FUNCTION(
            "GUI", "lancer_episode", "terminé"
        )

    if 'episode_data' in st.session_state:
        data    = st.session_state['episode_data']
        frames  = data['frames']
        rewards = data['rewards']
        actions = data['actions']
        obs_all = data['obs']
        n_steps = data['n_steps']
        total_r = data['total_reward']
        success = data['success']

        log.START_CALL_CONTROLLER_FUNCTION(
            "GUI", "afficher_episode",
            f"n_steps={n_steps} reward={total_r:.1f}"
        )

        if success:
            st.success(f"✓ Atterrissage réussi — {total_r:+.1f} pts "
                       f"— {n_steps} pas")
        else:
            st.warning(f"✗ Échec — {total_r:+.1f} pts — {n_steps} pas")

        # ── Fragment — simulation + télémétrie + animation ────
        _bloque_simulacion(frames, rewards, actions, obs_all, n_steps, vitesse)

        # ── Graphiques ────────────────────────────────────────
        col_g1, col_g2 = st.columns(2)
        with col_g1:
            st.subheader("Récompense cumulée")
            fig_r = _fig_reward(rewards)
            st.pyplot(fig_r, use_container_width=True)
            plt.close(fig_r)
        with col_g2:
            st.subheader("Distribution des actions")
            fig_a = _fig_actions(actions)
            st.pyplot(fig_a, use_container_width=True)
            plt.close(fig_a)

        # ── Vidéo de l'épisode — assemblée côté API ───────────
        st.subheader("Vidéo de l'épisode")
        st.caption(
            "La vidéo est assemblée par l'API à la demande — "
            "l'animation interactive ci-dessus reste disponible."
        )

        if st.button("🎬 Générer la vidéo MP4", key="btn_video_ep"):
            log.START_CALL_CONTROLLER_FUNCTION(
                "GUI", "get_episode_video", "GET /episode/video"
            )
            with st.spinner("Assemblage MP4 en cours…"):
                try:
                    r = requests.get(
                        f"{api_url}/episode/video",
                        timeout = 60,
                    )
                    if r.status_code == 200:
                        video_bytes = r.content
                        st.session_state['video_bytes'] = video_bytes
                        log.PARAMETER_VALUE(
                            "taille_mp4", f"{len(video_bytes):,} bytes"
                        )
                        log.FINISH_CALL_CONTROLLER_FUNCTION(
                            "GUI", "get_episode_video",
                            f"{len(video_bytes):,} bytes"
                        )
                    else:
                        detail = r.json().get('detail', r.text)
                        st.error(f"Erreur API : {detail}")
                        log.LEVEL_4_ERROR(
                            "GUI",
                            f"/episode/video → {r.status_code} : {detail}"
                        )
                        log.FINISH_CALL_CONTROLLER_FUNCTION(
                            "GUI", "get_episode_video", "ERREUR"
                        )
                except Exception as e:
                    st.error(f"Requête échouée : {e}")
                    log.LEVEL_4_ERROR("GUI", f"get_episode_video échoué : {e}")
                    log.FINISH_CALL_CONTROLLER_FUNCTION(
                        "GUI", "get_episode_video", "ERREUR"
                    )

        # Affichage si vidéo disponible en session_state
        if 'video_bytes' in st.session_state:
            st.video(st.session_state['video_bytes'])

        log.FINISH_CALL_CONTROLLER_FUNCTION("GUI", "afficher_episode", "rendu complet")

    log.FINISH_ACTION("GUI", "page_episode", "rendu terminé")

# ============================================================
# PAGE : DASHBOARD
# ============================================================
elif page == "Dashboard":

    log.START_ACTION("GUI", "page_dashboard", "rendu tableau de bord")
    st.title("Tableau de bord — performance")

    if 'historique' not in st.session_state:
        st.session_state['historique'] = []

    historique = st.session_state['historique']

    col_n, col_btn, col_reset = st.columns([2, 1, 1])
    with col_n:
        n_ep = st.selectbox("Jouer N épisodes", [1, 5, 10, 50], index=1)
    with col_btn:
        lancer_dash = st.button("▶ Lancer", type="primary",
                                use_container_width=True)
    with col_reset:
        if st.button("🗑 Reset", use_container_width=True):
            log.LEVEL_7_INFO("GUI", "reset historique dashboard")
            st.session_state['historique'] = []
            st.rerun()

    if lancer_dash:
        log.START_CALL_CONTROLLER_FUNCTION(
            "GUI", "lancer_dashboard", f"{n_ep} épisodes"
        )
        if health is None or health.get("status") != "ok":
            st.error("API non disponible.")
            log.LEVEL_4_ERROR("GUI", "tentative dashboard sans API")
        else:
            bar = st.progress(0, text="Exécution des épisodes…")
            for i in range(n_ep):
                seed_i = len(historique) + i
                log.STEP(2, f"épisode {i+1}/{n_ep}", f"seed={seed_i}")
                data = _api_post(
                    "/episode",
                    {"seed": seed_i, "render": False},
                    api_url
                )
                if data:
                    historique.append(data)
                    log.PARAMETER_VALUE(
                        f"ep{i+1} reward", data['total_reward']
                    )
                bar.progress(
                    (i + 1) / n_ep,
                    text=f"Épisode {i + 1} / {n_ep}"
                )
            bar.empty()
            st.session_state['historique'] = historique
            log.FINISH_CALL_CONTROLLER_FUNCTION(
                "GUI", "lancer_dashboard",
                f"{len(historique)} épisodes total"
            )
            st.rerun()

    if not historique:
        st.info("Aucune donnée — lancer des épisodes ci-dessus.")
    else:
        rewards_tot  = [ep['total_reward'] for ep in historique]
        succes_list  = [ep['success']       for ep in historique]
        actions_all  = [a for ep in historique for a in ep['actions']]
        obs_all_flat = [o for ep in historique for o in ep['obs']]

        n_total  = len(historique)
        moy      = np.mean(rewards_tot)
        taux     = np.mean(succes_list) * 100
        meilleur = np.max(rewards_tot)

        log.STEP(2, "métriques dashboard")
        log.PARAMETER_VALUE("n_total",  n_total)
        log.PARAMETER_VALUE("moy",      round(moy, 2))
        log.PARAMETER_VALUE("taux_%",   round(taux, 1))
        log.PARAMETER_VALUE("meilleur", round(meilleur, 2))

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Épisodes joués",     n_total)
        c2.metric("Récompense moyenne", f"{moy:+.1f}")
        c3.metric("Taux de succès",     f"{taux:.0f} %")
        c4.metric("Meilleur score",     f"{meilleur:+.1f}")

        st.markdown("---")

        # ── Récompense par épisode ────────────────────────────
        st.subheader("Récompense par épisode")
        fig1, ax1 = plt.subplots(figsize=(10, 3))
        fig1.patch.set_facecolor('none')
        ax1.set_facecolor('none')
        couleurs_ep = ['#1D9E75' if s else '#E24B4A' for s in succes_list]
        ax1.scatter(range(1, n_total + 1), rewards_tot,
                    c=couleurs_ep, s=30, zorder=3)
        ax1.plot(range(1, n_total + 1), rewards_tot,
                 color='#AFA9EC', linewidth=0.8, alpha=0.5)
        ax1.axhline(200, color='#1D9E75', linewidth=1,
                    linestyle='--', label='seuil 200')
        ax1.axhline(moy, color='#7F77DD', linewidth=1.5,
                    linestyle='--', label=f'moy. {moy:+.1f}')
        ax1.set_xlabel("Épisode", color='white', fontsize=9)
        ax1.set_ylabel("Récompense", color='white', fontsize=9)
        ax1.tick_params(colors='white', labelsize=8)
        patch_ok = mpatches.Patch(color='#1D9E75', label='succès')
        patch_ko = mpatches.Patch(color='#E24B4A', label='échec')
        ax1.legend(handles=[patch_ok, patch_ko],
                   fontsize=8, facecolor='#222', labelcolor='white',
                   loc='lower right')
        for sp in ax1.spines.values():
            sp.set_edgecolor('#444')
        plt.tight_layout()
        st.pyplot(fig1, use_container_width=True)
        plt.close(fig1)

        st.markdown("---")

        col_act, col_alt = st.columns(2)

        with col_act:
            st.subheader("Distribution des actions")
            filtre = st.radio(
                "Filtrer par",
                ["Tous", "Succès uniquement", "Échecs uniquement"],
                horizontal=True, key="filtre_actions"
            )
            if filtre == "Succès uniquement":
                actions_filtre = [
                    a for ep in historique if ep['success']
                    for a in ep['actions']
                ]
            elif filtre == "Échecs uniquement":
                actions_filtre = [
                    a for ep in historique if not ep['success']
                    for a in ep['actions']
                ]
            else:
                actions_filtre = actions_all

            if actions_filtre:
                fig2 = _fig_actions(actions_filtre)
                st.pyplot(fig2, use_container_width=True)
                plt.close(fig2)
            else:
                st.info("Aucune donnée pour ce filtre.")

        with col_alt:
            st.subheader("Décisions selon l'altitude")
            st.caption("obs[1] altitude vs action choisie")
            if obs_all_flat:
                altitudes = [o[1] for o in obs_all_flat]
                act_flat  = [a for ep in historique for a in ep['actions']]
                fig3, ax3 = plt.subplots(figsize=(5, 3.5))
                fig3.patch.set_facecolor('none')
                ax3.set_facecolor('none')
                couleurs_pts = [COULEURS_ACTIONS[a] for a in act_flat]
                ax3.scatter(act_flat, altitudes,
                            c=couleurs_pts, alpha=0.3, s=6)
                ax3.set_xlabel("action", color='white', fontsize=9)
                ax3.set_ylabel("altitude (obs[1])", color='white', fontsize=9)
                ax3.set_xticks(range(4))
                ax3.set_xticklabels(
                    [f"{i}\n{NOMS_ACTIONS[i]}" for i in range(4)],
                    fontsize=7, color='white'
                )
                ax3.tick_params(axis='y', colors='white', labelsize=7)
                for sp in ax3.spines.values():
                    sp.set_edgecolor('#444')
                plt.tight_layout()
                st.pyplot(fig3, use_container_width=True)
                plt.close(fig3)

        st.markdown("---")

        # ── Dernier épisode ───────────────────────────────────
        st.subheader("Dernier épisode — trajectoire")
        dernier = historique[-1]

        col_d1, col_d2 = st.columns(2)
        with col_d1:
            fig4 = _fig_reward(dernier['rewards'])
            st.pyplot(fig4, use_container_width=True)
            plt.close(fig4)

        with col_d2:
            st.markdown(
                f"**Résultat :** "
                f"{'✓ Succès' if dernier['success'] else '✗ Échec'}  "
                f"| **Score :** {dernier['total_reward']:+.1f} pts  "
                f"| **Durée :** {dernier['n_steps']} pas"
            )
            act_dern = dernier['actions']
            fig5, ax5 = plt.subplots(figsize=(5, 0.8))
            fig5.patch.set_facecolor('none')
            ax5.set_facecolor('none')
            for i, a in enumerate(act_dern):
                ax5.barh(0, 1, left=i,
                         color=COULEURS_ACTIONS[a],
                         height=0.8, linewidth=0)
            ax5.set_xlim(0, len(act_dern))
            ax5.set_ylim(-0.5, 0.5)
            ax5.axis('off')
            plt.tight_layout()
            st.pyplot(fig5, use_container_width=True)
            plt.close(fig5)

            for a_id, a_nom in NOMS_ACTIONS.items():
                pct = act_dern.count(a_id) / len(act_dern) * 100
                st.markdown(
                    f"<span style='background:{COULEURS_ACTIONS[a_id]};"
                    f"color:white;padding:1px 8px;border-radius:10px;"
                    f"font-size:12px'>{a_id} · {a_nom}</span> "
                    f"<span style='font-size:12px'>{pct:.0f}%</span>",
                    unsafe_allow_html=True
                )

    log.FINISH_ACTION("GUI", "page_dashboard", "rendu terminé")


# ============================================================
# PAGE : MODÈLE
# ============================================================
elif page == "Modèle":

    log.START_ACTION("GUI", "page_modele", "rendu page modèle")
    st.title("Informations du modèle")

    log.START_CALL_CONTROLLER_FUNCTION(
        "GUI", "charger_model_info", "/model/info"
    )
    info = _api_get("/model/info", api_url)

    if info is None:
        st.error("API non disponible ou pas de métadonnées JSON.")
        log.LEVEL_4_ERROR("GUI", "model/info inaccessible")
        log.FINISH_CALL_CONTROLLER_FUNCTION(
            "GUI", "charger_model_info", "ERREUR"
        )
        st.stop()

    log.PARAMETER_VALUE("algorithme", info.get('algorithme'))
    log.PARAMETER_VALUE("mean_reward",
        info.get('resultats', {}).get('mean_reward'))
    log.FINISH_CALL_CONTROLLER_FUNCTION(
        "GUI", "charger_model_info", "OK"
    )

    if "message" in info:
        st.warning(info["message"])
        st.stop()

    col_id1, col_id2, col_id3 = st.columns(3)
    col_id1.metric("Algorithme",   info.get("algorithme", "—"))
    col_id2.metric("Environnement", info.get("environnement", "—"))
    col_id3.metric("Policy",        info.get("policy", "—"))

    st.markdown(f"**Fichier modèle :** `{info.get('fichier_modele', '—')}`")
    st.markdown(
        f"**Auteur :** {info.get('auteur', '—')}  "
        f"| **Date :** {info.get('date', '—')[:10]}"
    )

    st.markdown("---")

    st.subheader("Résultats d'évaluation")
    res = info.get("resultats", {})
    c1, c2, c3 = st.columns(3)
    c1.metric("Récompense moyenne", f"{res.get('mean_reward', 0):+.1f}")
    c2.metric("Écart-type",         f"{res.get('std_reward', 0):.1f}")
    c3.metric("Critère ≥ 200",
              "✓ Validé" if res.get('critere_valide') else "✗ Non atteint")
    st.markdown(
        f"Évaluation sur **{res.get('n_eval_episodes', 100)} épisodes**."
    )

    st.markdown("---")

    st.subheader("Hyperparamètres PPO")
    hp = info.get("hyperparametres", {})
    if hp:
        df_hp = pd.DataFrame(
            list(hp.items()), columns=["Paramètre", "Valeur"]
        )
        st.dataframe(df_hp, hide_index=True, use_container_width=True)
    else:
        st.info("Hyperparamètres non disponibles dans le JSON.")

    st.markdown("---")

    st.subheader("Comparaison des agents")
    baseline = info.get("baseline_aleatoire", {})
    dqn      = info.get("comparaison_dqn", {})

    df_comp = pd.DataFrame({
        "Agent"              : ["Baseline aléatoire", "DQN", "PPO"],
        "Récompense moyenne" : [
            baseline.get("mean_reward", "—"),
            dqn.get("mean_reward", "—"),
            res.get("mean_reward", "—"),
        ],
        "Écart-type"         : [
            baseline.get("std_reward", "—"),
            dqn.get("std_reward", "—"),
            res.get("std_reward", "—"),
        ],
        "Critère ≥ 200"      : [
            "✗",
            "✓" if (dqn.get("mean_reward") or 0) >= 200 else "✗",
            "✓" if res.get('critere_valide') else "✗",
        ],
    })
    st.dataframe(df_comp, hide_index=True, use_container_width=True)

    agents   = ["Baseline", "DQN", "PPO"]
    scores   = [
        baseline.get("mean_reward") or -193.6,
        dqn.get("mean_reward")      or 0,
        res.get("mean_reward")      or 0,
    ]
    fig_c, ax_c = plt.subplots(figsize=(6, 3))
    fig_c.patch.set_facecolor('none')
    ax_c.set_facecolor('none')
    bars = ax_c.bar(agents, scores,
                    color=["#888780", "#EF9F27", "#534AB7"],
                    edgecolor='white', linewidth=0.5)
    ax_c.axhline(200, color='#1D9E75', linewidth=1.5,
                 linestyle='--', label='objectif 200')
    ax_c.axhline(0, color='#888', linewidth=0.5, linestyle=':')
    for bar, sc in zip(bars, scores):
        ax_c.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 2,
            f"{sc:+.0f}",
            ha='center', fontsize=9, color='white'
        )
    ax_c.set_ylabel("Récompense moyenne", color='white', fontsize=9)
    ax_c.tick_params(colors='white', labelsize=9)
    ax_c.legend(fontsize=8, facecolor='#222', labelcolor='white')
    for sp in ax_c.spines.values():
        sp.set_edgecolor('#444')
    plt.tight_layout()
    st.pyplot(fig_c, use_container_width=True)
    plt.close(fig_c)

    log.FINISH_ACTION("GUI", "page_modele", "rendu terminé")


# ============================================================
# PAGE : INTERACTIF — Human vs Agent (physique réelle)
# ============================================================
# -----------------------------------------------------------
elif page == "🕹️ Interactif":

    log.START_ACTION("GUI", "page_interactif", "rendu page interactive")
    st.title("🕹️ Mode interactif — Human vs Agent")
    st.caption(
        "Le lander tombe automatiquement sous l'effet de la gravité. "
        "Cliquez sur un bouton d'action avant le prochain pas "
        "pour intervenir — sinon do nothing est appliqué automatiquement."
    )

    # ***************************************************************
    # CONTRÔLES PRINCIPAUX — vitesse, seed, démarrer
    # ***************************************************************
    col_c1, col_c2, col_c3, col_c4 = st.columns([2, 1, 1, 1])
    with col_c1:
        intervalle = st.slider(
            "Intervalle (s) — temps pour choisir",
            min_value = 0.3,
            max_value = 3.0,
            value     = 0.8,
            step      = 0.1,
            key       = "slider_intervalle_interactif",
            help      = "Temps avant envoi automatique de do nothing"
        )
    with col_c2:
        seed_interactif = st.number_input(
            "Seed", value=42, step=1, key="seed_interactif"
        )
    with col_c3:
        btn_start_i = st.button(
            "▶ Démarrer",
            type                = "primary",
            key                 = "btn_start_interactif",
            use_container_width = True,
        )
    with col_c4:
        btn_reset_i = st.button(
            "↺ Reset",
            key                 = "btn_reset_interactif",
            use_container_width = True,
        )

    # ---------------------------------------------------------------
    # Démarrer / réinitialiser un épisode
    # ---------------------------------------------------------------
    if btn_start_i or btn_reset_i:
        log.START_CALL_CONTROLLER_FUNCTION(
            "GUI", "interactif_start",
            f"POST /interactif/start seed={seed_interactif}"
        )
        r = _api_post(
            "/interactif/start",
            {"seed": int(seed_interactif), "render": True},
            api_url
        )
        if r:
            st.session_state["int_obs"]        = r["obs"]
            st.session_state["int_frame"]      = r["frame_b64"]
            st.session_state["int_reward"]     = 0.0
            st.session_state["int_n_pas"]      = 0
            st.session_state["int_done"]       = False
            st.session_state["int_historique"] = []
            st.session_state["int_actif"]      = True
            log.PARAMETER_VALUE("seed", seed_interactif)
            log.FINISH_CALL_CONTROLLER_FUNCTION(
                "GUI", "interactif_start", "OK"
            )
            st.rerun()
        else:
            st.error("Impossible de démarrer — API non disponible.")
            log.LEVEL_4_ERROR("GUI", "interactif_start échoué")

    # ---------------------------------------------------------------
    # Attente de démarrage
    # ---------------------------------------------------------------
    if "int_frame" not in st.session_state:
        st.info("Cliquer sur **▶ Démarrer** pour lancer l'épisode.")
        log.FINISH_ACTION("GUI", "page_interactif", "en attente")
        st.stop()

    # ---------------------------------------------------------------
    # Épisode terminé — résultat final
    # ---------------------------------------------------------------
    if st.session_state.get("int_done", False):
        reward_fin = st.session_state["int_reward"]
        n_pas_fin  = st.session_state["int_n_pas"]
        historique = st.session_state.get("int_historique", [])

        if reward_fin >= 200:
            st.success(
                f"🏆 Atterrissage réussi !  "
                f"Score : **{reward_fin:+.1f} pts** — {n_pas_fin} pas"
            )
        else:
            st.warning(
                f"💥 Épisode terminé.  "
                f"Score : **{reward_fin:+.1f} pts** — {n_pas_fin} pas"
            )

        if len(historique) > 1:
            fig_fin, ax_fin = plt.subplots(figsize=(9, 2.5))
            fig_fin.patch.set_facecolor("none")
            ax_fin.set_facecolor("none")
            ax_fin.plot(
                [h["reward_cum"] for h in historique],
                color="#EF9F27", linewidth=1.5, label="Votre score"
            )
            ax_fin.axhline(200, color="#1D9E75",
                           linewidth=1, linestyle="--", label="seuil 200")
            ax_fin.set_xlabel("pas", fontsize=8, color="white")
            ax_fin.set_ylabel("récompense cumulée",
                              fontsize=8, color="white")
            ax_fin.tick_params(colors="white", labelsize=7)
            ax_fin.legend(fontsize=7, facecolor="#222",
                          labelcolor="white")
            for sp in ax_fin.spines.values():
                sp.set_edgecolor("#444")
            plt.tight_layout()
            st.pyplot(fig_fin, use_container_width=True)
            plt.close(fig_fin)

        log.FINISH_ACTION("GUI", "page_interactif", "épisode terminé")
        st.stop()

    # ---------------------------------------------------------------
    # Boutons d'action — HORS du fragment
    # Exécutent le step IMMÉDIATEMENT au clic sans attendre
    # le prochain cycle automatique du fragment
    # ---------------------------------------------------------------
    n_pas_courant = st.session_state.get("int_n_pas", 0)

    st.markdown("---")
    st.caption(
        f"⏱ Vous avez **{intervalle:.1f}s** pour choisir — "
        f"sinon **do nothing** est appliqué automatiquement."
    )

    LABELS_BTN = {
        1: "◀  Moteur gauche",
        2: "▲  Moteur principal",
        3: "▶  Moteur droit",
    }
    cols_btn = st.columns(3)
    for col_b, (a_id, a_label) in zip(cols_btn, LABELS_BTN.items()):
        with col_b:
            if st.button(
                a_label,
                key                 = f"btn_i_{n_pas_courant}_{a_id}",
                use_container_width = True,
            ):
                # ── Step immédiat — physique réelle ───────────
                log.START_CALL_CONTROLLER_FUNCTION(
                    "GUI", "action_joueur",
                    f"action={a_id} {NOMS_ACTIONS[a_id]} [JOUEUR]"
                )
                r = _api_post(
                    "/interactif/step",
                    {"action": a_id},
                    api_url
                )
                if r:
                    st.session_state["int_obs"]      = r["obs"]
                    st.session_state["int_frame"]    = r["frame_b64"]
                    st.session_state["int_reward"]   = r["reward_cum"]
                    st.session_state["int_n_pas"]    = r["n_pas"]
                    st.session_state["int_done"]     = r["done"]
                    st.session_state["int_historique"].append({
                        "n_pas"     : r["n_pas"],
                        "action"    : a_id,
                        "reward"    : r["reward"],
                        "reward_cum": r["reward_cum"],
                    })
                    log.PARAMETER_VALUE(
                        "action",
                        f"{a_id} {NOMS_ACTIONS[a_id]} [JOUEUR]"
                    )
                    log.PARAMETER_VALUE("reward",     r["reward"])
                    log.PARAMETER_VALUE("reward_cum", r["reward_cum"])
                    log.FINISH_CALL_CONTROLLER_FUNCTION(
                        "GUI", "action_joueur", f"done={r['done']}"
                    )
                    st.rerun()

    # ***************************************************************
    # FRAGMENT — boucle do nothing automatique
    # Re-exécuté toutes les `intervalle` secondes.
    # N'envoie QUE do nothing — les boutons joueur sont hors fragment
    # et ont déjà exécuté leur step avant que le fragment tourne.
    # ***************************************************************
    @st.fragment(run_every=intervalle)
    def _boucle_interactif():
        """
        Fragment auto-rafraîchi — envoie do nothing automatiquement
        si le joueur n'a pas cliqué pendant l'intervalle.
        Affiche la frame et la télémétrie courantes.
        """
        frame_b64  = st.session_state.get("int_frame",  "")
        obs_c      = st.session_state.get("int_obs",    [0]*8)
        reward_cum = st.session_state.get("int_reward", 0.0)
        n_pas      = st.session_state.get("int_n_pas",  0)
        done       = st.session_state.get("int_done",   False)
        historique = st.session_state.get("int_historique", [])

        if done or not st.session_state.get("int_actif", False):
            st.rerun()
            return

        # ── Envoyer do nothing — physique réelle ──────────────────
        log.START_CALL_MANAGER_FUNCTION(
            "GUI", "boucle_auto",
            f"pas={n_pas} action=0 [auto]"
        )
        r = _api_post(
            "/interactif/step",
            {"action": 0},
            api_url
        )
        if r:
            st.session_state["int_obs"]      = r["obs"]
            st.session_state["int_frame"]    = r["frame_b64"]
            st.session_state["int_reward"]   = r["reward_cum"]
            st.session_state["int_n_pas"]    = r["n_pas"]
            st.session_state["int_done"]     = r["done"]
            st.session_state["int_historique"].append({
                "n_pas"     : r["n_pas"],
                "action"    : 0,
                "reward"    : r["reward"],
                "reward_cum": r["reward_cum"],
            })
            frame_b64  = r["frame_b64"]
            obs_c      = r["obs"]
            reward_cum = r["reward_cum"]
            n_pas      = r["n_pas"]
            done       = r["done"]
            historique = st.session_state["int_historique"]

        log.FINISH_CALL_MANAGER_FUNCTION(
            "GUI", "boucle_auto",
            f"pas={n_pas} done={done}"
        )

        if done:
            st.rerun()
            return

        # ── Affichage frame + télémétrie ──────────────────────────
        col_f, col_t = st.columns([3, 2])

        with col_f:
            st.image(_decoder_frame(frame_b64), width="stretch")
            st.progress(
                n_pas / 1000,
                text=f"Pas {n_pas} / 1000  |  {reward_cum:+.1f} pts"
            )

        with col_t:
            st.metric("Récompense cumulée", f"{reward_cum:+.2f}")
            st.dataframe(
                pd.DataFrame({
                    "Variable": NOMS_OBS,
                    "Valeur"  : [f"{v:+.5f}" for v in obs_c],
                }),
                hide_index          = True,
                use_container_width = True,
                height              = 250,
            )

        # ── Mini-courbe ───────────────────────────────────────────
        if len(historique) > 2:
            fig_p, ax_p = plt.subplots(figsize=(8, 1.5))
            fig_p.patch.set_facecolor("none")
            ax_p.set_facecolor("none")
            ax_p.plot(
                [h["reward_cum"] for h in historique],
                color="#EF9F27", linewidth=1.2
            )
            ax_p.axhline(200, color="#1D9E75",
                         linewidth=0.8, linestyle="--")
            ax_p.tick_params(colors="white", labelsize=6)
            for sp in ax_p.spines.values():
                sp.set_edgecolor("#444")
            plt.tight_layout()
            st.pyplot(fig_p, use_container_width=True)
            plt.close(fig_p)

    # ── Appel du fragment ──────────────────────────────────────────
    _boucle_interactif()

    log.FINISH_ACTION("GUI", "page_interactif", "rendu terminé")