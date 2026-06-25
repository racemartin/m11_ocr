# ============================================================
# api.py — API FastAPI Eagle-1 / AstroDynamics
# Backend RL : toute la logique côté serveur
# Lancement : uvicorn src.api:app --host 0.0.0.0 --port 8000
# ============================================================

# — Bibliothèques standard
import base64                        # encodage frames en base64
import io                            # flux mémoire pour images
import json                          # lecture métadonnées JSON
import sys                           # manipulation du path Python
import tempfile                      # fichiers temporaires pour MP4
import os                            # suppression fichiers temporaires
from pathlib import Path             # chemins portables
from typing  import List, Optional   # annotations de types

# — Web framework
import uvicorn                       # serveur ASGI
from fastapi             import FastAPI, HTTPException  # framework API
from fastapi.middleware.cors import CORSMiddleware      # CORS pour Streamlit
from fastapi.responses   import Response               # réponse binaire MP4
from pydantic            import BaseModel              # validation données

# — Calcul vectoriel et image
import numpy             as np       # tableaux numériques
from PIL                 import Image  # conversion frame numpy → PNG

# — Environnement et modèle RL
import gymnasium         as gym      # API Gymnasium 1.3+
from stable_baselines3   import PPO, DQN  # algorithmes SB3

import imageio_ffmpeg

# — Outil de journalisation
sys.path.insert(0, str(Path(__file__).resolve().parent))
from tools.rafael.log_tool import LogTool  # journalisation colorée RFC 5424

# ----------------------------------------------------------
# Instance globale du logger
# ----------------------------------------------------------
log = LogTool(origin="api")

# ----------------------------------------------------------
# Configuration des chemins
# ----------------------------------------------------------
try:
    import google.colab
    BASE = Path('/content/drive/MyDrive/OpenClassrooms/m11_ocr')
except ImportError:
    BASE = Path(__file__).resolve().parent.parent

MODELS = BASE / 'models'

# ----------------------------------------------------------
# Noms des actions LunarLander-v3
# ----------------------------------------------------------
NOMS_ACTIONS = {
    0: "do nothing",
    1: "moteur gauche",
    2: "moteur principal",
    3: "moteur droit",
}

# ============================================================
# Chargement du modèle au démarrage (une seule fois)
# ============================================================

def _trouver_meilleur_modele():
    """
    Cherche le meilleur modèle disponible dans models/.
    Priorité : best_ppo/best_model.zip > ppo_*.zip > dqn_*.zip
    """
    log.START_CALL_CONTROLLER_FUNCTION(
        "API", "_trouver_meilleur_modele", "recherche du meilleur modèle"
    )

    # ----------------------------------------------------------
    # Inventaire complet des modèles disponibles
    # ----------------------------------------------------------
    chemin_best_ppo = MODELS / 'best_ppo' / 'best_model.zip'
    chemin_best_dqn = MODELS / 'best_dqn' / 'best_model.zip'
    jsons_ppo = sorted(MODELS.glob('ppo_lunarlander_*.json'),     reverse=True)
    jsons_dqn = sorted(MODELS.glob('*dqn_lunarlander_*.json'),    reverse=True)
    ppo_zips        = sorted(MODELS.glob('ppo_lunarlander_*.zip'),  reverse=True)
    dqn_zips        = sorted(MODELS.glob('*dqn_lunarlander_*.zip'), reverse=True)
    log.PARAMETER_VALUE("best_ppo existe",  chemin_best_ppo.exists())
    log.PARAMETER_VALUE("best_dqn existe",  chemin_best_dqn.exists())
    
    log.PARAMETER_VALUE("jsons_ppo trouvés", len(jsons_ppo))
    for j in jsons_ppo: log.PARAMETER_VALUE("  jsons_ppo", j.name)
    log.PARAMETER_VALUE("jsons_dqn trouvés", len(jsons_dqn))
    for j in jsons_dqn: log.PARAMETER_VALUE("  jsons_dqn", j.name)
        
    log.PARAMETER_VALUE("ppo_zips trouvés", len(ppo_zips))
    for z in ppo_zips: log.PARAMETER_VALUE("  ppo_zip", z.name)
    log.PARAMETER_VALUE("dqn_zips trouvés", len(dqn_zips))
    for z in dqn_zips: log.PARAMETER_VALUE("  dqn_zip", z.name)
    
    # ----------------------------------------------------------
    # Priorité : best_ppo > ppo_*.zip > best_dqn > dqn_*.zip
    # ----------------------------------------------------------
    if chemin_best_ppo.exists():
        json_p = jsons_ppo[0] if jsons_ppo else None
        log.PARAMETER_VALUE("modele choisi", str(chemin_best_ppo))
        log.FINISH_CALL_CONTROLLER_FUNCTION(
            "API", "_trouver_meilleur_modele", "best_ppo trouvé"
        )
        return chemin_best_ppo, 'PPO', json_p

    if ppo_zips:
        stem   = ppo_zips[0].stem
        json_p = MODELS / f'{stem}.json'
        log.PARAMETER_VALUE("modele choisi", str(ppo_zips[0]))
        log.FINISH_CALL_CONTROLLER_FUNCTION(
            "API", "_trouver_meilleur_modele", "ppo_*.zip trouvé"
        )
        return ppo_zips[0], 'PPO', json_p if json_p.exists() else None

    if chemin_best_dqn.exists():
        json_p = jsons_dqn[0] if jsons_dqn else None
        log.PARAMETER_VALUE("modele choisi", str(chemin_best_dqn))
        log.FINISH_CALL_CONTROLLER_FUNCTION(
            "API", "_trouver_meilleur_modele", "best_dqn trouvé"
        )
        return chemin_best_dqn, 'DQN', json_p

    if dqn_zips:
        stem   = dqn_zips[0].stem
        json_p = MODELS / f'{stem}.json'
        log.PARAMETER_VALUE("modele choisi", str(dqn_zips[0]))
        log.FINISH_CALL_CONTROLLER_FUNCTION(
            "API", "_trouver_meilleur_modele", "dqn_*.zip trouvé"
        )
        return dqn_zips[0], 'DQN', json_p if json_p.exists() else None

    log.LEVEL_4_ERROR("API", "Aucun modèle trouvé dans models/")
    log.FINISH_CALL_CONTROLLER_FUNCTION(
        "API", "_trouver_meilleur_modele", "AUCUN modèle"
    )
    return None, None, None

# *****************************************************************************
def ____START(): pass
# *****************************************************************************

log.START_ACTION("API", "startup", "chargement du modèle au démarrage")

chemin_modele, algo, chemin_json = _trouver_meilleur_modele()

if chemin_modele is None:
    log.LEVEL_5_WARNING("API", "API en mode dégradé — aucun modèle disponible")
    _model_init    = None
    _metadata_init = {}
else:
    log.STEP(2, "Chargement modèle SB3")
    log.PARAMETER_VALUE("chemin",     str(chemin_modele))
    log.PARAMETER_VALUE("algorithme", algo)
    _model_init    = (PPO if algo == 'PPO' else DQN).load(str(chemin_modele))
    _metadata_init = {}
    if chemin_json and chemin_json.exists():
        with open(chemin_json, encoding='utf-8') as f:
            _metadata_init = json.load(f)
        log.PARAMETER_VALUE("metadata JSON", chemin_json.name)

log.FINISH_ACTION("API", "startup", f"modèle {algo} prêt")

# ============================================================
# État global mutable du modèle actif
# Modifié par POST /models/load — partagé entre tous les endpoints
# ============================================================
_etat = {
    "model"         : _model_init,     # instance SB3 chargée
    "algo"          : algo,            # "PPO" ou "DQN"
    "chemin_modele" : chemin_modele,   # Path du .zip actif
    "metadata"      : _metadata_init,  # dict JSON des métadonnées
}

# ============================================================
# Application FastAPI
# ============================================================

app = FastAPI(
    title       = "Eagle-1 RL API",
    description = "Backend RL pour le pilote automatique Eagle-1 — AstroDynamics",
    version     = "1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins  = ["*"],
    allow_methods  = ["*"],
    allow_headers  = ["*"],
)

# *****************************************************************************
def ____CLASSES(): pass
# *****************************************************************************

# ============================================================
# Modèles Pydantic
# ============================================================

class EtatRequest(BaseModel):
    state: List[float]

class EpisodeRequest(BaseModel):
    seed:   Optional[int]  = 42
    render: Optional[bool] = True

class LoadRequest(BaseModel):
    """Identifiant du modèle à charger (id retourné par /models/list)."""
    id: str                              # ex: "best_ppo", "best_dqn"

class StepRequest(BaseModel):
    """Action choisie par le joueur (0–3) pour un pas."""
    action: int               # action du joueur : 0 1 2 3

# ============================================================
# État en mémoire — dernier épisode joué (pour /episode/video)
# ============================================================

# Stocke les frames numpy du dernier épisode pour assemblage MP4
# Réinitialisé à chaque appel de /episode
_dernier_episode_frames: list = []   # frames numpy RGB (H, W, 3)
_dernier_episode_seed:   int  = 0    # seed du dernier épisode

# *****************************************************************************
def ____FUNCTIONS(): pass
# *****************************************************************************

# ============================================================
# Fonctions utilitaires
# ============================================================

def _frame_vers_base64(frame: np.ndarray) -> str:
    """Convertit une frame RGB numpy en chaîne base64 PNG."""
    img    = Image.fromarray(frame.astype(np.uint8))
    buffer = io.BytesIO()
    img.save(buffer, format='PNG')
    return base64.b64encode(buffer.getvalue()).decode('utf-8')


def _verifier_modele():
    """Lève une erreur HTTP 503 si le modèle n'est pas chargé."""
    if _etat["model"] is None:
        log.LEVEL_4_ERROR("API", "modèle non disponible — requête rejetée")
        raise HTTPException(
            status_code = 503,
            detail      = "Aucun modèle disponible. Entraîner un agent d'abord."
        )

# ============================================================
# Endpoints
# ============================================================

def ___HEALTH(): pass
# ####################################################################
# GET /health
# ####################################################################
@app.get("/health")
def health():
    """État du service et infos du modèle chargé."""
    log.START_ACTION("API", "/health", "vérification état du service")

    resultat = {
        "status"        : "ok" if _etat["model"] is not None else "degraded",
        "modele"        : str(_etat["chemin_modele"].name) if _etat["chemin_modele"] else None,
        "algorithme"    : _etat["algo"],
        "mean_reward"   : _etat["metadata"].get('resultats', {}).get('mean_reward'),
        "critere_valide": _etat["metadata"].get('resultats', {}).get('critere_valide'),
        "environnement" : _etat["metadata"].get('environnement', 'LunarLander-v3'),
    }

    log.PARAMETER_VALUE("status",      resultat["status"])
    log.PARAMETER_VALUE("modele",      resultat["modele"])
    log.PARAMETER_VALUE("mean_reward", resultat["mean_reward"])
    log.FINISH_ACTION("API", "/health", resultat["status"])

    return resultat


def ___ACTION(): pass
# ####################################################################
# POST /action
# ####################################################################
@app.post("/action")
def predict_action(req: EtatRequest):
    """Reçoit obs[8] et retourne l'action optimale du modèle."""
    log.START_ACTION("API", "/action", "prédiction action")
    _verifier_modele()

    log.START_CALL_CONTROLLER_FUNCTION(
        "API", "predict_action", "validation état"
    )
    if len(req.state) != 8:
        log.LEVEL_4_ERROR(
            "API", f"état invalide : {len(req.state)} valeurs reçues"
        )
        raise HTTPException(
            status_code = 422,
            detail      = f"State doit contenir 8 valeurs, reçu {len(req.state)}"
        )
    log.PARAMETER_VALUE("state[0] x", req.state[0])
    log.PARAMETER_VALUE("state[1] y", req.state[1])
    log.PARAMETER_VALUE("state[4] θ", req.state[4])
    log.FINISH_CALL_CONTROLLER_FUNCTION(
        "API", "predict_action", "validation OK"
    )

    log.START_CALL_MANAGER_FUNCTION("API", "model.predict", "inférence réseau")
    obs       = np.array(req.state, dtype=np.float32)
    action, _ = _etat["model"].predict(obs, deterministic=True)
    action    = int(action)
    log.PARAMETER_VALUE("action",      action)
    log.PARAMETER_VALUE("action_name", NOMS_ACTIONS.get(action))
    log.FINISH_CALL_MANAGER_FUNCTION(
        "API", "model.predict", f"action={action}"
    )

    log.FINISH_ACTION("API", "/action", f"→ {action} {NOMS_ACTIONS[action]}")

    return {
        "action"     : action,
        "action_name": NOMS_ACTIONS.get(action, "inconnu"),
    }


def ___EPISODE(): pass
# ####################################################################
# POST /episode
# ####################################################################
@app.post("/episode")
def jouer_episode(req: EpisodeRequest):
    """
    Exécute un épisode complet.
    Retourne frames base64, récompenses, actions, observations et métriques.
    Toute la logique RL reste côté API.
    """
    log.START_ACTION("API", "/episode", f"seed={req.seed} render={req.render}")
    _verifier_modele()

    # ----------------------------------------------------------
    # Création de l'environnement
    # ----------------------------------------------------------
    log.START_CALL_CONTROLLER_FUNCTION(
        "API", "jouer_episode", "création environnement"
    )
    render_mode = "rgb_array" if req.render else None
    env         = gym.make("LunarLander-v3", render_mode=render_mode)
    obs, _      = env.reset(seed=req.seed)
    log.PARAMETER_VALUE("render_mode", render_mode)
    log.PARAMETER_VALUE("seed",        req.seed)
    log.FINISH_CALL_CONTROLLER_FUNCTION(
        "API", "jouer_episode", "environnement prêt"
    )

    # ----------------------------------------------------------
    # Collecte
    # ----------------------------------------------------------
    global _dernier_episode_frames, _dernier_episode_seed
    frames_b64      = []
    frames_np       = []    # frames numpy conservées pour /episode/video
    rewards_cumules = []
    actions_list    = []
    obs_list        = []
    reward_cumule   = 0.0
    terminated      = False
    truncated       = False
    n_pas           = 0

    log.START_CALL_MANAGER_FUNCTION(
        "API", "boucle_episode", "exécution pas à pas"
    )

    while not (terminated or truncated):

        # Capturer la frame AVANT l'action
        if req.render:
            # log.START_CALL_ENTITY_FUNCTION("API", "render", f"frame {n_pas}")
            frame = env.render()
            frames_b64.append(_frame_vers_base64(frame))
            frames_np.append(frame)                        # ← stockage numpy
            # log.FINISH_CALL_ENTITY_FUNCTION( "API", "render", f"frame {n_pas} encodée" )

        # Prédiction déterministe
        # log.START_CALL_ENTITY_FUNCTION("API", "model.predict", f"pas {n_pas}")
        action, _                             = _etat["model"].predict(
                                                  obs, deterministic=True)
        action                                = int(action)
        obs, reward, terminated, truncated, _ = env.step(action)

        # log.PARAMETER_VALUE("action",     f"{action} {NOMS_ACTIONS[action]}")
        # log.PARAMETER_VALUE("reward",     round(float(reward), 3))
        # log.PARAMETER_VALUE("terminated", terminated)
        # log.FINISH_CALL_ENTITY_FUNCTION(  "API", "model.predict", f"action={action}" )

        reward_cumule  += float(reward)
        rewards_cumules.append(round(reward_cumule, 3))
        actions_list.append(action)
        obs_list.append([round(float(v), 5) for v in obs])
        n_pas += 1

        log.PARAMETER_VALUE("reward action",     f"{n_pas} {reward_cumule} {action} {NOMS_ACTIONS[action]}")

    env.close()

    # Conserver les frames numpy pour /episode/video
    _dernier_episode_frames = frames_np
    _dernier_episode_seed   = req.seed or 0

    log.FINISH_CALL_MANAGER_FUNCTION(
        "API", "boucle_episode",
        f"{n_pas} pas — reward={reward_cumule:.2f}"
    )

    log.STEP(2, "résumé épisode")
    log.PARAMETER_VALUE("n_pas",        n_pas)
    log.PARAMETER_VALUE("total_reward", round(reward_cumule, 2))
    log.PARAMETER_VALUE("success",      reward_cumule >= 200.0)
    log.PARAMETER_VALUE("n_frames",     len(frames_b64))

    log.FINISH_ACTION(
        "API", "/episode",
        f"reward={reward_cumule:.1f} success={reward_cumule >= 200.0}"
    )

    return {
        "frames"      : frames_b64,
        "rewards"     : rewards_cumules,
        "actions"     : actions_list,
        "obs"         : obs_list,
        "total_reward": round(reward_cumule, 2),
        "n_steps"     : n_pas,
        "success"     : reward_cumule >= 200.0,
        "seed"        : req.seed,
    }


def ___MODELS_LIST(): pass
# ####################################################################
# GET /models/list — liste des modeles disponibles
# ####################################################################
@app.get("/models/list")
def models_list():
    """
    Inventaire de tous les modeles disponibles dans models/.
    Retourne la liste avec leurs metadonnees pour le selecteur du GUI.
    """
    log.START_ACTION("API", "/models/list", "inventaire modeles")

    modeles = []

    # ----------------------------------------------------------
    # Fonction interne — lire le JSON d un modele
    # ----------------------------------------------------------
    def _lire_meta(chemin_zip, id_modele, algo_defaut):
        chemin_json  = chemin_zip.with_suffix('.json')
        mean_reward  = None
        critere      = False
        algo_lu      = algo_defaut
        if chemin_json.exists():
            try:
                with open(chemin_json, encoding='utf-8') as fj:
                    meta    = json.load(fj)
                mean_reward = meta.get('resultats', {}).get('mean_reward')
                critere     = meta.get('resultats', {}).get('critere_valide', False)
                algo_lu     = meta.get('algorithme', algo_defaut)
            except Exception:
                pass
        return {
            "id"            : id_modele,
            "chemin"        : str(chemin_zip.relative_to(MODELS)),
            "algorithme"    : algo_lu,
            "mean_reward"   : mean_reward,
            "critere_valide": critere,
            "actif"         : (
                _etat["chemin_modele"] is not None and
                chemin_zip.resolve() == _etat["chemin_modele"].resolve()
            ),
        }

    # ----------------------------------------------------------
    # Scanner les modeles connus
    # ----------------------------------------------------------
    log.START_CALL_CONTROLLER_FUNCTION(
        "API", "models_list", "scan MODELS/"
    )

    # best_ppo
    c = MODELS / 'best_ppo' / 'best_model.zip'
    if c.exists():
        modeles.append(_lire_meta(c, "best_ppo", "PPO"))
        log.PARAMETER_VALUE("best_ppo", modeles[-1]['mean_reward'])

    # best_dqn
    c = MODELS / 'best_dqn' / 'best_model.zip'
    if c.exists():
        modeles.append(_lire_meta(c, "best_dqn", "DQN"))
        log.PARAMETER_VALUE("best_dqn", modeles[-1]['mean_reward'])

    # ppo_lunarlander_*.zip
    for z in sorted(MODELS.glob('*ppo_lunarlander_*.zip'), reverse=True):
        modeles.append(_lire_meta(z, z.stem, "PPO"))
        log.PARAMETER_VALUE(f"ppo {z.stem}", modeles[-1]['mean_reward'])

    # dqn_lunarlander_*.zip
    for z in sorted(MODELS.glob('*dqn_lunarlander_*.zip'), reverse=True):
        modeles.append(_lire_meta(z, z.stem, "DQN"))
        log.PARAMETER_VALUE(f"dqn {z.stem}", modeles[-1]['mean_reward'])

    log.PARAMETER_VALUE("total modeles", len(modeles))
    log.FINISH_CALL_CONTROLLER_FUNCTION(
        "API", "models_list", f"{len(modeles)} modeles"
    )
    log.FINISH_ACTION("API", "/models/list", f"{len(modeles)} modeles trouves")

    return {"modeles": modeles}


def ___MODELS_LOAD(): pass
# ####################################################################
# POST /models/load — charger un modele par son id
# ####################################################################
@app.post("/models/load")
def models_load(req: LoadRequest):
    """
    Charge un modele en memoire par son id (retourne par /models/list).
    Remplace le modele actif sans redemarrer l API.
    """
    log.START_ACTION("API", "/models/load", f"id={req.id}")

    # ----------------------------------------------------------
    # Resoudre le chemin depuis l id
    # ----------------------------------------------------------
    log.START_CALL_CONTROLLER_FUNCTION(
        "API", "models_load", f"resolution id={req.id}"
    )

    if req.id == "best_ppo":
        chemin = MODELS / 'best_ppo' / 'best_model.zip'
        algo_c = 'PPO'
    elif req.id == "best_dqn":
        chemin = MODELS / 'best_dqn' / 'best_model.zip'
        algo_c = 'DQN'
    else:
        candidats = list(MODELS.glob(f'{req.id}.zip'))
        if not candidats:
            log.LEVEL_4_ERROR("API", f"modele introuvable : {req.id}")
            raise HTTPException(
                status_code = 404,
                detail      = f"Modele '{req.id}' introuvable dans models/"
            )
        chemin = candidats[0]
        algo_c = 'DQN' if 'dqn' in req.id.lower() else 'PPO'

    if not chemin.exists():
        log.LEVEL_4_ERROR("API", f"fichier absent : {chemin}")
        raise HTTPException(
            status_code = 404,
            detail      = f"Fichier {chemin} introuvable"
        )

    log.PARAMETER_VALUE("chemin", str(chemin))
    log.PARAMETER_VALUE("algo",   algo_c)
    log.FINISH_CALL_CONTROLLER_FUNCTION(
        "API", "models_load", "chemin resolu"
    )

    # ----------------------------------------------------------
    # Charger le modele SB3 en memoire
    # ----------------------------------------------------------
    log.START_CALL_MANAGER_FUNCTION(
        "API", "SB3.load", f"{algo_c}.load({chemin.name})"
    )
    nouveau_model = (PPO if algo_c == 'PPO' else DQN).load(str(chemin))
    log.FINISH_CALL_MANAGER_FUNCTION("API", "SB3.load", "modele charge")

    # ----------------------------------------------------------
    # Lire les metadonnees JSON
    # ----------------------------------------------------------
    chemin_json  = chemin.with_suffix('.json')
    nouveau_meta = {}
    if chemin_json.exists():
        with open(chemin_json, encoding='utf-8') as fj:
            nouveau_meta = json.load(fj)
        log.PARAMETER_VALUE("metadata JSON", chemin_json.name)

    # ----------------------------------------------------------
    # Mettre a jour l etat global
    # ----------------------------------------------------------
    _etat["model"]         = nouveau_model
    _etat["algo"]          = algo_c
    _etat["chemin_modele"] = chemin
    _etat["metadata"]      = nouveau_meta

    log.PARAMETER_VALUE("modele actif", chemin.name)
    log.PARAMETER_VALUE("algorithme",   algo_c)
    log.FINISH_ACTION("API", "/models/load", f"{algo_c} {chemin.name} charge")

    return {
        "status"     : "ok",
        "id"         : req.id,
        "modele"     : chemin.name,
        "algorithme" : algo_c,
        "mean_reward": nouveau_meta.get('resultats', {}).get('mean_reward'),
    }


def ___EPISODE_VIDEO(): pass
# ####################################################################
# GET /episode/video — MP4 du dernier épisode joué
# ####################################################################
@app.get("/episode/video")
def get_episode_video():
    """
    Assemble et retourne le MP4 du dernier épisode joué via /episode.
    Le GUI appelle /episode d'abord (animation interactive),
    puis /episode/video uniquement si l'utilisateur demande la vidéo.
    Retourne un fichier MP4 binaire (Response media_type video/mp4).
    """
    log.START_ACTION("API", "/episode/video", "assemblage MP4 dernier épisode")
    _verifier_modele()

    # ----------------------------------------------------------
    # Vérifier qu'un épisode a été joué avec render=True
    # ----------------------------------------------------------
    if not _dernier_episode_frames:
        log.LEVEL_5_WARNING("API", "aucun épisode avec frames disponible")
        log.FINISH_ACTION("API", "/episode/video", "aucune frame")
        raise HTTPException(
            status_code = 404,
            detail      = "Aucun épisode joué avec render=True. "
                          "Appeler /episode avec render:true d'abord."
        )

    log.PARAMETER_VALUE("n_frames", len(_dernier_episode_frames))
    log.PARAMETER_VALUE("seed",     _dernier_episode_seed)

    # ----------------------------------------------------------
    # Assemblage MP4 via imageio-ffmpeg
    # ----------------------------------------------------------
    log.START_CALL_MANAGER_FUNCTION("API", "assembler_mp4", "imageio-ffmpeg")
    try:

        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as tmp:
            tmp_path = tmp.name

        h, w = _dernier_episode_frames[0].shape[:2]
        writer = imageio_ffmpeg.write_frames(
            tmp_path,
            (w, h),                       # (width, height)
            fps              = 30,
            ffmpeg_log_level = "quiet",
        )
        writer.send(None)                 # initialiser le générateur
        for frame in _dernier_episode_frames:
            writer.send(frame)
        writer.close()

        with open(tmp_path, 'rb') as f:
            video_bytes = f.read()
        os.unlink(tmp_path)

        log.PARAMETER_VALUE("taille_mp4", f"{len(video_bytes):,} bytes")
        log.FINISH_CALL_MANAGER_FUNCTION(
            "API", "assembler_mp4", f"{len(video_bytes):,} bytes"
        )

    except Exception as e:
        log.LEVEL_4_ERROR("API", f"assemblage MP4 échoué : {e}")
        log.FINISH_CALL_MANAGER_FUNCTION("API", "assembler_mp4", "ERREUR")
        raise HTTPException(
            status_code = 500,
            detail      = f"Assemblage MP4 échoué : {e}"
        )

    log.FINISH_ACTION(
        "API", "/episode/video",
        f"MP4 {len(video_bytes):,} bytes — seed={_dernier_episode_seed}"
    )

    # Retourner le MP4 binaire directement
    return Response(
        content      = video_bytes,
        media_type   = "video/mp4",
        headers      = {
            "Content-Disposition":
                f"inline; filename=eagle1_episode_{_dernier_episode_seed}.mp4"
        }
    )

def ___MODEL_INFO(): pass
    
# ####################################################################
# GET /model/info
# ####################################################################
@app.get("/model/info")
def model_info():
    """Métadonnées complètes du modèle : hyperparamètres, résultats, comparaisons."""
    log.START_ACTION("API", "/model/info", "lecture métadonnées")
    _verifier_modele()

    if not _etat["metadata"]:
        log.LEVEL_5_WARNING("API", "pas de fichier JSON de métadonnées")
        log.FINISH_ACTION("API", "/model/info", "sans metadata")
        return {
            "message"   : "Pas de métadonnées JSON disponibles",
            "modele"    : str(_etat["chemin_modele"].name),
            "algorithme": _etat["algo"],
        }

    log.PARAMETER_VALUE("projet",     _etat["metadata"].get('projet'))
    log.PARAMETER_VALUE("algorithme", _etat["metadata"].get('algorithme'))
    log.PARAMETER_VALUE("mean_reward",
        _etat["metadata"].get('resultats', {}).get('mean_reward'))
    log.FINISH_ACTION("API", "/model/info", "OK")

    return _etat["metadata"]


# ============================================================
# État global — environnement interactif (physique réelle)
# L'env reste OUVERT entre les appels /interactif/step
# pour que la gravité et la physique s'appliquent vraiment.
# ============================================================
_env_interactif: dict = {
    "env"         : None,   # instance gym ouverte
    "obs"         : None,   # observation courante float32[8]
    "reward_cum"  : 0.0,    # récompense cumulée
    "n_pas"       : 0,      # compteur de pas
    "terminated"  : False,  # épisode terminé normalement
    "truncated"   : False,  # épisode tronqué (limite de pas)
    "seed"        : 0,      # graine de l'épisode courant
}


def ___INTERCTIF_START(): pass
# ####################################################################
# POST /interactif/start — démarrer un épisode interactif
# ####################################################################
@app.post("/interactif/start")
def interactif_start(req: EpisodeRequest):
    """
    Ouvre un nouvel environnement LunarLander-v3 et retourne
    l'observation initiale + la première frame RGB.
    L'environnement reste ouvert — la physique (gravité, inertie)
    est réelle à chaque pas via /interactif/step.
    """
    log.START_ACTION("API", "/interactif/start", f"seed={req.seed}")

    # ----------------------------------------------------------
    # Fermer l'épisode précédent si encore ouvert
    # ----------------------------------------------------------
    if _env_interactif["env"] is not None:
        log.STEP(2, "fermeture env précédent")
        _env_interactif["env"].close()
        _env_interactif["env"] = None

    # ----------------------------------------------------------
    # Créer et réinitialiser l'environnement
    # ----------------------------------------------------------
    log.START_CALL_CONTROLLER_FUNCTION(
        "API", "interactif_start", "gym.make LunarLander-v3"
    )
    env    = gym.make("LunarLander-v3", render_mode="rgb_array")
    obs, _ = env.reset(seed=req.seed)
    frame  = env.render()

    # Stocker dans l'état global
    _env_interactif["env"]       = env
    _env_interactif["obs"]       = obs
    _env_interactif["reward_cum"]= 0.0
    _env_interactif["n_pas"]     = 0
    _env_interactif["terminated"]= False
    _env_interactif["truncated"] = False
    _env_interactif["seed"]      = req.seed or 0

    log.PARAMETER_VALUE("seed",            req.seed)
    log.PARAMETER_VALUE("obs shape",       obs.shape)
    log.FINISH_CALL_CONTROLLER_FUNCTION(
        "API", "interactif_start", "env prêt"
    )
    log.FINISH_ACTION("API", "/interactif/start", "OK")

    return {
        "obs"        : [round(float(v), 5) for v in obs],
        "frame_b64"  : _frame_vers_base64(frame),
        "reward_cum" : 0.0,
        "n_pas"      : 0,
        "done"       : False,
        "seed"       : req.seed,
    }


# ####################################################################
# POST /interactif/step — exécuter un pas avec l'action du joueur
# ####################################################################


def ___INTERCTIF_STEP(): pass

@app.post("/interactif/step")
def interactif_step(req: StepRequest):
    """
    Applique l'action du joueur dans l'environnement ouvert.
    La physique réelle (gravité, propulseurs, inertie) s'applique.
    Retourne la nouvelle observation, frame RGB, récompense et done.
    """
    log.START_ACTION(
        "API", "/interactif/step",
        f"action={req.action} pas={_env_interactif['n_pas']}"
    )

    # ----------------------------------------------------------
    # Vérifier que l'environnement est ouvert
    # ----------------------------------------------------------
    if _env_interactif["env"] is None:
        log.LEVEL_4_ERROR("API", "aucun épisode interactif en cours")
        raise HTTPException(
            status_code = 400,
            detail      = "Aucun épisode interactif en cours. "
                          "Appeler /interactif/start d'abord."
        )

    if _env_interactif["terminated"] or _env_interactif["truncated"]:
        log.LEVEL_5_WARNING("API", "épisode déjà terminé")
        raise HTTPException(
            status_code = 400,
            detail      = "Épisode terminé. Appeler /interactif/start "
                          "pour recommencer."
        )

    # Valider l'action
    if req.action not in (0, 1, 2, 3):
        raise HTTPException(
            status_code = 422,
            detail      = f"Action invalide : {req.action}. "
                          f"Valeurs acceptées : 0 1 2 3"
        )

    # ----------------------------------------------------------
    # Exécuter le pas dans l'environnement (physique réelle)
    # ----------------------------------------------------------
    log.START_CALL_MANAGER_FUNCTION(
        "API", "env.step", f"action={req.action}"
    )
    env = _env_interactif["env"]

    obs, reward, terminated, truncated, _ = env.step(req.action)
    frame = env.render()

    _env_interactif["obs"]        = obs
    _env_interactif["reward_cum"] += float(reward)
    _env_interactif["n_pas"]      += 1
    _env_interactif["terminated"] = terminated
    _env_interactif["truncated"]  = truncated

    done = terminated or truncated

    log.PARAMETER_VALUE(
        "action", f"{req.action} {NOMS_ACTIONS[req.action]}"
    )
    log.PARAMETER_VALUE("reward",      round(float(reward), 3))
    log.PARAMETER_VALUE("reward_cum",  round(_env_interactif["reward_cum"], 2))
    log.PARAMETER_VALUE("terminated",  terminated)
    log.PARAMETER_VALUE("truncated",   truncated)
    log.FINISH_CALL_MANAGER_FUNCTION(
        "API", "env.step", f"done={done}"
    )

    # Fermer l'env si épisode terminé
    if done:
        log.STEP(2, "épisode terminé — fermeture env")
        env.close()
        _env_interactif["env"] = None

    log.FINISH_ACTION(
        "API", "/interactif/step",
        f"pas={_env_interactif['n_pas']} done={done}"
    )

    return {
        "obs"        : [round(float(v), 5) for v in obs],
        "frame_b64"  : _frame_vers_base64(frame),
        "reward"     : round(float(reward), 3),
        "reward_cum" : round(_env_interactif["reward_cum"], 2),
        "n_pas"      : _env_interactif["n_pas"],
        "done"       : done,
        "success"    : _env_interactif["reward_cum"] >= 200.0,
        "action_name": NOMS_ACTIONS[req.action],
    }


def ___INTERCTIF_RESET(): pass
        
# ####################################################################
# POST /interactif/reset — réinitialiser l'épisode interactif
# ####################################################################
@app.post("/interactif/reset")
def interactif_reset(req: EpisodeRequest):
    """
    Ferme l'épisode en cours et en démarre un nouveau.
    Alias de /interactif/start — conservé pour clarté sémantique.
    """
    log.START_ACTION("API", "/interactif/reset", f"seed={req.seed}")
    log.FINISH_ACTION("API", "/interactif/reset", "→ /interactif/start")
    return interactif_start(req)


# ============================================================
# Point d'entrée direct
# ============================================================
if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=False)
