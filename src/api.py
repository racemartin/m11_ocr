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
from pathlib import Path             # chemins portables
from typing  import List, Optional   # annotations de types

# — Web framework
import uvicorn                       # serveur ASGI
from fastapi             import FastAPI, HTTPException  # framework API
from fastapi.middleware.cors import CORSMiddleware      # CORS pour Streamlit
from pydantic            import BaseModel              # validation données

# — Calcul vectoriel et image
import numpy             as np       # tableaux numériques
from PIL                 import Image  # conversion frame numpy → PNG

# — Environnement et modèle RL
import gymnasium         as gym      # API Gymnasium 1.3+
from stable_baselines3   import PPO, DQN  # algorithmes SB3

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


log.START_ACTION("API", "startup", "chargement du modèle au démarrage")

chemin_modele, algo, chemin_json = _trouver_meilleur_modele()

if chemin_modele is None:
    log.LEVEL_5_WARNING("API", "API en mode dégradé — aucun modèle disponible")
    model    = None
    metadata = {}
else:
    log.STEP(2, "Chargement modèle SB3")
    log.PARAMETER_VALUE("chemin",     str(chemin_modele))
    log.PARAMETER_VALUE("algorithme", algo)
    model    = (PPO if algo == 'PPO' else DQN).load(str(chemin_modele))
    metadata = {}
    if chemin_json and chemin_json.exists():
        with open(chemin_json, encoding='utf-8') as f:
            metadata = json.load(f)
        log.PARAMETER_VALUE("metadata JSON", chemin_json.name)

log.FINISH_ACTION("API", "startup", f"modèle {algo} prêt")

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

# ============================================================
# Modèles Pydantic
# ============================================================

class EtatRequest(BaseModel):
    state: List[float]

class EpisodeRequest(BaseModel):
    seed:   Optional[int]  = 42
    render: Optional[bool] = True

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
    if model is None:
        log.LEVEL_4_ERROR("API", "modèle non disponible — requête rejetée")
        raise HTTPException(
            status_code = 503,
            detail      = "Aucun modèle disponible. Entraîner un agent d'abord."
        )

# ============================================================
# Endpoints
# ============================================================

# ####################################################################
# GET /health
# ####################################################################
@app.get("/health")
def health():
    """État du service et infos du modèle chargé."""
    log.START_ACTION("API", "/health", "vérification état du service")

    resultat = {
        "status"        : "ok" if model is not None else "degraded",
        "modele"        : str(chemin_modele.name) if chemin_modele else None,
        "algorithme"    : algo,
        "mean_reward"   : metadata.get('resultats', {}).get('mean_reward'),
        "critere_valide": metadata.get('resultats', {}).get('critere_valide'),
        "environnement" : metadata.get('environnement', 'LunarLander-v3'),
    }

    log.PARAMETER_VALUE("status",      resultat["status"])
    log.PARAMETER_VALUE("modele",      resultat["modele"])
    log.PARAMETER_VALUE("mean_reward", resultat["mean_reward"])
    log.FINISH_ACTION("API", "/health", resultat["status"])

    return resultat


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
    action, _ = model.predict(obs, deterministic=True)
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
    frames_b64      = []
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
            log.START_CALL_ENTITY_FUNCTION("API", "render", f"frame {n_pas}")
            frame = env.render()
            frames_b64.append(_frame_vers_base64(frame))
            log.FINISH_CALL_ENTITY_FUNCTION(
                "API", "render", f"frame {n_pas} encodée"
            )

        # Prédiction déterministe
        log.START_CALL_ENTITY_FUNCTION("API", "model.predict", f"pas {n_pas}")
        action, _                             = model.predict(
                                                  obs, deterministic=True)
        action                                = int(action)
        obs, reward, terminated, truncated, _ = env.step(action)

        log.PARAMETER_VALUE("action",     f"{action} {NOMS_ACTIONS[action]}")
        log.PARAMETER_VALUE("reward",     round(float(reward), 3))
        log.PARAMETER_VALUE("terminated", terminated)
        log.FINISH_CALL_ENTITY_FUNCTION(
            "API", "model.predict", f"action={action}"
        )

        reward_cumule  += float(reward)
        rewards_cumules.append(round(reward_cumule, 3))
        actions_list.append(action)
        obs_list.append([round(float(v), 5) for v in obs])
        n_pas += 1

    env.close()

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


# ####################################################################
# GET /model/info
# ####################################################################
@app.get("/model/info")
def model_info():
    """Métadonnées complètes du modèle : hyperparamètres, résultats, comparaisons."""
    log.START_ACTION("API", "/model/info", "lecture métadonnées")
    _verifier_modele()

    if not metadata:
        log.LEVEL_5_WARNING("API", "pas de fichier JSON de métadonnées")
        log.FINISH_ACTION("API", "/model/info", "sans metadata")
        return {
            "message"   : "Pas de métadonnées JSON disponibles",
            "modele"    : str(chemin_modele.name),
            "algorithme": algo,
        }

    log.PARAMETER_VALUE("projet",     metadata.get('projet'))
    log.PARAMETER_VALUE("algorithme", metadata.get('algorithme'))
    log.PARAMETER_VALUE("mean_reward",
        metadata.get('resultats', {}).get('mean_reward'))
    log.FINISH_ACTION("API", "/model/info", "OK")

    return metadata


# ============================================================
# Point d'entrée direct
# ============================================================
if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=False)
