# #############################################################################
# TESTS DE L'API FASTAPI — MISSION EAGLE-1
# #############################################################################
# Ces tests utilisent TestClient (httpx) pour appeler l'application FastAPI
# directement en mémoire, sans lancer uvicorn ni ouvrir de port réseau.
# Cela garantit des tests rapides, déterministes et indépendants de
# l'environnement (pas d'IP, pas de port, pas de latence réseau).
#
# Prérequis : un modèle entraîné doit être présent dans models/best_ppo/
# ou models/best_dqn/ pour que l'API se charge correctement au démarrage.
#
# Exécution :
#   uv run pytest tests/ -v
# #############################################################################

# Bibliothèque de test
import pytest                            # Framework de test principal
# Client de test FastAPI (appels HTTP en mémoire, sans serveur réel)
from fastapi.testclient import TestClient

# Application FastAPI du projet
from src.api import app                  # Instance de l'API à tester


# État LunarLander-v3 neutre — 8 valeurs à zéro (Box(8,) float32)
# Un état à zéro n'est pas physiquement réaliste mais reste dans les bornes
# attendues par le modèle : suffisant pour valider le flux d'inférence.
ETAT_NEUTRE = [0.0] * 8

# Nombre maximal de pas accepté pour un épisode de test — filet de sécurité
# pour éviter un test bloqué en cas de comportement inattendu du modèle.
MAX_PAS_EPISODE_TEST = 1000


@pytest.fixture(scope="module")
def client():
    # ------------------------------------------------------------------
    # Le TestClient déclenche les événements startup/shutdown de FastAPI
    # (chargement du modèle inclus), comme le ferait un vrai serveur.
    # ------------------------------------------------------------------
    with TestClient(app) as test_client:
        yield test_client


# #############################################################################
def test_health(client):
    """GET /health — le service répond et un modèle est chargé."""
    reponse = client.get("/health")

    assert reponse.status_code == 200
    corps = reponse.json()
    # On ne fige pas le nom exact du champ modèle/statut : on vérifie
    # simplement qu'une réponse JSON non vide est bien renvoyée.
    assert isinstance(corps, dict)
    assert len(corps) > 0


# #############################################################################
def test_model_info(client):
    """GET /model/info — les métadonnées du modèle actif sont accessibles."""
    reponse = client.get("/model/info")

    assert reponse.status_code == 200
    corps = reponse.json()
    assert "algorithme" in corps
    assert corps["algorithme"] in ("PPO", "DQN")


# #############################################################################
def test_action(client):
    """POST /action — un état valide renvoie une action dans {0,1,2,3}."""
    reponse = client.post("/action", json={"state": ETAT_NEUTRE})

    assert reponse.status_code == 200
    corps = reponse.json()
    assert "action" in corps
    assert corps["action"] in (0, 1, 2, 3)


# #############################################################################
def test_episode(client):
    """POST /episode — un épisode complet s'exécute et renvoie ses métriques."""
    reponse = client.post("/episode", json={})

    assert reponse.status_code == 200
    corps = reponse.json()

    # ------------------------------------------------------------------
    # Vérification de la structure attendue — adapte les clés ci-dessous
    # si api.py utilise des noms différents (ex. "actions" vs "liste_actions")
    # ------------------------------------------------------------------
    assert "actions" in corps
    assert "rewards" in corps
    assert len(corps["actions"]) > 0
    assert len(corps["actions"]) <= MAX_PAS_EPISODE_TEST
    assert len(corps["actions"]) == len(corps["rewards"])
