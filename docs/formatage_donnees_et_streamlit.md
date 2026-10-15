# Complément livrable — Formatage des données et choix de Streamlit

Ce document complète le notebook principal de la mission Eagle-1. Il répond explicitement aux deux points demandés lors de la revue :

1. documenter le formatage des données échangées entre l'environnement, l'API et le modèle ;
2. justifier le choix de Streamlit comme système d'exposition et de visualisation des résultats.

Le contenu est également repris dans la nouvelle version du notebook :

```text
notebooks/Notebook_mission_LunarLander_V3_livrable_corrige.ipynb
```

---

## 1. Formatage des données

### 1.1 Observation LunarLander-v3

L'environnement `LunarLander-v3` expose un espace d'observation continu :

```python
Box(8,)
```

Chaque observation est un vecteur de 8 valeurs numériques. L'ordre est conservé de bout en bout : Gymnasium → JSON API → NumPy → modèle Stable-Baselines3.

| Index | Nom | Description | Type JSON reçu par l'API | Type utilisé pour le modèle |
|------:|-----|-------------|---------------------------|-----------------------------|
| 0 | `x` | position horizontale de l'atterrisseur | `float` | `np.float32` |
| 1 | `y` | position verticale | `float` | `np.float32` |
| 2 | `vx` | vitesse horizontale | `float` | `np.float32` |
| 3 | `vy` | vitesse verticale | `float` | `np.float32` |
| 4 | `theta` | angle d'inclinaison | `float` | `np.float32` |
| 5 | `dtheta` | vitesse angulaire | `float` | `np.float32` |
| 6 | `contact_gauche` | contact du pied gauche avec le sol | `0.0` ou `1.0` | `np.float32` |
| 7 | `contact_droit` | contact du pied droit avec le sol | `0.0` ou `1.0` | `np.float32` |

Dans l'API, la validation de longueur est explicite :

```python
if len(req.state) != 8:
    raise HTTPException(status_code=422, detail="State doit contenir 8 valeurs")
```

Puis la conversion est effectuée avant l'inférence :

```python
obs = np.array(req.state, dtype=np.float32)
action, _ = model.predict(obs, deterministic=True)
```

Cette conversion garantit que le format utilisé par l'API correspond au format attendu par le modèle entraîné avec Stable-Baselines3.

---

### 1.2 Représentation des actions

L'espace d'action est discret :

```python
Discrete(4)
```

L'action renvoyée par le modèle est donc un entier dans `{0, 1, 2, 3}`.

| Action | Nom renvoyé par l'API | Signification |
|------:|------------------------|---------------|
| 0 | `do nothing` | aucune poussée moteur |
| 1 | `moteur gauche` | activation du moteur latéral gauche |
| 2 | `moteur principal` | poussée verticale principale |
| 3 | `moteur droit` | activation du moteur latéral droit |

Dans `src/api.py`, cette correspondance est centralisée dans `NOMS_ACTIONS`, ce qui permet à l'API et au tableau de bord de présenter les décisions sous une forme lisible.

---

### 1.3 Récompense instantanée et récompense cumulée

L'environnement Gymnasium retourne à chaque pas une récompense instantanée :

```python
obs, reward, terminated, truncated, _ = env.step(action)
```

Dans l'API, cette récompense est accumulée pour obtenir une métrique d'épisode :

```python
reward_cumule += float(reward)
rewards_cumules.append(round(reward_cumule, 3))
```

| Notion | Source | Niveau d'analyse | Usage dans le projet |
|--------|--------|------------------|----------------------|
| Récompense instantanée | valeur `reward` retournée par `env.step(action)` | pas de temps individuel | signal local de l'environnement |
| Récompense cumulée | somme des récompenses instantanées depuis le début de l'épisode | épisode complet | courbe du dashboard, comparaison des épisodes, critère de succès |

Le critère de succès de la mission est évalué sur la récompense cumulée :

```python
success = reward_cumule >= 200.0
```

---

### 1.4 Exemples de requêtes et réponses JSON

#### Endpoint `POST /action`

Requête :

```json
{
  "state": [0.0, 1.2, 0.03, -0.15, 0.01, 0.0, 0.0, 0.0]
}
```

Réponse :

```json
{
  "action": 2,
  "action_name": "moteur principal"
}
```

#### Endpoint `POST /episode`

Requête :

```json
{
  "seed": 42,
  "render": true
}
```

Réponse simplifiée :

```json
{
  "frames": ["base64_png_1", "base64_png_2"],
  "rewards": [1.2, 2.7, 4.1],
  "actions": [2, 2, 0],
  "obs": [[0.0, 1.2, 0.03, -0.15, 0.01, 0.0, 0.0, 0.0]],
  "total_reward": 225.9,
  "n_steps": 314,
  "success": true,
  "seed": 42
}
```

---

### 1.5 Flux de données

```text
Gymnasium obs[8]
      ↓
JSON API : liste de 8 floats
      ↓
FastAPI /action
      ↓
np.array(..., dtype=np.float32)
      ↓
Stable-Baselines3 model.predict(...)
      ↓
action int ∈ {0,1,2,3}
      ↓
env.step(action)
      ↓
reward instantanée + observation suivante
      ↓
reward cumulée + visualisation Streamlit
```

---

## 2. Justification du choix de Streamlit

Streamlit a été choisi comme couche d'exposition des résultats, en complément de l'API FastAPI. Le rôle de FastAPI est de conserver la logique métier RL côté serveur : chargement du modèle, inférence, exécution d'épisodes et sérialisation JSON. Le rôle de Streamlit est de rendre ces résultats explorables par un utilisateur non technique.

### 2.1 Besoins du projet

| Besoin | Réponse apportée par Streamlit |
|--------|--------------------------------|
| Visualiser un épisode complet | affichage des frames, slider de pas, animation et vidéo |
| Comprendre les décisions de l'agent | histogramme des actions, télémétrie pas à pas, nom lisible des actions |
| Suivre la performance | courbe de récompense cumulée, métriques `total_reward`, `n_steps`, `success` |
| Comparer les modèles | page modèle avec informations PPO/DQN et résultats d'évaluation |
| Interagir sans relancer le notebook | boutons, sidebar, champs de configuration de l'URL API |
| Séparer backend et frontend | appels HTTP `requests` vers FastAPI |

---

### 2.2 Comparaison avec d'autres solutions

| Critère | Streamlit | Dash | Gradio | Notebook seul |
|--------|-----------|------|--------|---------------|
| Rapidité de développement | très rapide | moyenne | rapide | déjà disponible |
| Dashboard multi-composants | très adapté | très adapté | plus limité | peu adapté |
| Animation et exploration d'épisodes | slider, images, vidéo | possible mais plus verbeux | possible mais moins dashboard | dépend de l'exécution cellule |
| Communication avec FastAPI | simple via `requests` | simple aussi | possible | possible mais moins applicatif |
| Lisibilité pour un jury/mentor | forte | forte mais plus technique | forte pour démo modèle | moins proche d'un outil utilisateur |
| Maintenance | un fichier Python principal (`src/gui.py`) | architecture plus lourde | simple mais moins flexible | non destiné à être une interface finale |

Streamlit représente donc un compromis adapté au livrable : il permet d'exposer rapidement les résultats du modèle tout en conservant une architecture claire et testable.

---

### 2.3 Séparation des responsabilités

| Composant | Responsabilité | Fichier principal |
|-----------|----------------|-------------------|
| Environnement Gymnasium | simulation physique LunarLander | `src/api.py` |
| Modèle Stable-Baselines3 | décision état → action | `models/*.zip` |
| API FastAPI | validation, inférence, épisodes, JSON | `src/api.py` |
| Interface Streamlit | visualisation, interaction, dashboard | `src/gui.py` |
| Tests | validation des endpoints | `tests/test_api.py` |

Cette séparation permet de démontrer deux compétences complémentaires :

- formatage et exposition fiable des données via une API ;
- restitution interactive et pédagogique des résultats via un tableau de bord.
