# Student Mental Health Risk — Projet Data Science

> **Projet pédagogique individuel · Jury blanc · 9–14 novembre 2026**  
> **Date limite : samedi 14/11/2026 à 23 h 59 · Soutenance : lundi 16/11/2026**

## 1. Contexte et périmètre

Un établissement d'enseignement souhaite explorer les facteurs **associés** au bien-être psychologique des étudiants et expérimenter une classification supervisée de niveaux de risque à partir d'un jeu de données public. Le projet couvre l'audit des données, l'analyse exploratoire (EDA), le prétraitement, la comparaison et l'optimisation de modèles, l'évaluation indépendante, l'orchestration avec Apache Airflow et la présentation des résultats dans Streamlit.

> [!IMPORTANT]
> **Usage exclusivement pédagogique.** Les prédictions sont des estimations statistiques propres au jeu de données utilisé : elles ne sont **ni un diagnostic, ni un dépistage clinique**, et ne doivent jamais servir à prendre des décisions disciplinaires, de sélection, d'exclusion ou de traitement concernant des personnes. Aucune recommandation individualisée ne doit être déclenchée automatiquement par le modèle.

Les expressions « détection », « prévention » et « priorisation » décrivent ici des **objectifs de recherche et de visualisation agrégée**, pas un système opérationnel de tri des étudiants. Tout usage réel exigerait une validation externe, un encadrement éthique, une base légale et une supervision humaine qualifiée.

## 2. Données

- **Source :** [Mental Health Risk Prediction Dataset (Students) — Kaggle](https://www.kaggle.com/datasets/maqdadali/mental-health-risk-prediction-dataset-students)
- **Nature :** données tabulaires à auditer avant toute modélisation.
- **Variable cible :** `Mental_Health_Risk`, avec les classes `Low`, `Medium` et `High`. Le fichier ne décrit pas la méthode de création des étiquettes.
- **Licence, provenance, consentement et représentativité :** vérifier les informations réellement fournies par le créateur du dataset et documenter les inconnues.
- **Confidentialité :** éviter d'inclure des identifiants personnels, observations sensibles brutes ou copies publiques des données dans le dépôt Git.

### Points de vigilance à vérifier à J1

1. Taille du jeu de données, noms et types de colonnes, unités et cardinalités.
2. Définition de la cible : binaire ou multiclasse, répartition des classes, éventuelle génération artificielle / heuristique des étiquettes.
3. Valeurs manquantes, doublons et valeurs impossibles ; distinguer anomalies réelles et valeurs rares mais plausibles.
4. Colonnes identifiantes et **fuites de cible** (variables dérivées de la cible ou disponibles seulement après l'événement prédit).
5. Risques de biais, échantillonnage non représentatif et absence éventuelle de validation clinique.

## 3. Objectifs et livrables

| Axe | Objectif | Livrable attendu |
|---|---|---|
| Audit & EDA | Décrire la qualité des données, la cible et les associations | Notebook et rapport EDA avec figures |
| Préparation | Construire des transformations explicables et reproductibles | Pipeline `ColumnTransformer`/`Pipeline` testé |
| Modélisation | Comparer au moins 3 modèles sur un protocole équitable | Tableau de comparaison et validation croisée |
| Optimisation | Optimiser et évaluer sans réutiliser le test pour choisir le modèle | Meilleur pipeline, métriques, matrice de confusion, analyse d'erreurs |
| Orchestration | Exécuter les étapes reproductibles sous contrôle Airflow | DAG, planification, logs et gestion des échecs |
| Application | Montrer les résultats et simuler une prédiction anonyme | Tableau de bord Streamlit et démonstration |
| Documentation | Permettre à un examinateur de reproduire le travail | README, instructions d'installation, sources, limites et tests |

**Indicateurs de classification :** accuracy, précision, rappel, F1-score (notamment *macro-F1* en multiclasse ou déséquilibre), matrice de confusion ; ROC-AUC/PR-AUC uniquement lorsque leur définition est pertinente pour le type de cible et les scores de prédiction. Toujours publier le support par classe et, si possible, la dispersion de validation croisée. Aucun seuil de performance chiffré n'est garanti à l'avance.

## 4. Stack technique

| Besoin | Technologies prévues |
|---|---|
| Langage / environnement | Python 3.14 pour les notebooks et Docker, `uv`, Git |
| Manipulation | NumPy, pandas |
| Exploration / graphiques | Matplotlib, Seaborn, éventuellement Plotly |
| Modèles / pipeline | scikit-learn, imbalanced-learn si pertinent |
| Optimisation / suivi | GridSearchCV ou RandomizedSearchCV, JSON/CSV de métriques |
| Sérialisation | joblib (chargement depuis une source de confiance uniquement) |
| Orchestration | Apache Airflow sous Linux, WSL2 ou Docker |
| Interface | Streamlit |
| Tests | pytest et tests de fumée / reproductibilité |

### Environnement des notebooks

Depuis la racine du dépôt, `uv sync` installe l’environnement de `uv.lock`. L’image Docker commune utilise `requirements-ml.txt` pour le modèle et `app/requirements.txt` pour l’interface.

### Récupérer le dataset

```bash
# Authentification Kaggle configurée au préalable
kaggle datasets download \
  -d maqdadali/mental-health-risk-prediction-dataset-students \
  -p data/raw --unzip
```

Vérifier le nom des fichiers obtenus avant d'écrire les chemins dans les scripts ; il n'est pas présumé ici. Ne pas versionner `data/raw/` ni les fichiers renfermant des données sensibles sans autorisation explicite.

## 5. Arborescence du pipeline et de la démo

```text
soutnance_1/
├── compose.yaml                      # Services et volumes locaux
├── Dockerfile                        # Image commune à Airflow et Streamlit
├── requirements-ml.txt               # Dépendances communes du modèle
├── data/raw/student-mental.csv        # Source montée en lecture seule
├── notebooks/                        # Exploration et expérimentation
├── dags/student_risk_pipeline.py      # Planification et dépendances des tâches
├── src/soutnance_1/
│   ├── config.py                     # Chemins, cible et paramètres communs
│   ├── preparation.py
│   ├── pretraitement.py
│   ├── entrainement.py
│   └── evaluation.py
└── app/
    ├── requirements.txt
    └── streamlit_app.py              # Résultats et formulaire de simulation
```

Le pipeline Docker publie ses propres artefacts dans les volumes déclarés par `compose.yaml`. Les notebooks utilisent les dossiers `artifacts/` et `models/`.

## 6. Protocole scientifique

### Étape 1 — Audit et EDA

- Charger les données et contrôler dimensions, schéma, types, valeurs nulles, doublons et valeurs aberrantes.
- Produire distributions et statistiques descriptives, profil de la cible, analyses univariées et bivariées, corrélations adaptées aux types de variables.
- Examiner les associations avec la cible sans les interpréter comme des relations causales.
- Décider des colonnes exclues avec justification (identifiants, fuite de cible, variables inadaptées).
- Enregistrer une synthèse EDA avec les hypothèses, limites et références des figures.

### Étape 2 — Séparation et prétraitement

1. Fixer la graine aléatoire (`random_state=42`, sauf justification différente).
2. Définir `X` et `y` puis effectuer le découpage **80 % entraînement / 20 % test**. Stratifier si les classes et leurs effectifs le permettent ; sinon documenter l'alternative.
3. **Geler le test :** ne pas le consulter pour sélectionner les variables, calculer des statistiques d'imputation, calibrer un seuil ou choisir des hyperparamètres.
4. Sur les plis d'entraînement uniquement, apprendre l'imputation, l'encodage et la mise à l'échelle dans un `ColumnTransformer` / `Pipeline`.
5. Traiter les valeurs aberrantes seulement après analyse métier ; préférer, au besoin, un traitement robuste entraîné sur le train plutôt que supprimer automatiquement les observations extrêmes.
6. Mesurer le déséquilibre : commencer par `class_weight` / un modèle de référence avant le rééchantillonnage. Si utilisé, placer le suréchantillonnage **dans** `imblearn.pipeline.Pipeline` et **à l'intérieur des plis de CV** (jamais avant le split ni sur le test). Pour les catégories non numériques, vérifier l'adéquation de `SMOTENC` ou d'une autre méthode plutôt qu'appliquer aveuglément SMOTE aux colonnes one-hot.
7. Tester l'absence de fuite de données, la stabilité du schéma et la gestion des catégories inconnues.

### Étape 3 — Modèles de référence et comparaison

- Baseline `DummyClassifier` pour établir un plancher de performance.
- Trois algorithmes comparables : **régression logistique**, **Random Forest**, **HistGradientBoosting** ou **Decision Tree** selon le schéma constaté. Choisir au minimum trois modèles de classification distincts.
- Utiliser les mêmes partitions / plis et une validation croisée stratifiée si applicable.
- Comparer F1 macro, rappel, précision, accuracy et matrices de confusion ; préciser la classe positive si le problème est binaire.
- Analyser le compromis entre erreurs, interprétabilité, latence et simplicité, sans surinterpréter un seul score.

### Étape 4 — Optimisation et évaluation finale

- Optimiser 1 à 2 modèles prometteurs sur **l'entraînement uniquement**, avec `RandomizedSearchCV` ou `GridSearchCV`.
- Choisir le candidat via la validation croisée et documenter l'espace de recherche et les meilleures valeurs.
- **Évaluer une seule fois le modèle retenu sur le test indépendant**, puis analyser les confusions et erreurs. Les analyses supplémentaires de test doivent être descriptives, sans nouvelle sélection de modèle sur ce test.
- Réaliser une vérification de robustesse (par exemple répétition de CV / stabilité par segment si effectifs suffisants) et signaler les limites.
- Sauvegarder le pipeline **complet** (transformations + estimateur) avec joblib et les métadonnées : version des dépendances, noms et ordre des colonnes, classes, graine, paramètres, métriques.

### Étape 5 — Airflow et Streamlit

Le DAG `student_risk_pipeline` enchaîne quatre tâches.

```text
prepare → preprocess → train → evaluate
```

- `prepare` retire `Student_ID` et effectue une séparation stratifiée 80/20 avec la graine 22.
- `preprocess` prépare le `ColumnTransformer`. Chaque pli apprend ses transformations pendant l’entraînement.
- `train` optimise la régression logistique, KNN et Random Forest sur cinq plis par F1 macro, puis sauvegarde le meilleur pipeline dans `best_model.joblib`.
- `evaluate` évalue ce pipeline sur les 20 % réservés au test et sauvegarde `classification_report.txt` et `confusion_matrix.png`.

#### 1. Démarrer les services sous Windows

Ouvrir Docker Desktop en mode conteneurs Linux. Depuis PowerShell, à la racine du dépôt, vérifier que `docker version` affiche une section `Server` et que `data/raw/student-mental.csv` est présent.

```powershell
docker compose config --quiet
docker compose up --build -d
docker compose ps -a
```

Le service `airflow` lance `airflow standalone` avec SQLite. Le volume `airflow-home` conserve le nouvel historique Airflow. Streamlit réutilise la même image et lit le modèle depuis le volume `pipeline-results`.

#### 2. Lancer et suivre un run Airflow

Ouvrir Airflow et se connecter avec `admin`. Airflow génère le mot de passe à la première initialisation et le conserve dans `airflow-home`. [Airflow](http://localhost:8081)

```powershell
docker compose exec airflow cat /opt/airflow/simple_auth_manager_passwords.json.generated
```

Dans **Dags**, ouvrir `student_risk_pipeline` et cliquer sur **Trigger**. L’optimisation des modèles peut prendre plusieurs minutes. Le DAG est actif à sa création et s’exécute tous les jours, sans rattrapage. Après modification du code, relancer `docker compose up --build -d`.

Dans **Grid** ou **Graph**, sélectionner le run, cliquer sur une tâche et ouvrir **Logs**. Après avoir corrigé la cause d’un échec, utiliser **Clear task instance** sur la tâche et les tâches en aval à rejouer.

```powershell
docker compose exec airflow airflow dags list-import-errors
docker compose exec airflow airflow dags list-runs student_risk_pipeline
docker compose logs --tail 100 airflow
```

#### 3. Consulter les performances et simuler une prédiction

Après le succès du DAG, ouvrir Streamlit, renseigner un profil fictif et cliquer sur **Predict**. Le tableau de comparaison, le rapport et la matrice de confusion se trouvent dans `/home/airflow/student-artifacts` dans le conteneur `airflow`. [Streamlit](http://localhost:8501)

Pour arrêter les services en conservant les résultats et l’historique Airflow :

```powershell
docker compose down
```

#### Démo Streamlit sans Docker

Si le moteur Docker est indisponible, exécuter les étapes directement avec Python depuis la racine du dépôt. `runtime/` reçoit les artefacts locaux et l’environnement de démo.

```powershell
uv venv runtime/venv
uv pip install --python runtime/venv/Scripts/python.exe -r requirements-ml.txt -r app/requirements.txt
$env:PYTHONPATH = 'src'
$env:OPENBLAS_NUM_THREADS = '1'
$env:OMP_NUM_THREADS = '1'
$env:STUDENT_OUTPUT_DIR = 'runtime/artifacts'
$env:STUDENT_MODEL_PATH = 'runtime/artifacts/best_model.joblib'
foreach ($stage in 'preparation', 'pretraitement', 'entrainement', 'evaluation') {
    & runtime/venv/Scripts/python.exe -m "soutnance_1.$stage"
    if ($LASTEXITCODE -ne 0) { throw "Échec de l’étape $stage" }
}
& runtime/venv/Scripts/python.exe -m streamlit run app/streamlit_app.py --server.address 127.0.0.1 --server.port 8502
```

Ouvrir [la démo locale](http://localhost:8502). La planification et le suivi des tâches passent par les services Airflow.

## 7. Organisation du travail — 5 jours

| Jour | Date | Priorité | Résultat vérifiable en fin de journée |
|---|---|---|---|
| **J1** | Lun. 09/11 | Cadrage, chargement, audit et première EDA | Données inspectées, cible confirmée, limites recensées |
| **J2** | Mar. 10/11 | EDA approfondie, split et pipeline anti-fuite | Figures EDA + pipeline de préparation testé |
| **J3** | Mer. 11/11 | ≥3 modèles, CV, métriques et comparaison | Tableau comparatif reproductible |
| **J4** | Jeu. 12/11 | Tuning, test final, sérialisation, DAG initial | Pipeline retenu + score du test + DAG structurel |
| **J5** | Ven. 13/11 | Airflow, Streamlit, QA, documentation, répétition | Application et pipeline démontrables, dépôt prêt |
| **Livraison** | Sam. 14/11, **23 h 59** | Date limite / marge de sécurité | Version finale déposée avant la limite |
| **Jury** | Lun. 16/11 | Soutenance de 45 min | Démo et réponses techniques préparées |

La répartition détaillée, les Epics, tâches, estimations, dates et critères d'acceptation figurent dans **`jira_student_mental_health_plan.csv`**. Les cinq journées de production sont du 9 au 13 ; le 14 est traité comme une **date limite**, non comme un sixième jour de travail promis.

## 8. Définition de « terminé » (DoD)

Une tâche n'est considérée terminée que si :

- le code et les artefacts attendus sont enregistrés aux chemins convenus ;
- les décisions techniques sont justifiées dans une note, le README ou un notebook ;
- l'exécution réussit avec une graine / configuration explicite ;
- l'absence de fuite est vérifiée là où nécessaire ;
- les erreurs sont traitées, les métriques et graphiques sont lisibles ;
- les résultats ne suggèrent aucune validité clinique et ne divulguent aucune donnée individuelle.

### Risques à surveiller

| Risque | Mesure de maîtrise |
|---|---|
| Cible artificielle / mal documentée | Exposer sa construction inconnue et limiter les conclusions au dataset |
| Fuite de cible, prétraitement avant split | Audit des colonnes + pipeline appris dans la CV |
| Classes déséquilibrées | Macro-F1, rappel par classe, poids ou sampling dans les plis |
| Très bons scores trompeurs | Baseline, comparaison CV/test, recherche de variables proxy, test indépendant |
| Installation Airflow trop coûteuse | Environnement séparé WSL2 / Docker, DAG minimal vérifiable en premier |
| Surcharge de J5 | Avoir un modèle sérialisé et des métriques stables dès J4 |
| Données sensibles ou confusion clinique | Utilisation de données anonymes/synthétiques dans la démonstration, avertissements explicites |

## 9. Préparation du jury (16 novembre 2026)

| Segment | Durée | Démonstration prévue |
|---|---:|---|
| Présentation | 10 min | Problématique, source, EDA, aperçu Streamlit |
| Analyse du code | 10 min | Pipeline sans fuite, modèle, tests et architecture du DAG |
| Mise en situation | 20 min | Formulaire, cas invalides, lancement DAG, graphiques et limites |
| Code review / questions | 5 min | Reproductibilité, SMOTE, métriques, choix du modèle, limites éthiques |

**Checklist de livraison :** README à jour ; scripts/notebooks reproductibles ; données exclues du dépôt si requis ; pipeline joblib ; métriques figées ; figures ; DAG ; application Streamlit ; tests ; captures de démonstration ; limitations et présentation.

## 10. Références

- [Dataset — Kaggle](https://www.kaggle.com/datasets/maqdadali/mental-health-risk-prediction-dataset-students)
- [scikit-learn — Pipelines et bonnes pratiques](https://scikit-learn.org/stable/modules/compose.html)
- [scikit-learn — Cross-validation](https://scikit-learn.org/stable/modules/cross_validation.html)
- [imbalanced-learn — Common pitfalls](https://imbalanced-learn.org/stable/common_pitfalls.html)
- [Apache Airflow — Quick Start](https://airflow.apache.org/docs/apache-airflow/stable/start.html)
- [Streamlit — Documentation](https://docs.streamlit.io/)
- [Atlassian — Import CSV](https://support.atlassian.com/jira-cloud-administration/docs/import-data-from-a-csv-file/)
