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
- **Variable cible :** **à confirmer dans le fichier téléchargé** ; ne pas supposer son nom, son nombre de classes ni la manière dont ses étiquettes ont été produites.
- **Licence, provenance, consentement et représentativité :** vérifier les informations réellement fournies par le créateur du dataset et documenter les inconnues.
- **Confidentialité :** éviter d'inclure des identifiants personnels, observations sensibles brutes ou copies publiques des données dans le dépôt Git.

### Points de vigilance à vérifier à J1

1. Taille du jeu de données, noms et types de colonnes, unités et cardinalités.
2. Définition de la cible : binaire ou multiclasse, répartition des classes, éventuelle génération artificielle / heuristique des étiquettes.
3. Valeurs manquantes, doublons et valeurs impossibles ; distinguer anomalies réelles et valeurs rares mais plausibles.
4. Colonnes identifiantes et **fuites de cible** (variables dérivées de la cible ou disponibles seulement après l'événement prédit).
5. Risques de biais, échantillonnage non représentatif et absence éventuelle de validation clinique.

> Aucune liste de variables, aucun score et aucune interprétation du dataset ne sont annoncés avant inspection du fichier source.

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
| Langage / environnement | Python 3.11 ou 3.12, `uv` ou `venv`, Git |
| Manipulation | NumPy, pandas |
| Exploration / graphiques | Matplotlib, Seaborn, éventuellement Plotly |
| Modèles / pipeline | scikit-learn, imbalanced-learn si pertinent |
| Optimisation / suivi | GridSearchCV ou RandomizedSearchCV, JSON/CSV de métriques |
| Sérialisation | joblib (chargement depuis une source de confiance uniquement) |
| Orchestration | Apache Airflow sous Linux, WSL2 ou Docker |
| Interface | Streamlit |
| Tests | pytest et tests de fumée / reproductibilité |

### Installation de départ (environnement Data Science)

```bash
# Depuis la racine du dépôt — Python 3.11 conseillé
uv venv --python 3.11

# Sous Linux / WSL
source .venv/bin/activate

# Sous Windows PowerShell, à la place :
# .venv\Scripts\Activate.ps1

uv pip install pandas numpy matplotlib seaborn plotly \
  scikit-learn imbalanced-learn streamlit joblib pytest kaggle
```

**Note :** ce sont les dépendances proposées, pas un environnement déjà figé. Avant livraison, produire un `requirements.txt` / `pyproject.toml` et verrouiller les versions réellement utilisées. Pour Airflow, utiliser un **environnement distinct** ou Docker afin d'éviter les conflits de dépendances, et suivre les contraintes officielles correspondant à la version de Python : [documentation Airflow](https://airflow.apache.org/docs/apache-airflow/stable/start.html). Le prototype n'exige pas de GPU.

### Récupérer le dataset

```bash
# Authentification Kaggle configurée au préalable
kaggle datasets download \
  -d maqdadali/mental-health-risk-prediction-dataset-students \
  -p data/raw --unzip
```

Vérifier le nom des fichiers obtenus avant d'écrire les chemins dans les scripts ; il n'est pas présumé ici. Ne pas versionner `data/raw/` ni les fichiers renfermant des données sensibles sans autorisation explicite.

## 5. Arborescence cible (à créer progressivement)

```text
student-mental-health-risk/
├── README.md
├── pyproject.toml                 # Ou requirements.txt
├── .gitignore
├── configs/
│   └── config.yaml               # Cible, seed, chemins, paramètres
├── data/
│   ├── raw/                      # Entrées : non versionnées
│   ├── interim/
│   └── processed/
├── notebooks/
│   ├── 01_data_audit_eda.ipynb
│   ├── 02_preprocessing.ipynb
│   └── 03_modeling_evaluation.ipynb
├── src/
│   └── student_risk/
│       ├── data.py
│       ├── validation.py
│       ├── preprocessing.py
│       ├── training.py
│       ├── evaluation.py
│       └── inference.py
├── dags/
│   └── student_risk_pipeline.py
├── app/
│   └── streamlit_app.py
├── models/                       # Pipeline joblib local, version contrôlée
├── reports/
│   ├── figures/
│   ├── metrics/
│   └── model_card.md
└── tests/
    ├── test_preprocessing.py
    ├── test_training.py
    └── test_inference.py
```

Il s'agit d'une **architecture proposée**, pas d'une déclaration que tous ces fichiers existent déjà. Centraliser les paramètres et la logique réutilisable dans `src/` ; utiliser les notebooks pour l'exploration et la narration, non comme unique source du pipeline Airflow.

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

**DAG cible :**

```text
validate_source
       |
prepare_split
       |
train_and_cross_validate
       |
select_and_package
       |
publish_validation_metrics
```

- Tâches idempotentes, chemins et configurations explicites, journalisation, gestion des exceptions et fichiers produits vérifiés.
- Prévoir une exécution planifiée et une démonstration de reprise / visualisation des statuts. La **mesure du test final est un contrôle ponctuel**, pas une nouvelle étape de sélection exécutée après chaque relance du DAG.
- La partie Streamlit présente le résumé EDA, le comparatif, les métriques de test figées, la matrice de confusion, les limites et un formulaire de simulation utilisant **des entrées anonymes ou synthétiques**.
- Afficher sur tous les écrans de prédiction : *« Démonstration pédagogique — résultat non clinique, non diagnostique et non destiné à une décision individuelle. »*
- Contrôler les types, bornes et catégories à l'entrée ; ne pas collecter ni conserver de données personnelles dans l'interface de démonstration.

### Exécution de l'application (après implémentation)

```bash
streamlit run app/streamlit_app.py
```

### Tests (après implémentation)

```bash
pytest -q
```

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

## 10. Import Jira du CSV fourni

Le fichier `jira_student_mental_health_plan.csv` contient les colonnes : `Work item ID`, `Work Type`, `Summary`, `Description`, `Parent`, `Priority`, `Status`, `Labels`, `Start date`, `Due Date`, `Original Estimate`.

1. Dans Jira Cloud, ouvrir l'import CSV (chemin exact selon droits et interface Jira) et choisir **le projet de destination**.
2. Déclarer l'encodage **UTF-8**, séparateur **virgule**, et le format de date **`yyyy-MM-dd`** (Java SimpleDateFormat) correspondant au fichier.
3. Mapper `Work Type` → **Work Type / Issue Type**, `Summary` → **Summary**, `Parent` → **Parent**, `Work item ID` → **Work item ID**, `Description` → **Description**, `Priority` → **Priority**, `Status` → **Status**, `Labels` → **Labels**, `Due Date` → **Due Date**.
4. Mapper `Start date` → **Start date** uniquement si ce champ est disponible dans le projet, et `Original Estimate` → **Original Estimate** uniquement si le suivi du temps est activé (nombre **de secondes**). Sinon, **décocher** ces colonnes lors du mapping.
5. Vérifier l'existence des types **Epic** et **Task**, du statut **To Do** et des priorités **High** / **Medium** dans la configuration du projet ; mapper les valeurs si les noms locaux diffèrent.
6. Importer et vérifier 2 Epics, quelques tâches et leurs relations `Parent`, les dates et les estimations avant de considérer l'import réussi. Les IDs numériques du CSV sont des références **internes d'import**, pas des clés Jira définitives.

Atlassian documente les champs `Summary`, `Work Type`, `Parent`, `Work item ID`, `Due Date`, `Labels`, `Priority` et l'estimation initiale en **secondes** : [guide officiel CSV Jira Cloud](https://support.atlassian.com/jira-cloud-administration/docs/import-data-from-a-csv-file/), [préparation et hiérarchie](https://support.atlassian.com/jira-software-cloud/docs/prepare-a-csv-file-for-import/). L'import hiérarchique dépend de la configuration et des autorisations du projet Jira.

## 11. Références

- [Dataset — Kaggle](https://www.kaggle.com/datasets/maqdadali/mental-health-risk-prediction-dataset-students)
- [scikit-learn — Pipelines et bonnes pratiques](https://scikit-learn.org/stable/modules/compose.html)
- [scikit-learn — Cross-validation](https://scikit-learn.org/stable/modules/cross_validation.html)
- [imbalanced-learn — Common pitfalls](https://imbalanced-learn.org/stable/common_pitfalls.html)
- [Apache Airflow — Quick Start](https://airflow.apache.org/docs/apache-airflow/stable/start.html)
- [Streamlit — Documentation](https://docs.streamlit.io/)
- [Atlassian — Import CSV](https://support.atlassian.com/jira-cloud-administration/docs/import-data-from-a-csv-file/)

---

**État initial :** planification et documentation de démarrage. Résultats expérimentaux, variables et métriques réelles à compléter après l'analyse du dataset.
