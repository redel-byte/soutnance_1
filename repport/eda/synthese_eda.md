# Synthèse EDA

Le fichier contient 2000 lignes et 17 colonnes. L’audit compte
394 lignes avec des valeurs manquantes, 0 doublons exacts et
0 valeurs invalides. La règle de l’IQR signale 61 valeurs atypiques.
Dans 37 lignes, le temps sur les réseaux sociaux dépasse le temps d’écran total.
La période de mesure reste à vérifier.

## Classes de risque

- Low · 660 lignes (33.0 %)
- Medium · 660 lignes (33.0 %)
- High · 680 lignes (34.0 %)

Le sommeil médian est de 7.6 h pour Low et de 6.1 h pour High.
Cette association décrit le fichier. Les effectifs par genre sont Female (971), Male (932), Non-binary (97).
La licence, le mode de collecte, la représentativité et la méthode d’étiquetage restent à confirmer.
L’analyse est descriptive et l’usage est pédagogique.

`Student_ID` est exclu des variables explicatives et `Mental_Health_Risk` est la cible.
La date de collecte et la construction des étiquettes restent à vérifier pour évaluer le risque de fuite.

## Figures

- [Valeurs manquantes](01_valeurs_manquantes.png)
- [Classes de risque](02_cible.png)
- [Distributions numériques](03_distributions_numeriques.png)
- [Corrélations de Spearman](04_correlations_spearman.png)
- [Variables numériques par classe](05_numeriques_par_risque.png)
- [Variables catégorielles par classe](06_categorielles_par_risque.png)
