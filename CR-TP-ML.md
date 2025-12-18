#######################################################################

# Binome : BEZMATERNYKH Igor / AUDIC Clément

#

## Jeu de données : pré-traitement

Donnez la liste des features et ce qu'elles représentent (préciser les éventuels changements effectués en pré-traitement ou si pas de changement)

- Liste des features :

* AGEP : L'age de la personne
* COW : Classe d'emploi (Civilian employed, Unemployed, Not in labor force, Armed forces)
* SCHL : Niveau d'éducation
* MAR : Statut marital
* OCCP : Emploi occupé en code que nous avons regrouppé en catégories
* POBP : Place of birth que nous avons regrouppé par régions géographiques (Afrique du Nord, Europe de l'Est)
* RELP : Relation par rapport à la personne possèdant le logement
* WKHP : Nombre d'heures travaillées par semaine
* SEX : Sexe
* RAC1P : Ethnicité

## Expérimentation 1 : Comparaison de modèles par défaut

- Jeux de données utilisé :

  - Taille ensemble d'entrainement (nb lignes et nb colonnes) : 124736 lignes et 10 colonnes
  - Taille ensemble de test (nb lignes et nb colonnes) : 41579 lignes et 10 colonnes

- Résultats (hyper-paramètres par défaut)

| Evaluation en train                 | Random Forest                                                   | Adaboost                                                    | XGBoost                                                |
| ----------------------------------- | --------------------------------------------------------------- | ----------------------------------------------------------- | ------------------------------------------------------ |
| accuracy                            | 0.9876699589533093                                              | 0.7991918932786044                                          | 0.8361018471010775                                     |
| Temps calcul (temps d'entrainement) | 23.58                                                           | 6.59                                                        | 1.77                                                   |
| Matrice confusion                   | ![cm](images/RandomForestClassifier/confusion_matrix_train.png) | ![cm](images/AdaBoostClassifier/confusion_matrix_train.png) | ![cm](images/XGBClassifier/confusion_matrix_train.png) |

</br>

| Evaluation en test               | Random Forest                                             | Adaboost                                              | XGBoost                                          |
| -------------------------------- | --------------------------------------------------------- | ----------------------------------------------------- | ------------------------------------------------ |
| accuracy                         | 0.7996584814449602                                        | 0.7965800043291085                                    | 0.8172394718487698                               |
| Temps calcul (Temps d'inférence) | 1.27                                                      | 0.54                                                  | 0.05                                             |
| Matrice confusion                | ![cm](images/RandomForestClassifier/confusion_matrix.png) | ![cm](images/AdaBoostClassifier/confusion_matrix.png) | ![cm](images/XGBClassifier/confusion_matrix.png) |

- Commentaires et Analyse :
  On remarque que le RandomForest a tendence à énormément overfit avec un score d'entrainement proche de 1 mais un score de test bien plus faible. L'Adaboost et le XGBoost ont des scores d'entrainement et de test plus proches, mais le XGBoost a de meilleurs performances globales. On remarque aussi que XGBoost est plusieurs fois plus rapide que les autres. Les temps d'inférence sont pluls proches mais on remarque que le XGBoost est plus optimisé puisqu'il est 10 fois plus rapide que les autres.

## Expérimentation 2 : Comparaison Modèles ML par défaut

- Jeux de données utilisé :
  - Taille ensemble d'entrainement : 124736 lignes et 10 colonnes
  - Taille ensemble de test : 41579 lignes et 10 colonnes

### Random Forest (RF)

- Processus d'entrainement :
  - Recherche des hyperparamètres : Utilisation d'Optuna, une bibliothéque faisant de la recherche de paramètres par optimisation bayésienne plutôt que par recherche exhaustive.
  - Listes des hyperparamètres testés et valeurs :
    > 'learning_rate': ('choice', [0.01, 0.05, 0.1]),
    > 'max_depth': ('choice', [1, 3, 6, 10, 15, 20]),
    > 'n_estimators': ("choice", [50, 100, 200, 300, 400]),
  - Nombre de plis pour la validation croisée : 3
  - Nombre total d'entrainement : $20 \times 3 = 60$
- Résultats :
  - Meilleurs hyperparamètres :
    > {'max_depth': 15, 'n_estimators': 300}
  - Performances en entraintement :
    - Accuracy : 0.8409440738840431
    - Temps de calcul : 36.44 seconds
    - Matrice de Confusion : ![cm](images/RandomForestClassifier/optimized_confusion_matrix_train.png)
  - Performance en test :
    - Accuracy : 0.8119964405108347
    - Temps de calcul : 1.62 seconds
    - Matrice de Confusion : ![cm](images/RandomForestClassifier/optimized_confusion_matrix.png)
  - Commentaires / analyses (par rapport résultat expe 1)
    L'optimisation des hyperparamètres a permis d'améliorer la performance de RandomForest sur le dataset de test de 1.2% en réduisant l'overfitting. La différence entre les scores d'entrainement et de test est en effet nettement réduite, le score d'entrainement passant à 84% au lieu de 99%. Le temps d'entrainement s'est en revanche aggrandi à cause d'un plus grand nombre d'estimateurs.

### ADABOOST

AdaBoostClassifier || Test score: 0.7987686091536593 | Train score: 0.8036733581323756 | Training time: 60.93 seconds || Test time: 4.14 seconds

- Processus d'entrainement :
  - Recherche des hyperparamètres
  - Listes des hyperparamètres testés et valeurs :
    > 'learning_rate': {'type':'float', 'low':0.01, 'high':0.3},
    > 'n_estimators': {'type':'int', 'low':50, 'high':450, 'step':10},
  - Nombre de plis pour la validation croisée : 3
  - Nombre total d'entrainement : 60
- Résultats :
  - Meilleurs hyperparamètres :
  - Performances en entraintement :
    - Accuracy : 0.8036733581323756
    - Temps de calcul : 60.93 seconds
    - Matrice de Confusion : ![cm](images/AdaBoostClassifier/optimized_confusion_matrix_train.png)
  - Performance en test :
    - Accuracy : 0.7987686091536593
    - Temps de calcul : 4.14 seconds
    - Matrice de Confusion : ![cm](images/AdaBoostClassifier/optimized_confusion_matrix.png)
  - Commentaires / analyses (par rapport résultat expe 1)
    L'optimisation des hyperparamètres n'a pas été très efficace. Le score de test n'a augmenté que de 0.2%, avec un score d'entrainement qui a augmenté de 0.4%, ce qui veut dire qu'il ya peut être eu un léger overfitting. Le temps d'entrainement a en revanche beaucoup augmenté, passant de 6.59 secondes à presque 1 minute, ce qui est surement dû à l'augmentation du nombre d'estimateurs.

### XGBOOST

- Processus d'entrainement :
  - Recherche des hyperparamètres
  - Listes des hyperparamètres testés et valeurs :
    > 'learning_rate': ('choice', [0.05, 0.1, 0.3]),
    > 'max_depth': ('choice', [3, 6, 10, 15, 20]),
    > 'n_estimators': ("choice", [50, 100, 200, 400])
  - Nombre de plis pour la validation croisée :
  - Nombre total d'entrainement :
- Résultats :
  - Meilleurs hyperparamètres :
    > {'learning_rate': 0.05, 'max_depth': 6, 'n_estimators': 400}
  - Performances en entraintement :
    - Accuracy : 0.8320933812211391
    - Temps de calcul (temps d'entrainement) : 2.71 seconds
    - Matrice de Confusion : ![cm](images/XGBClassifier/optimized_confusion_matrix_train.png)
  - Performance en test :
    - Accuracy : 0.8193318742634503
    - Temps de calcul : 0.08 seconds
    - Matrice de Confusion : ![cm](images/XGBClassifier/optimized_confusion_matrix.png)
  - Commentaires / analyses (par rapport résultat expe 1)
    L'optimisation des hyperparamètres n'a pas été très efficace sur XGBoost, ce qui est normal, car c'est un modèle réputé pour être très performant et ne nécessitant pas beaucoup d'optimisation. On peut quand-même remarquer que l'on a pu éviter un peu d'overfitting car le score d'entrainement a baissé de 3% tandis que le score de test a augmenté de 0.2%. Le temps d'entrainement a par contre augmenté de manière peu significative.

## Expérimentation 3 : Comparaison des "meilleurs modèles

- Jeux de données utilisé :

  - Taille ensemble d'entrainement : 124736 lignes et 10 colonnes
  - Taille ensemble de test : 41579 lignes et 10 colonnes

- Résultats des meilleurs modèles obtenus dans Expe 2

| Evaluation en train | Random Forest                                                             | Adaboost                                                              | XGBoost                                                          |
| ------------------- | ------------------------------------------------------------------------- | --------------------------------------------------------------------- | ---------------------------------------------------------------- |
| accuracy            | 0.8409440738840431                                                        | 0.8036733581323756                                                    | 0.8320933812211391                                               |
| Temps calcul        | 36.44                                                                     | 60.93                                                                 | 2.71                                                             |
| Matrice confusion   | ![cm](images/RandomForestClassifier/optimized_confusion_matrix_train.png) | ![cm](images/AdaBoostClassifier/optimized_confusion_matrix_train.png) | ![cm](images/XGBClassifier/optimized_confusion_matrix_train.png) |

| Evaluation en test | Random Forest                                                       | Adaboost                                                        | XGBoost                                                    |
| ------------------ | ------------------------------------------------------------------- | --------------------------------------------------------------- | ---------------------------------------------------------- |
| accuracy           | 0.8119964405108347                                                  | 0.7987686091536593                                              | 0.8193318742634503                                         |
| Temps calcul       | 1.62                                                                | 4.14                                                            | 0.08                                                       |
| Matrice confusion  | ![cm](images/RandomForestClassifier/optimized_confusion_matrix.png) | ![cm](images/AdaBoostClassifier/optimized_confusion_matrix.png) | ![cm](images/XGBClassifier/optimized_confusion_matrix.png) |

- Commentaires et Analyse :

L'optimisation d'hyperparamètres a permis de réduire la différence de score entre les différents modèles. En fin de compte, XGBoost reste meilleur, mais la performance des deux autres s'en approchent. Générallement les temps de calcul ont augmenté, notamment du à l'augmentation du nombre d'estimateurs.

## Expérimentation 4 : inférence sur un autre jeu de données (optionnel)

Résultats / Commentaires / Analyses :

## Expérimentation 5 : impact de la taille du jeu de données

Résultats / Commentaires / Analyses :

## Modèle choisi pour la suite :

- quel modèle :
- pourquoi ?

## Explicabilité : "permutation feature importance"

- Résultats obtenus :
- Analyses :

## Explicabilité : avec LIME et SHAP

- Méthode LIME
  - Exemple(s) choisi(s)
  - Résultats
  - Commentaires / analyses
- Méthode SHAP
  - Exemple(s) choisi(s)
  - Résultats
  - Commentaires / analyses
- Comparaison LIME et SHAP
- Analyse summary-plot de SHAP

## Explicabilité : contrefactuelle

Résultats / Commentaires / Analyses :
