# **testMansa**



Pour info, le notebook a été rédigé dans google colab (Python 3.7.12). Et l'api en local (Python 3.8).
L'une ou l'autre version devrait suffire pour le notebook et l'api. Le notebook peut nécessiter l'installation 
de ipykernel (pip install ipykernel) si celui-ci est lancé à partir de VSCode par exemple. 

Le parti pris ici est d'estimer des intervalles de prédiction pour la somme des dépenses des 30 jours suivants. 
La démarche est inspirée du papier suivant : http://proceedings.mlr.press/v80/pearce18a/pearce18a.pdf
En bref, le modèle est un réseau de neurones, relativement simple, bien qu'il contienne un mécanisme d'attention (self-attention) afin de prendre en compte les dépendances entre les données d'entrée. 

Son output est donc composé de deux valeurs, une borne supérience et une borne inférieure matérialisant l'intervalle de prédiction (et non de confiance) du modèle. Le papier propose une explication de la différence en partie 2.1 si toutefois c'est nécessaire. L'api retourne en sortie ces deux valeurs, ainsi que leur moyenne pour obtenir un "prediction amount". 
Ce type de méthode est je pense adaptée à ce genre de données à composantes stochastiques. Les dépenses sont composées d'éléments réguliers (factures, loyers, etc), mais aussi d'éléments aléatoires.


Le notebook contient l'ensemble de la démarche (mansa_inference.ipynb), de la préparation des données à l'entraînement du modèle. Les choix et étapes y sont commentées, ainsi que de potentielles autres pistes. 

**Spoiler**

- Aucune recherche d'hyper-paramètre a été effectuée à l'aide de librairie externe (Optuna par exemple, où des algorithmes inspirée par des Bandits - HyperBand, peuvent être adaptés aux réseaux de neuronnes, étant donné le contrôle qu'ils donnent en termes de "budget", par construction, et par opposition à des méthodes de recherche aléatoires par exemple). Cependant quelques essais ont été faits "à la main" pour l'occasion.

- Le choix a été fait de ne pas ajouter de composante auto-régressive dans le modèle (type LSTM, GRU, etc). Leur optimisation est plus lente que d'autres modèles, et peuvent présenter des limites selon la taille des séries temporelles (limites dans l'apprentissage des dépendances de long terme). Il peut être intéressant de les rajouter à des fins d'amélioration cependant, en prenant en compte leurs limites.

- D'autres méthodes axées sur les arbres de régression (XgBoost, LightGBM, ...) peuvent être des concurrents sérieux, à condition de produire des features de qualités et structurées (à l'aide de librairies comme TsFresh par exemple, et/ou des features issues du métier). Le parti pris ici a été d'assumer la nature non structurée des données. 

- Des visualisations supplémentaires pourraient être faites, notamment sur les paramètre du mécanisme d'attention. A des fins de contrôle de la convergence et de l'apprentissage du modèle, mais aussi à des fins informatives. 

- Des benchmarks n'ont pas été faits, le focus a été mis sur le modèle, mais cette étape est nécessaire pour prendre la pleine mesure des performances. Je pense à de simples benchmark comme des moyennes par exemple.

- Le modèle utilise la syntax de Keras (sous TF 2.x), par simplicité. Tensorboard peut être un outil utile à l'évaluation et le monitoring de l'entraînement du modèle. Cette étape a été jugée prématurés ici. 




