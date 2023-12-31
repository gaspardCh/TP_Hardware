# TP_Hardware

Objectif: L'utilisation de GPUs pour des modèles profonds a beaucoup des advantages, notament la vitesse d'excecution à cause de la parallélisation des calculs. L'objectif de ce TP est à faire une implémentation du model LeNet5 avec cuda pour l'accélérer.



<img width="637" alt="image" src="https://github.com/gaspardCh/TP_Hardware/assets/118471792/d6ec0f52-16a8-4fb7-b94a-f3f2c3d84015">


On peut voir que le LeNet5 a beaucoup des couches qui sont paralysable- les convolutions et les subsamplings par example. Notre objectif va être à créer des algorithmes capables de faire ces opérations de manière parallèle.


Partie 1 - Prise en main de Cuda : Multiplication de matrices

La multiplication et l'addition des matrices sont 2 opérations pour lequel chaque terme de l'output peut être calculé indépendamment des autres termes de sortie. Cuda peut nous aider à faire ces calculs en parallèle. Mais s'agit-il de la meilleure stratégie, quelle que soit la taille de l'entrée ?

En théorie, la complexité est :
-pour l'addition:
 pour le CPU: en O(n*p) (On a n*p termes qui sont calculés consécutivement l'un après l'autre)
 pour le GPU: O(1) 


pour la multiplication, en O(n³) pour le CPU et en O(n) pour le GPU

on mesure le temps d'exécution des différentes opération pour n = p = 100: 
pour l'addition, environ 64µs pour le CPU et 146µs pour le GPU
pour la multiplication, environ 3.94 ms pour le CPU et 0.11ms pour le GPU
