# TP_Hardware

Objectif: L'utilisation de GPUs pour des modèles profonds a beaucoup des advantages, notament la vitesse d'excecution à cause de la parallélisation des calculs. L'objectif de ce TP est à faire une implémentation du model LeNet5 avec cuda pour l'accélérer.



<img width="637" alt="image" src="https://github.com/gaspardCh/TP_Hardware/assets/118471792/d6ec0f52-16a8-4fb7-b94a-f3f2c3d84015">


On peut voir que le LeNet5 a beaucoup des couches qui sont parallélisable- les convolutions et les subsamplings par example. Notre objectif va être à créer des algorithmes capables de faire ces opérations de manière parallèle.


Partie 1 - Prise en main de Cuda : Multiplication de matrices

La multiplication et l'addition des matrices sont 2 opérations pour lequel chaque terme de l'output peut être calculé indépendamment des autres termes de sortie. Cuda peut nous aider à faire ces calculs en parallèle. Mais s'agit-il de la meilleure stratégie, quelle que soit la taille de l'entrée ?

En théorie, la complexité est :

-pour l'addition: 

 pour le CPU: en O(n*p) (On a n*p termes qui sont calculés consécutivement l'un après l'autre)
 
 pour le GPU: O(1) (tous les termes sont calculés en même temps)

-pour la multiplication:

 pour le CPU: en O(n³) (Nous avons n*n termes et pour calculer chaque terme, nous devons effectuer n opérations)
 
 pour le GPU: en O(n) (Nous calculons tous les termes en même temps, mais pour calculer chaque terme, nous devons effectuer n opérations)
 

on mesure le temps d'exécution des différentes opération pour n = p = 100: 
pour l'addition, environ 64µs pour le CPU et 146µs pour le GPU
pour la multiplication, environ 3.94 ms pour le CPU et 0.11ms pour le GPU

Conclusions :
Le temps nécessaire au transfert d'informations entre le CPU et le GPU n'est pas toujours négligeable au détriment d'autres commandes.
Il n'y a d'intérêt à utiliser le GPU que si la partie parallélisée permet de gagner plus de temps que le transfert d'informations. Dans notre cas, cela est envisageable avec des matrices d'entrée de grande taille.  


Partie 2 - Premières couches du réseau de neurone LeNet-5 : Convolution 2D et subsampling

La convolution et le sous-échantillonnage peuvent être facilement parallélisés grâce au GPU.













