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




Partie 3 - Ajout les poids et les dernières couches

Nous pouvons utiliser le fichier Python pour entraîner le réseau neuronal et obtenir les poids et les bias du modèle. 

Nous avons obtenu des résultats suffisament satisfaisants pour récuperer les parametres:
<img width="530" alt="image" src="https://github.com/gaspardCh/TP_Hardware/assets/118471792/b538b7be-196d-4831-979c-fd8d43b845ab">

On a obtenu un grand modèle:

<img width="424" alt="image" src="https://github.com/gaspardCh/TP_Hardware/assets/118471792/469c8d00-8b43-4306-8552-cef8b09e147e">


Explications du nombre de paramètres :

conv2d: ((width of kernel)x(length of kernel)+ 1 (for the bias))x(depth of Kernel)=(5x5+1)x6=156 parameters

conv2d_1: ((width of kernel)x(length of kernel)x(depth of initial Kernel)+ 1 (for the bias))x(depth of Kernel)=(5x5x6+1)x16=2416 parameters

Average_pooling and flatten layers- we do not learn parameters

Dense layers: (input size + 1(bias))x (output size)

dense: (400+1)x120= 48120
dense_1: (120+1)x84= 10164
dense_2: (84+1)x10= 850

On a donc 61706 trainable parameters (on n'as pas fait une batch normalization et tous les parameters sont appris).

On a recuperé tous les poids dans le fichier FashionMNIST_weights.h5.








