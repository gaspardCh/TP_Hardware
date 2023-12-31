# TP_Hardware

Objectif: L'utilisation de GPUs pour des modèles profonds a beaucoup des advantages, notament la vitesse d'excecution à cause de la parallélisation des calculs. L'objectif de ce TP est à faire une implémentation du model LeNet5 avec cuda pout accélérer les calculs. 


<img width="637" alt="image" src="https://github.com/gaspardCh/TP_Hardware/assets/118471792/d6ec0f52-16a8-4fb7-b94a-f3f2c3d84015">



En théorie, la complexité est :
pour l'addition, en O(n*p) pour le CPU et O(1) ? pour le GPU 
pour la multiplication, en O(n³) pour le CPU et en O(n) pour le GPU

on mesure le temps d'exécution des différentes opération pour n = p = 100: 
pour l'addition, environ 64µs pour le CPU et 146µs pour le GPU
pour la multiplication, environ 3.94 ms pour le CPU et 0.11ms pour le GPU
