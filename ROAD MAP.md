ROAD MAP: 

- On commence par uniquement du static dispatching
- On essaye d'implementer le vol de travail par la suite

2) Distribution du texte
- Tout le monde a acces au texte en entier
- Chaque machine calcule avec le score sur une partie du texte

Resultat (1 pattern, texte 24k char):
Sequential version : 0.12s
Distributed version over 1 core, 1 machine : 0.27s
Distributed version over 8 cores, 1 machine : 0.06s
Distributed version over 8 cores, 8 machines : 0.40s
Distributed version over 64 cores, 8 machines : 0.19s

3) Distribution des patterns
- Chaque machine calcule un unique patterne et tout le texte

Resultat (10 patterns, texte 24k char):
Sequential version : 0.26s
Texte-Distributed over 64 cores, 8 machines : 0.02s
Patern-Distributed over 64 cores, 8 machines : 0.06s

4) Distribution de pairs (texte, pattern)


Recap :
apm : sequentiel (aucune modif)
apm1 : mpi sequentiel
apm2 : texte distribue brutalement
apm3 : patterns distribue brutalement uniquement
apm4 : si nb pattern > comm_size, on distribue sur les patterns uniquement. Sinon on distribue brutalement le texte mais on divise le travail par portion de texte et pattern.
apm5 : 

Next step : 
- Distribue des petites parties de texte
- Bien gerer l'ouverture du/des fichiers