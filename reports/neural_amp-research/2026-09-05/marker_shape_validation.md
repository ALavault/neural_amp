# Validation exploratoire du repérage relatif des marqueurs

Script : `validate_marker_shape.py`; résultats : `validate_marker_shape.json`.
Grille déterministe de 324 cas : retards -12/-5/0/1/5/12, bruit
0/0,001/0,003, gains 0,5/1/-1, rebond tardif 0,8/1,2 et largeur initiale
1/0,7/1,5. Paramètres définis avant l'exécution, aucune optimisation a posteriori.

Détecteur : corrélation normalisée absolue des différences premières de la
réponse précoce [-24,+72], recherche ±32 échantillons; refus si score <0,8 ou
maximum à la frontière. Aucun signal audio n'est déplacé.

Résultats : forme initiale stable, 85/108 acceptés; forme modifiée,
164/216 acceptés. Les 249 acceptés retrouvent exactement le retard injecté.
75 refus. Aucun test ne démontre une robustesse générale hors de cette grille;
les familles de réponses synthétiques restent limitées.

Contre-exemple obligatoire : deux réponses wet partageant un retard absolu de
15 échantillons ont un retard relatif mesuré nul. Le détecteur établit une
stabilité relative, pas la synchronisation absolue dry/wet.

Application diagnostique aux seules réponses des marqueurs trainval (données
déjà autorisées) : Rodent idmt-gtr2 +5 échantillons entre début et fin; les
quatre autres sources 0. Fuzzy Logic : cinq sources à 0. Les signaux bruts
restent inchangés. Les tests du benchmark restent fermés.

## Conséquence pour v10

Cette méthode ne remplace pas honnêtement la gate ±1 absolue. Deux voies sont
à arbitrer avant préenregistrement : obtenir une référence indépendante de
synchronisation de capture, ou définir explicitement un benchmark de paires
publiées avec critère de stabilité relative et incertitude d'alignement.
La seconde voie change le contrat scientifique et requiert une décision
utilisateur. Aucun lock v10 ni entraînement créé. Les résultats développement
v9 restent conservés; leur réutilisation dans une lignée future doit être
spécifiée prospectivement.
