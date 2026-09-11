# Rodent : diagnostic du détecteur de marqueurs

Analyse des cinq sources trainval Rodent et Fuzzy Logic, début et fin (20 fenêtres).
Script reproductible : `inspect_rodent_markers.py`; mesures :
`rodent_marker_diagnosis.json`; figure : `rodent_marker_diagnosis.png`.
Aucun signal corrigé, aucune archive test ouverte, aucun entraînement.

## Résultat

Le maximum absolu n'identifie pas l'arrivée du marqueur Rodent. La réponse est
déjà importante autour de l'impulsion dry; elle comporte une excursion négative
proche de -0,13, puis un rebond positif vers +600 à +720 échantillons. Ce rebond
atteint 0,1816 à 0,1879 et devient le maximum choisi par le détecteur.
L'amplitude minimale 0,05 est satisfaite. Le décalage du maximum 670–720
échantillons ne démontre donc pas un retard de transport de 14–15 ms.

Le premier franchissement du seuil exploratoire max(0,005, 10 écarts-types du
bruit initial) est entre -12 et -6 échantillons du marqueur dry pour Rodent.
Fuzzy Logic franchit ce seuil vers -10, mais son pic coïncide avec le dry.
Ce seuil est un diagnostic, pas une nouvelle règle d'acceptation. Les réponses
avant le pic dry ne suffisent pas à conclure à une anticipation physique :
filtrage de capture et conventions de repérage peuvent intervenir.

Comparaison des formes wet début/fin, fenêtres [-100,+1500] : corrélations
Rodent sans déplacement 0,988–0,994. Une recherche exploratoire limitée à ±20
échantillons sur ces seules réponses donne 0 sur quatre sources, +5 sur
idmt-gtr2. Cela ne prouve pas la tolérance historique ±1; cela contredit
l'interprétation antérieure des différences de maxima comme dérive de délai.
Aucune corrélation sur le contenu musical ni recalage de données réalisé.

## Provenance

Les README embarqués Rodent et DRY-with-markers déclarent deux impulsions de
synchronisation, les mêmes sources dry et le réglage Rodent 10/5/5/Normal.
Le dossier est `V100_F050_D050_MNormal`, les fichiers emploient `M000` et
yt-bass comporte un double point. Le résolveur cible un unique membre par source
dans ce dossier; longueurs et fréquences correspondent. Ces vérifications
supportent le mapping, sans prouver l'alignement musical à l'échantillon.
Les README ne donnent ni correction de délai précise ni réponse attendue de
l'impulsion après pédale.

## Décision proposée

V9 reste INVALID au sens de sa règle figée, mais la déclaration antérieure
« non-conformité archive démontrée » était trop forte. Le défaut établi est
l'assimilation du maximum wet à l'instant du marqueur.

Avant v10 : définir prospectivement une mesure robuste de localisation de la
réponse, valider ses erreurs sur fixtures avec déformation d'impulsion, bruit,
retards connus et dérive, puis la confronter aux deux marqueurs trainval. Ne pas
décaler Rodent de 700 échantillons sur la foi du maximum; ne pas élargir la
tolérance à 720 uniquement pour faire passer les données. Une référence de
capture indépendante reste nécessaire pour garantir un alignement absolu ±1.
