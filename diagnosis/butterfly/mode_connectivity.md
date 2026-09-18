# Les enfants divergents finissent-ils dans le même bassin que le témoin ?

Interpolation linéaire des poids, lecture seule, CPU.
Scripts : `diagnosis/butterfly/mode_connectivity.py`, `diagnosis/butterfly/weight_distance.py`.
Sorties : `mode_connectivity.json`, `weight_distance.json`.

## Hypothèse testée

Connectivité linéaire des modes (Frankle et al., 2020) : en interpolant les poids entre deux
solutions, θ(α) = (1−α)·θ_témoin + α·θ_enfant, une bosse d'erreur au-dessus de la droite joignant
les extrémités — la barrière — signale deux bassins distincts ; son absence signale un seul bassin,
les deux solutions étant reliées par un chemin droit sans dégradation.

**Prédiction posée avant la mesure : pas de barrière pour les enfants « rejoue », barrière pour les
enfants « décide ».** Autrement dit, la connectivité devait fournir une signature séparant les deux
bras.

L'alignement préalable des canaux, habituel dans ce type de mesure, est ici inutile : les enfants
descendent d'un même parent par une perturbation d'un pas float32, pôles et canaux sont donc dans
le même ordre par construction. Les points de contrôle ne portent aucune statistique de
normalisation : rien à recalibrer après interpolation. La garde de polarité nie la couche de
sortie ; le script refuse d'interpoler deux extrémités dont les nombres de retournements diffèrent
de parité, ce qui ferait passer le chemin par un modèle de sortie nulle et produirait une fausse
barrière.

## Méthode

Grille α ∈ {0 ; 0,25 ; 0,5 ; 0,75 ; 1}, ESR de test du protocole (moyenne sur les 12 segments de
5 s) à chaque point, pour les huit enfants non témoins de la fourche 100.
Barrière = max_α [ESR(α) − droite(α)]. Pour distinguer une vraie bosse d'un artefact d'un segment,
on compte aussi les segments dont le milieu dépasse **leur propre** droite. Distance L2 dans
l'espace des poids en regard, globale puis par famille de paramètres.

Contrôle du chemin d'exécution : à α = 0 le script mesure 0,03413 contre 0,03414 enregistré pour le
témoin — la mesure reproduit la valeur du protocole.

## Observations

| enfant | ESR à α = 0 ; 0,25 ; 0,5 ; 0,75 ; 1 | barrière | segments au-dessus | ‖Δθ‖ |
|---|---|---|---|---|
| décide k1 | 0,0341 0,0458 **0,0567** 0,0566 0,0510 | +0,0141 (+41 %) | 12/12 | 8,22 |
| décide k2 | 0,0341 0,0349 0,0358 0,0367 0,0378 | +0,0000 (0 %) | 8/12 | 6,07 |
| décide k3 | 0,0341 0,0368 0,0388 0,0391 0,0382 | +0,0027 (+8 %) | 12/12 | 8,37 |
| décide k4 | 0,0341 0,0351 0,0359 0,0362 0,0365 | +0,0006 (+2 %) | 9/12 | 6,39 |
| rejoue k1 | 0,0341 0,0343 0,0342 0,0341 0,0339 | +0,0002 (+1 %) | 8/12 | 5,33 |
| rejoue k2 | 0,0341 0,0343 0,0343 0,0340 0,0336 | +0,0004 (+1 %) | 9/12 | 6,03 |
| rejoue k3 | 0,0341 0,0348 0,0353 0,0349 0,0339 | +0,0013 (+4 %) | **12/12** | 7,77 |
| rejoue k4 | 0,0341 0,0343 0,0345 0,0339 0,0333 | +0,0007 (+2 %) | 11/12 | 6,48 |

**Positif.** L'enfant le plus divergent, décide k1, montre une vraie barrière : au milieu du chemin
l'ESR vaut 0,0567, pire que le témoin (0,0341) **et** pire que l'enfant lui-même (0,0510), et la
bosse est présente dans les 12 segments sur 12. Une perturbation d'un pas float32 à l'époque 100
suffit donc à faire terminer un run dans un bassin séparé, au sens de la connectivité linéaire.

**Négatif : la prédiction est réfutée en tant que discriminateur.** La barrière ne sépare pas les
bras. Elle suit la distance parcourue dans l'espace des poids :

- ρ(‖Δθ‖, barrière) = +0,90 (p = 0,002, n = 8) ;
- ρ(‖Δθ‖, nombre de segments au-dessus) = +0,91 (p = 0,001) ;
- ρ(barrière, ESR final) = +0,43 (p = 0,29) — pas de lien établi.

L'enfant « rejoue » le plus éloigné (k3, ‖Δθ‖ = 7,77) présente lui aussi 12 segments sur 12
au-dessus de leur droite, alors que son ESR final ne diffère du témoin que de 0,6 %. Inversement
décide k2, dont l'ESR final excède celui du témoin de 11 %, n'a aucune barrière.

**Négatif.** Les distances ne séparent pas non plus les bras : moyenne 7,26 (décide) contre 6,40
(rejoue), plages entièrement chevauchantes (6,07–8,37 contre 5,33–7,77). Les deux bras diffèrent
d'un facteur 17 en dispersion d'ESR et de presque rien en distance parcourue.

**Négatif.** La décomposition par famille de paramètres (distance relative à la norme de la famille
chez le témoin) ne sépare pas davantage : le bras « décide » bouge un peu plus dans six familles
sur sept, mais chaque plage chevauche celle de l'autre bras — par exemple sur les convolutions,
0,201–0,305 (décide) contre 0,178–0,277 (rejoue). Indicatif seulement, la structure interne est
très inégale : les taux de décroissance `log_A_real` (0,16–0,28) et les convolutions (0,18–0,31)
bougent dix fois plus, relativement, que les pas de temps `log_dt` (0,014–0,023) et les fréquences
de pôles `A_imag` (0,021–0,030). Après l'époque 100, l'entraînement ne re-règle pratiquement plus
les fréquences des pôles ; il ajuste les décroissances et le mélange.

## Ce que cela permet de conclure

- Un changement d'ordre des opérations flottantes suffit à faire changer de bassin : décide k1 est
  séparé du témoin par une barrière franche, visible dans chaque segment de test.
- Mais la connectivité linéaire **ne fournit pas la signature cherchée**. La hauteur de barrière est
  prédite par la distance parcourue (ρ = 0,90), pas par le fait de prendre ses propres décisions, et
  elle n'est pas liée de façon établie à l'écart d'ESR final (ρ = 0,43, n. s.).
- Les décisions pilotées par la validation ne déplacent donc pas les poids plus loin, ni dans des
  familles de paramètres différentes. Ce qu'elles changent est l'endroit où l'on s'arrête sur la
  surface d'erreur, pas l'ampleur du trajet. Cela contredit l'idée intuitive d'une divergence qui
  s'accumulerait dans l'espace des poids.

## Ce que cela ne permet pas de conclure

- Quatre enfants par bras, une fourche, un seed. Les ρ ci-dessus portent sur n = 8 et mélangent les
  deux bras ; ils décrivent ces huit runs, ils n'établissent pas une loi.
- La grille α est à pas 0,25 : une barrière plus étroite entre deux points resterait invisible, et
  la hauteur mesurée est une borne inférieure.
- La mesure porte sur l'ESR de **test**, pas sur la perte d'entraînement effectivement optimisée.
  « Bassin » s'entend au sens de la surface d'erreur que l'on évalue, pas exactement de celle que la
  descente a suivie.
- ‖Δθ‖ est une norme euclidienne brute sur des paramètres de natures différentes ; les comparaisons
  entre familles dépendent du paramétrage et ne valent qu'indicativement.

## Complément : où se loge l'excès d'erreur ? (même mesure, sans calcul nouveau)

Rapport de l'ESR de chaque enfant à celle du témoin, segment par segment, aux extrémités du chemin
(champ `esr_per_segment` du JSON) :

| enfant | rapport par segment 0 → 11 | maximum |
|---|---|---|
| décide k1 | 2,34 1,56 1,48 1,58 1,23 1,53 1,56 1,28 0,99 1,09 1,21 1,22 | 2,34 (segment 0) |
| décide k2 | 1,18 1,17 1,10 1,11 1,07 1,13 1,05 1,07 1,01 1,06 1,09 1,14 | 1,18 (segment 0) |
| décide k3 | 1,37 1,11 1,12 1,12 1,04 1,13 1,08 1,08 1,01 1,07 1,10 1,07 | 1,37 (segment 0) |
| rejoue k3 | 0,97 0,97 0,95 0,96 1,00 1,04 1,01 1,02 0,99 1,02 1,02 1,03 | 1,04 (segment 5) |

L'excès de décide k1 est **large** : onze segments sur douze au-dessus du témoin, de 1,2 à 1,6 ×,
avec un maximum à 2,3 × sur le segment 0. Ce n'est donc pas une spécialisation locale — l'enfant
est moins bon presque partout. Rejoue k3, à distance et barrière comparables, reste plat partout
(0,95–1,04). Parcourir une longue distance et franchir une barrière ne dégrade pas en soi ; ce que
fait le bras « décide », c'est arriver à un point moins bon presque partout.

Détail secondaire, non expliqué : le segment 0 est le plus dégradé chez les trois enfants
« décide », alors qu'il n'est pas le plus difficile pour le témoin (ESR 0,0426 contre 0,0813 sur le
segment 4).

## Prochaine expérience discriminante

Rapprocher l'excès relatif par segment de caractéristiques mesurables du contenu de chaque segment
(RMS, facteur de crête, centroïde spectral) : si le segment 0 se distingue par une de ces
caractéristiques et que l'excès des enfants « décide » la suit, la divergence a un prédicat
audible, et c'est lui qu'il faut écouter en priorité. Coût : lecture du fichier de test, aucune
inférence.
