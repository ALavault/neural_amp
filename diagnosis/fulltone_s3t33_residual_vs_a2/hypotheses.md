# Hypothèses — écart résiduel S3 t33 (0,140) contre A2 (0,064), Fulltone

Écrites avant toute mesure, 2026-09-16 (voir `date` ci-dessous). Toutes les mesures sont
en lecture seule sur les runs `m4_memory_fulltone_s3t33_seed{0,1,2}_v1` et
`m4_fulltone_b0_seed{0,1,2}_v1`, sur CPU.

Faits de départ (rapport M4_MEMORY) : S3 t33 test 0,140 / 0,145 / 0,133, fenêtres
d'entraînement 101–200 : 0,104 / 0,115 / 0,113 ; A2 test 0,067 / 0,064 / 0,041,
fenêtres 0,039 / 0,032 / 0,017. Écart médian test 0,076, écart médian sur les fenêtres
d'entraînement 0,081. Bornes LS (train → test) : FIR 65 taps 0,122, Hammerstein 63 taps 0,091.

## H1 — Optimisation incomplète (famille : optimisation)
Mécanisme : 200 pas d'Adam à lr 0,004 sur 2 fenêtres ne suffisent pas à amener la cascade
FIR33 → spline → FIR33 à son optimum ; la validation oscille de ±0,03 entre évaluations.
Mesure : (a) FIR LS 65 taps ajusté sur les 400 fenêtres vues, évalué sur les fenêtres
101–200 : un modèle non linéaire contenant l'identité doit au moins l'égaler ; (b) borne
atteignable de la classe propre de S3 : moindres carrés alternés post-FIR ↔ spline
(pré-FIR, slow, résidu figés au checkpoint), ajustés sur les 400 fenêtres, évalués sur les
fenêtres 101–200 et sur test.
Prédiction si vraie : le refit alterné baisse l'ESR des fenêtres 101–200 d'au moins 0,02
(0,104 → ≤ 0,085) et l'ESR test d'au moins 0,015.

## H2 — Réponse linéaire dépendante du niveau (famille : classe / capacité)
Mécanisme : la Full-Drive 2 est une topologie Tube-Screamer (écrêtage à diodes dans la
boucle de contre-réaction avec condensateur) : la forme de la réponse en fréquence change
avec le niveau. Une cascade FIR → non-linéarité statique → FIR ne peut pas le reproduire ;
un WaveNet profond peut.
Mesure : FIR LS 65 taps x → cible par tranche d'enveloppe causale de l'entrée (train),
réponses normalisées au gain à 200 Hz ; ESR train/test d'un banc de FIR commuté par niveau
contre FIR unique ; même estimation sur la sortie d'A2 et de S3 t33.
Prédiction si vraie : les réponses par tranche diffèrent de ≥ 3 dB en 100–300 Hz après
normalisation ; le banc commuté gagne ≥ 0,02 en test sur le FIR unique ; A2 reproduit la
variation de forme et S3 t33 ne la reproduit pas (< 1 dB).

## H3 — Régime de niveau (famille : données / généralisation)
Mécanisme : l'excès d'erreur se concentre aux bas niveaux (pour 17 taps, ESR local 0,52 sous
|x| < 0,05), régime où la cible est surtout du bruit de fond ou une saturation résiduelle.
Mesure : ESR local par tranche d'enveloppe et part de l'erreur totale, S3 t33 contre A2, sur
test ; gain variable par segment de 50 ms appliqué à S3.
Prédiction si vraie : ≥ 50 % de l'excès S3 − A2 est dans les tranches |x| < 0,1 ; le gain
par segment retire ≥ 0,02.

## H4 — Transfert et sélection de checkpoint (famille : généralisation / mesure)
Mécanisme : la guitare de test (Ibanez) diffère de celle d'entraînement (gtr2), la
sélection se fait sur 5 points de validation (guitare SG).
Mesure : écart S3 − A2 sur les fenêtres 101–200 contre écart sur test, par seed.
Prédiction si vraie : écart test ≥ écart fenêtres + 0,03.

## H5 — Localisation spectrale (famille : mesure, descriptive)
Mesure : erreur par bande (0–100, 100–300, 300–1k, 1–3k, > 3k Hz) sur test et sur les
fenêtres 101–200 pour S3 t33, A2, identité, FIR LS 65, Hammerstein.
Prédiction : l'excès résiduel garde la répartition de la cible (≈ 60 % en 100–300 Hz).

## H6 — Résidu de S3 encore linéairement récupérable (famille : capacité linéaire)
Mécanisme : 65 échantillons de mémoire ne suffiraient toujours pas.
Mesure : FIR LS 257 taps x → (cible − S3) ajusté sur train, évalué sur test.
Prédiction si vraie : ≥ 30 % de l'énergie du résidu de S3 est retirée en test.

## Contrôles
- Reproduire 0,1041 (S3 t33 seed 0, fenêtres 101–200) et l'ESR test à partir des
  prédictions sauvées avant toute conclusion ; trancher entre 0,062 (ancien diagnostic) et
  0,039 (résumé) pour A2 sur les fenêtres.
- Vérifier sur le checkpoint t33 que résidu et modulation lente sont inertes ; occupation de
  la spline (histogramme de drive·pre(x)+offset face au pas de nœud 0,25).
2026-09-16T08:07:52+02:00
