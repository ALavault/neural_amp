# DIAGNOSIS_vs100r_192k : pourquoi le modèle entraîné à 192 kHz s'écarte de celui à 48 kHz

Voie produit (D-PRODUCT-001). Régime strict : l'étude a une pré-inscription
(`demo/capture/valvestate/predictions_192k.md`). Sections 1 à 3 écrites le 2026-10-03 avant
toute mesure nouvelle ; scripts et tables dans `diagnosis/vs100r_192k/`.

## 1. Écart

Métrique : ESR à 48 kHz sur la région notée de chacune des 8 prises de test, sortie du modèle à
192 kHz décimée par `resample_poly(·, 1, 4)` (`demo/capture/valvestate/eval_192k.json`,
lecture descriptive : une graine par bras).

| prise | gain, contour | prises d'entraînement au même couple | A48 | B192 | ln(B/A) |
|---|---|---|---|---|---|
| 2 | 7,5 ; 2,5 | 1 | 0,0684 | 0,0687 | +0,00 |
| 3 | 5 ; 10 | 1 | 0,0221 | 0,0085 | −0,95 |
| 14 | 5 ; 7,5 | 1 | 0,0186 | 0,0088 | −0,74 |
| 16 | 2,5 ; 7,5 | **0** | 0,1100 | 0,3506 | **+1,16** |
| 17 | 5 ; 0 | 1 | 0,0065 | 0,0070 | +0,07 |
| 30 | 10 ; 5 | 1 | 0,0519 | 0,0275 | −0,64 |
| 33 | 7,5 ; 2,5 | 1 | 0,0162 | 0,0094 | −0,54 |
| 36 | 2,5 ; 5 | 2 | 0,0216 | 0,0138 | −0,45 |

- Moyenne des ln ESR : A48 −3,58, B192 −3,84 (B192 meilleur de 0,26). Écart-type entre prises
  de ln(B/A) : 0,67.
- Niveaux par bande : B192 plus proche du réel au-dessus de 2,5 kHz (|e| à 16 kHz 0,26 contre
  0,53 dB), un peu moins proche entre 100 Hz et 2 kHz (0,4–0,5 contre 0,2–0,3 dB).
- **Dispersion** : un seul run par bras. La dispersion entre graines sur le VS100R est inconnue
  jusqu'aux graines 43 et 44. Ordres de grandeur dans le dépôt : sur le Big Muff, l'écart-type
  de ln ESR entre graines vaut 0,27 à 0,53, et la graine multiplie surtout l'erreur de tous les
  segments par un même facteur (`DIAGNOSIS_seeds.md`). Sur le Marshall paramétrique (G2.2),
  l'écart entre deux bras sur un même réglage non vu varie d'une partition à l'autre avec un
  écart-type de 0,17 à 1,10.
- **Conclusion du §1** : l'écart moyen (−0,26) est dans la dispersion entre graines connue ;
  non distinguable du bruit, on ne l'explique pas. L'écart propre à la prise 16 (+1,16, contre
  −0,45 à −0,95 sur cinq autres prises) est un effet prise × run. Sa taille est comparable à la
  variabilité d'un réglage non vu entre runs dans G2.2. Le diagnostic porte sur sa nature, sans
  prétendre l'attribuer à la fréquence d'entraînement avant les graines 43 et 44.

## 2. Comparabilité

- Jeu de test : mêmes 8 prises, même région notée (`product_vs100r_eval.scored`).
- Référence : la capture à 192 kHz décimée par `resample_poly(·, 1, 4)`, la décimation qui a
  produit `explore_48k`.
- Protocole : mêmes drapeaux, même graine 42, `--split-seed 925`, 15 000 pas, dernier
  checkpoint pour les deux. Calendrier rejoué : changements de taux enregistrés à 7 799 → 7 899
  et 10 699 → 10 788 autour des pas 7 830 et 10 701 (cadence d'enregistrement).
- Budget réel : B192 a vu environ 4 fois moins de durée de son (43 époques contre 172) ; c'est
  une différence de protocole déclarée, pas un défaut.
- **Instrument** : la chaîne d'évaluation redonne au millionième les ESR par prise enregistrées
  le 2026-09-27 pour A48 (`demo/capture/valvestate/eval_vs100r_gc_film_joint_s925_r2.json`).
- Taille du jeu d'évaluation : 8 prises, dont une seule à couple non vu. Toute lecture sur ce
  couple repose sur une prise.

## 3. Hypothèses

Déjà lu (et hypothèses que cela a suggérées, marquées comme telles) : extraits de 6 s de la page
d'écoute (prises 16, 17, 30). À la prise 16, B192 est plus fort de 3 à 8 dB entre 125 Hz et
4 kHz, et A48 aussi plus fort que le réel, de 1,6 dB. Cela suggère une erreur de niveau propre
au réglage. Sur la région notée entière, la prise 17 ne montre plus d'écart : la page avait choisi
une fenêtre de 6 s.

Écrites avant toute mesure nouvelle. Provenance : C = Claude, X = Codex (`task` en lecture
seule, faits sans hypothèses, 2026-10-03).

**H1, généralisation du conditionnement (C, X1).** Au couple non vu (2,5 ; 7,5), le FiLM de
B192 interpole autrement que celui de A48 et place les portes dans un autre régime. Mesures :
(a) gain oracle par prise ; (b) sortie sur l'audio de la prise 16 avec les commandes des couples
voisins vus (2,5 ; 5) et (5 ; 7,5). Prédictions : (a) un gain scalaire seul ramène l'ESR de B192
sur la prise 16 sous 0,15 (erreur surtout de niveau) ; (b) avec les commandes voisines, l'ESR de
B192 sur cette prise est plus basse qu'avec les vraies, alors que celle de A48 ne l'est pas.

**H2, variabilité d'entraînement (C, X6).** L'écart prise 16 est de la même nature que l'écart
entre runs à un réglage non vu, sans lien avec la fréquence. Mesure décisive : les graines 43 et
44 (en cours). Mesure provisoire : comparer 1,16 aux écarts-types par réglage non vu de G2.2.
Prédiction : chez au moins une graine de plus, le signe ou la taille de ln(B/A) à la prise 16
change nettement (de plus de 0,8).

**H3, exposition et optimisation (C, X2).** B192 a vu 4 fois moins de son et serait moins
convergé. Mesure : pente relative de la perte de validation sur les 2 000 derniers pas, par bras.
Prédiction : pente de B192 au moins 3 fois celle de A48, et B192 moins bon sur la plupart des
prises. Le §1 contredit déjà la seconde partie (B192 meilleur sur 5 prises sur 8).

**H4, objectif dominé par l'ultrason (C, X3).** Le terme log-magnitude de la MR-STFT pèse les
cases au-dessus de 24 kHz. Mesure : sur les prises de test à 192 kHz, part de la MR-STFT de B192
venant des cases au-dessus de 24 kHz. Prédiction : plus de 50 %.

**H5, débuts de segment (X4).** Avec des segments de 0,75 s, B192 voit 4 fois plus de départs à
état nul. Il compenserait des transitoires de départ, et se tromperait en lecture continue.
Mesure : inférence par morceaux de 0,75 s remis à zéro, contre inférence continue, pour B192 (et
3 s pour A48). Prédiction : sur la prise 16, l'ESR de B192 par morceaux est inférieure d'au moins
10 % à l'ESR continue.

**H6, instrument et alignement (C, X7).** Un décalage temporel fractionnaire différent entre les
bras. Mesure : retard oracle par prise (±2 échantillons, pas de 0,05). Prédiction : il ramène
ln(B/A) de la prise 16 sous 0,3.

**H7, paramétrisation ZOH (X5).** Décaler Δ ne rend pas les coefficients d'entrée identiques ; les
réponses des SSM divergeraient. Mesure : réponses fréquentielles des couches SSM en unités
physiques. Rang bas : elle n'explique pas un effet concentré sur un réglage.
