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

## 4. Mesures et verdict

Scripts et sorties : `diagnosis/vs100r_192k/measures.py` → `measures.json` ; mesures nées en
cours de route (§6) : `real_contour.py`, `bigmuff_lead.py`. Sections 4 à 6 corrigées le
2026-10-03 après relecture Codex (lecture seule), voir la fin du document.

| | mesure | chiffre | verdict |
|---|---|---|---|
| H1a | gain oracle, prise 16 | A48 −1,85 dB, ESR 0,110 → 0,057 ; B192 −3,56 dB, 0,351 → 0,127 | **soutenue** : 0,127 < 0,15 ; un gain scalaire retire 64 % de l'ESR de B192, sans dire d'où vient l'erreur de niveau |
| H1b | prise 16 avec les commandes voisines | B192 : vraies 0,351, (2,5 ; 5) 0,059, (5 ; 7,5) 1,32 ; A48 : 0,110, 0,093, 1,27 | **réfutée telle qu'écrite** : A48 s'améliore aussi (−16 %). Observé : B192 est beaucoup plus sensible au contour (facteur 6) |
| H2 | graines 43 et 44 ; critère écrit : ln(B/A) de la prise 16 varie de plus de 0,8 | en cours | **indécidable** à ce jour |
| H3 | perte de validation moyenne, pas 12–13 k contre ≥ 14 k | baisse A48 0,64 %, B192 0,28 % | **réfutée telle qu'écrite** ; deux moyennes de fenêtres ne prouvent pas une convergence suffisante |
| H4 | part de la MR-STFT de B192 au-dessus de 24 kHz, 2 prises | terme log-magnitude : 74,2–75,4 % (prise 16), 90,4–90,7 % (prise 3) ; numérateur de convergence spectrale : 4·10⁻⁶ à 10⁻⁴ | **soutenue** sur ces deux prises, pour ce terme de la perte ; ni la perte totale ni le gradient ne sont mesurés |
| H5 | morceaux de 0,75 s remis à zéro contre lecture continue | prise 16 : B192 0,3516 contre 0,3506 ; morceaux pires ou égaux sur les 8 prises | **réfutée** pour l'inférence ; un effet des remises à zéro pendant l'entraînement n'est pas exclu |
| H6 | retard oracle (±2 échantillons, pas de 0,05) | prise 16 : A48 +1,45 → 0,091, B192 +0,85 → 0,343, ln(B/A) 1,32 | **réfutée pour la prise 16** |
| H7 | — | non faite | non testée |

Retard oracle sur les 7 autres prises (descriptif) : A48 de +0,25 à +1,35 échantillon (médiane
0,85) ; B192 médiane 0,15. Compensé, l'ESR de A48 baisse d'un facteur 2 à 9,4 sur 6 prises
(prise 3 : 0,022 → 0,0024 ; prise 30 : 0,052 → 0,0056), et de 2 % seulement sur la prise 2.
Après compensation des deux bras, A48 est meilleur sur 7 prises sur 8 ; sur la prise 2, B192 reste
à peine meilleur (0,0669 contre 0,0670). Ce retard est choisi sur la cible de test de chaque
prise : il montre la sensibilité du classement à l'alignement, pas une correction utilisable. Les
gains et retards oracles ne s'additionnent pas.

## 5. Contribution retenue

**Prise 16 : une erreur de niveau du modèle à 192 kHz (H1a).**
- Prédicat : l'écart se concentre sur la seule prise de test dont le couple gain–contour est
  absent de l'entraînement. À gain 2,5, l'entraînement n'a que les contours 2,5 et 5 (1 et 2
  prises).
- Ce qui est établi : B192 est trop fort de 3,6 dB sur cette prise, contre 1,9 dB pour A48. Un
  gain scalaire retire 64 % de son ESR. Avec les commandes (2,5 ; 5), il prédit mieux la cible
  qu'avec (2,5 ; 7,5), d'un facteur 6 ; A48, de 16 %.
- Non établi : le mécanisme. Restent concurrents : la généralisation du conditionnement dans une
  zone sans données, une position réelle des boutons différente de la position notée, la
  variabilité des captures et l'effet propre du run (H2). La sortie réelle de la prise 16 est
  proche de celle de la prise 26, notée contour 5 (ESR 0,010, alignement oracle entier) ; cela
  décrit deux captures, pas une loi physique du contour.
- Prédiction qu'un correctif devra tenir, à confirmer sur d'autres données que ce test inspecté :
  ESR de la prise 16 sous 0,06 sans dégrader les autres prises de plus de 10 %.

**Autres prises : le classement entre les bras dépend de l'alignement temporel (H6, descriptif).**
- Établi : sans compensation, B192 est meilleur sur 5 prises ; avec un retard oracle par prise,
  A48 l'est sur 7. La sortie de A48 a un retard oracle médian de 0,85 échantillon, celle de B192
  de 0,15.
- Non établi : l'origine de ce décalage. Un retard oracle peut absorber une erreur de phase
  propre au modèle ou au réglage. Les candidats restent un décalage de capture entre les deux
  entrées de l'interface et une contribution du ZOH. Les modèles ZOH du Big Muff entraînés à
  48 kHz n'ont pas d'avance (médianes −0,07 à −0,17, comme les FOH ; `bigmuff_lead.json`). Cela
  réfute une avance générale de tout ZOH à 48 kHz, pas une contribution du ZOH sur la VS100R.
  L'instrument diffère en outre : découpe puis décalage circulaire pour le Big Muff, décalage
  puis découpe pour la VS100R.
- Prédiction qu'un correctif devra tenir, avec un décalage fixé hors du test (sur l'entraînement
  ou par la mesure Zen Go) : avancer les cibles VS100R de ce décalage puis réentraîner A48 donne
  un retard oracle médian sous 0,2 échantillon et une ESR sous 0,01 sur au moins 6 prises sur 8.

## 6. Non expliqué et à tester

- Bilan des prédictions du §3 : tenues H1a et H4 (sur deux prises, un terme) ; réfutées telles
  qu'écrites H1b, H3, H5, H6 (prise 16) ; indécidable H2 ; non testée H7.
- L'écart moyen entre les bras (−0,26 en ln ESR) est dans la dispersion entre graines connue
  ailleurs ; H2 se tranche avec les graines 43 et 44, au critère écrit (variation de plus de 0,8
  sur la prise 16).
- H4 : l'effet de l'ultrason sur la sortie dans la bande n'est pas mesuré ; il faudrait un bras à
  perte limitée à 24 kHz, autre expérience.
- H7 non testée : un effet de la discrétisation dépendant du régime pourrait se concentrer sur un
  réglage, donc la concentration ne l'élimine pas.
- **À tester, né pendant les mesures :**
  - Le décalage entre l'entrée bouclée et l'envoi de la Zen Go, par une mesure indépendante de
    l'oracle (`demo/capture/valvestate/INTER_INPUT_TEST.md`, prises de l'utilisateur). Prédiction :
    un retard de l'envoi d'environ 18 à 20 µs (0,85 à 0,95 échantillon à 48 kHz).
  - Le même instrument de retard (même découpe, même décalage, avec prolongement plutôt que
    décalage circulaire) appliqué aux bras ZOH et FOH sur la VS100R et sur le Big Muff.
  - La répétabilité des captures et des réglages. Deux prises au même couple nominal diffèrent
    de 0,0002 à 0,41 en ESR (médiane 0,008, 115 paires non indépendantes ; trois prises à
    (2,5 ; 5) : 0,006, 0,071, 0,13). Au gain 5, contour 5 contre 7,5 : médiane 0,015 (0,010 à
    0,215). Cette dispersion mélange réglages, captures et alignements ; l'attribuer aux
    potentiomètres réglés à la main reste une hypothèse, et ce n'est pas une borne inférieure de
    l'erreur du modèle. Mesure : vérifier les entrées capturées, l'ordre des prises et la
    position réelle des boutons.
  - Les seuils 0,06, 10 % et 0,01 sur 6 prises sur 8 viennent du test déjà inspecté : ils
    demandent une confirmation sur des données indépendantes.

## Corrections du 2026-10-03 (relecture Codex, lecture seule)

Première version des sections 4 à 6 (commit 5a8a223) corrigée :
- H1a était noté « soutenue en partie » avec « 0,127 > 0,15 » : c'est faux, 0,127 < 0,15.
- H1b était noté « soutenue », alors qu'A48 s'améliore aussi : réfutée telle qu'écrite.
- « Invente un effet du contour » et « l'avantage apparent de B192 est un avantage d'alignement »
  dépassaient les mesures : ramenés à ce qui est établi.
- L'élimination du ZOH par le Big Muff était trop large, et l'instrument n'est pas le même.
- Chiffres corrigés : médianes de retard hors prise 16 (0,85 et 0,15, maximum 1,35), facteur 2 à
  9,4 sur 6 prises (pas sur la prise 2), quasi-égalité sur la prise 2, minimum de répétabilité
  0,0002, contour au gain 5 en médiane, parts H4 exactes.
- Le « plancher des données » n'est plus présenté comme établi.

## Mise à jour du 2026-10-03, graine 43 (`measures_s43.json`, `eval_192k.json`)

- **H2 soutenue** au critère écrit d'avance. À la prise 16, ln(B/A) passe de +1,16 (graine 42 :
  A48 0,110, B192 0,351) à −1,12 (graine 43 : A48 0,385, B192 0,125), soit une variation de
  2,28 > 0,8. L'écart de la prise 16 dépend du run, pas de la fréquence d'entraînement. La
  contribution retenue au §5 pour la prise 16 (erreur de niveau de B192) est donc propre au run
  de graine 42.
- Les quatre modèles prédisent mieux la cible de la prise 16 avec les commandes (2,5 ; 5) qu'avec
  les vraies (2,5 ; 7,5) : A48 0,110 → 0,093 et 0,385 → 0,101 ; B192 0,351 → 0,059 et 0,125 →
  0,056. Le contour 7,5 à gain 2,5 est mal prédit par tous les runs, à des degrés différents.
  Cela ne tranche pas entre la généralisation du conditionnement et une position réelle des
  boutons proche du contour 5.
- Alignement, retrouvé à la graine 43 : retard oracle médian hors prise 16 de 0,80 échantillon
  pour A48 contre 0,10 pour B192 (graine 42 : 0,85 et 0,15). Après compensation, A48 est
  meilleur sur 6 prises sur 8 (graine 42 : 7 sur 8).
- Descriptif, sans lecture pré-inscrite (2 graines sur 3) : moyennes des ln ESR A48 −3,58 et
  −3,51, B192 −3,84 et −4,19. Les deux runs A48 sont très proches entre eux (écart 0,07).

## Mise à jour du 2026-10-04, graine 44 et bilan sur trois graines (`measures_s44.json`, `eval_192k.json`)

- **H2, verdict final : soutenue.** À la prise 16, ln(B/A) vaut +1,16 (graine 42), −1,12 (43) et
  −0,45 (44 : A48 0,169, B192 0,108). L'écart de cette prise dépend du run ; il ne départage pas
  les fréquences d'entraînement.
- Les six modèles prédisent mieux la cible de la prise 16 avec les commandes (2,5 ; 5) qu'avec
  les vraies (2,5 ; 7,5). Graine 44 : A48 0,169 → 0,094, B192 0,108 → 0,061. Comme à la mise à
  jour précédente, cela ne tranche pas entre généralisation du conditionnement et position réelle
  des boutons.
- **Alignement, graine 44** : retard oracle médian hors prise 16 de 0,80 échantillon pour A48 et
  de 0,05 pour B192. Après compensation, A48 est meilleur sur 7 prises sur 8 (la prise 16
  exceptée : 0,146 contre 0,104).
- **Sur les trois graines** : retards médians hors prise 16 de 0,85, 0,80 et 0,80 pour A48, et de
  0,15, 0,10 et 0,05 pour B192. Après compensation, A48 est meilleur sur 7, 6 et 7 prises sur 8.
  La lecture pré-inscrite M1 (« B192 plus proche » en ESR brute, d −0,422,
  `predictions_192k.md`) est donc à lire comme un avantage d'alignement, pas de timbre. L'origine
  du décalage d'environ 0,8 échantillon reste à établir hors oracle (test Zen Go, §6).
- H3 ne se relit pas sur ces graines : leurs calendriers placent une division du taux dans la
  fenêtre de mesure (graine 44 : pas 14 181), si bien que la baisse de la validation (A48 3,7 %,
  B192 6,9 %) suit cette division plutôt que la convergence.
