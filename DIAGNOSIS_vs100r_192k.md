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
cours de route (§6) : `real_contour.py`, `bigmuff_lead.py`.

| | mesure | chiffre | verdict |
|---|---|---|---|
| H1a | gain oracle, prise 16 | A48 −1,85 dB, ESR 0,110 → 0,057 ; B192 −3,56 dB, 0,351 → 0,127 | soutenue en partie : le niveau explique une part, pas tout (0,127 > 0,15 manqué de peu dans le bon sens) |
| H1b | prise 16 avec les commandes voisines | B192 : vraies 0,351, (2,5 ; 5) **0,059**, (5 ; 7,5) 1,32 ; A48 : 0,110, 0,093, 1,27 | **soutenue** : avec le contour 5, B192 est 6 fois meilleur ; A48 l'est aussi, de 16 % seulement |
| H2 | graines 43 et 44 | en cours | **indécidable** à ce jour |
| H3 | baisse relative de la validation, pas 12–13 k contre ≥ 14 k | A48 0,64 %, B192 0,28 % | **réfutée** : B192 n'est pas moins convergé |
| H4 | part de la MR-STFT de B192 au-dessus de 24 kHz | terme log-magnitude : 74–75 % (prise 16), 90–91 % (prise 3) ; numérateur de convergence spectrale : 10⁻⁵ | **soutenue** comme fait ; son effet sur l'écart n'est pas mesuré |
| H5 | morceaux de 0,75 s remis à zéro contre lecture continue, prise 16 | B192 0,3516 contre 0,3506 ; morceaux pires ou égaux sur les 8 prises | **réfutée** |
| H6 | retard oracle (±2 échantillons) | prise 16 : A48 +1,45 → 0,091, B192 +0,85 → 0,343, ln(B/A) 1,32 | **réfutée pour la prise 16** ; voir ci-dessous pour les autres prises |
| H7 | — | non faite | rang bas, voir §6 |

Lecture de H6 sur les 7 autres prises : A48 a un retard oracle de +0,75 à +1,45 échantillon
(médiane 0,95 ; prise 2 : +0,25). Compensé, son ESR baisse de 2 à 9 fois (prise 3 : 0,022 →
0,0024 ; prise 30 : 0,052 → 0,0056). B192 a un retard médian de 0,2 échantillon et gagne moins
à la compensation. Après compensation des deux, A48 est meilleur sur 7 prises sur 8, la 8ᵉ (prise
2) étant à égalité.

## 5. Contribution retenue

**Prise 16 : le modèle à 192 kHz invente un effet du contour à gain 2,5 (H1).**
- Prédicat : l'écart se concentre sur la seule prise de test dont le couple gain–contour est
  absent de l'entraînement. À gain 2,5, l'entraînement n'a que les contours 2,5 et 5 (1 et 2
  prises), donc le contour 7,5 est une extrapolation dans cette tranche.
- Mécanisme : dans l'ampli réel, passer du contour 5 à 7,5 ne change presque rien. La sortie
  réelle de la prise 16 est à 0,010 en ESR de celle de la prise 26 (contour 5), dans la
  répétabilité de deux prises au même réglage. Au gain 5, l'écart entre contour 5 et 7,5 vaut
  0,015. Le FiLM de B192 prolonge au contraire la forte variation vue entre les contours 2,5 et 5
  (0,26 à 0,88). Celui de A48 la prolonge à peine.
- Prédiction qu'un correctif devra tenir : ramener l'ESR de la prise 16 sous 0,06, le niveau
  obtenu avec les commandes (2,5 ; 5), sans dégrader les autres prises de plus de 10 %.
- Réserve : une seule prise, une seule graine par bras. Ce comportement peut être propre au run
  (H2) plutôt qu'à la fréquence.

**Autres prises : l'avantage apparent de B192 est un avantage d'alignement (H6).**
- Prédicat : sur 7 prises sur 8, la sortie de A48 est en avance d'environ un échantillon sur le
  réel. Cette avance fait l'essentiel de son ESR, et B192 n'en a pas, ou peu.
- Mécanisme supposé : un décalage temporel fractionnaire dans la capture VS100R, que le modèle à
  48 kHz ne sait pas reproduire et que le modèle à 192 kHz approche (environ 4 échantillons à
  192 kHz). L'avance du ZOH (T1) est écartée comme explication générale : les modèles ZOH du
  Big Muff entraînés à 48 kHz n'ont pas d'avance (retards médians −0,07 à −0,17 échantillon,
  comme les FOH ; `bigmuff_lead.json`).
- Prédiction qu'un correctif devra tenir : avancer les cibles VS100R du décalage mesuré, puis
  réentraîner A48, donne un retard oracle médian sous 0,2 échantillon et une ESR sous 0,01 sur au
  moins 6 prises sur 8.

## 6. Non expliqué et à tester

- L'écart moyen entre les bras (−0,26 en ln ESR) est dans la dispersion entre graines ; H2 se
  tranche avec les graines 43 et 44.
- H4 : l'ultrason pèse 75 à 90 % du terme log-magnitude de B192. Son effet sur la sortie dans la
  bande n'est pas mesuré ; il faudrait un bras à perte limitée à 24 kHz, autre expérience.
- H7 non faite : elle n'explique pas un écart concentré sur un réglage.
- **À tester, né pendant les mesures :**
  - Le décalage entre l'entrée bouclée et l'envoi de la Zen Go. Mesure :
    `demo/capture/valvestate/INTER_INPUT_TEST.md`, prises de l'utilisateur. Prédiction : un
    retard de l'envoi d'environ 20 µs (0,95 échantillon à 48 kHz).
  - La répétabilité des réglages. Deux prises au même couple nominal diffèrent de 0,006 à 0,41
    en ESR (médiane 0,008 sur 115 paires ; trois prises à (2,5 ; 5) : 0,006, 0,071, 0,13). C'est
    un plancher des données : les potentiomètres sont réglés à la main, et à gain faible un petit
    écart de position change beaucoup la sortie. A48 réaligné (0,002 à 0,010) est déjà à ce
    plancher. Mesure : l'ordre chronologique des prises et la position réelle des boutons.
