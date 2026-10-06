# Où se loge l'erreur des modèles VS100R ? Niveau, mémoire, désaccord des graines (exploration)

Voie produit, régime exploration, 2026-10-06. Chaque affirmation sera marquée **établie** (mesure
citée) ou **supposée**. Origine : brainstorming sur les signaux synthétiques de la prochaine
campagne. Trois observations hors ligne y étaient proposées pour savoir quelle propriété manque au
signal d'entraînement, avant d'ajouter des signaux.

**Décision que la mesure change** : la priorité des sections ajoutées à la source de la séance au
micro. La source contient déjà niveaux, polarité et salves (`demo/capture/valvestate_mic/`) ; la
mesure dira lesquelles analyser d'abord et lesquelles garder pour une campagne plus longue.

## Mesure (prédictions écrites avant la mesure)

Script : `error_structure.py`. Lecture seule, sur CPU.
- Modèles : A48 (`vs100r_gc_film_joint_s925_r2`, `vs100r48_film_s43`, `vs100r48_film_s44`) et
  bras C (`vs100r48fast_film_s42`, `_s43`, `_s44`).
- Données : section di_test (jamais entraînée) des 35 prises hors faible signal, décimée à
  48 kHz, région notée de `product_vs100r_eval.py`.
- Trames de 10 ms : niveau efficace de l'entrée (dBFS), énergie de la cible et de l'erreur. Une
  ESR par groupe de trames = somme des énergies d'erreur / somme des énergies de cible.

**O1, niveau d'entrée rare.** Densité d'entraînement par classe de niveau : part des trames de
10 ms des entrées `trainval` (31 prises, 150 s chacune) dans chaque classe de 5 dB, de −60 à
−10 dBFS. Corrélation de Spearman entre la densité (log) et l'ESR (log) des classes ayant au moins
200 trames de test, par modèle.
- H_niveau : l'erreur se concentre aux niveaux rares en entraînement ; ρ ≤ −0,5 en médiane sur
  les 6 modèles.
- Contraire : |ρ| < 0,3.

**O2, mémoire après un événement fort.** Trame forte : niveau au-dessus du 90ᵉ centile des trames
de sa prise. Pour les trames de niveau moyen (entre les 30ᵉ et 70ᵉ centiles de leur prise), on
classe le temps écoulé depuis la dernière trame forte : < 50 ms, 50–200 ms, 200–1 000 ms et
≥ 1 000 ms.
- H_mémoire : à niveau courant comparable, l'erreur est plus forte juste après un événement fort.
  ESR(50–200 ms) / ESR(≥ 1 000 ms) ≥ 1,5 en médiane sur les modèles.
- Contraire : rapport entre 0,8 et 1,25.

**O3, désaccord des graines.** Par bras (A48, C) et par trame : désaccord = moyenne sur les
graines de l'énergie de (sortie − moyenne des sorties), erreur = moyenne sur les graines de
l'énergie de (sortie − cible), toutes deux divisées par l'énergie de la cible, ce qui retire
l'effet du niveau. Spearman entre désaccord et erreur sur les trames dont la cible dépasse
−50 dBFS, par prise, puis médiane sur les prises.
- H_désaccord : le désaccord prédit l'erreur ; ρ ≥ 0,5 pour les deux bras.
- Contraire : ρ < 0,2.

Ces hypothèses ne s'excluent pas. Lecture : chacune « soutenue », « contraire » ou
« intermédiaire ».

**Limites connues d'avance** :
- L'ESR brute contient le décalage d'environ un échantillon de A48 (`DIAGNOSIS_vs100r_192k.md`),
  qui pèse surtout dans les aigus. Les trois observations comparent des groupes de trames d'un
  même modèle, donc ce biais est commun, mais il peut écraser les différences.
- La guitare lie niveau, attaque et décroissance : O2 ne contrôle que le niveau courant, pas le
  contenu spectral.

## Résultats (`error_structure.json`, 6 modèles, 35 prises)

| | lecture préinscrite | valeur (médiane des modèles) |
|---|---|---|
| O1, niveau rare | intermédiaire | ρ = +0,45 (seuil de H_niveau : ≤ −0,5) |
| O2, mémoire après un fort | intermédiaire | rapport 0,24 (seuil de H_mémoire : ≥ 1,5) |
| O3, désaccord des graines | intermédiaire | ρ = 0,52 (A48), 0,27 (C) |

- **O1 (établi) : sens opposé à H_niveau.** L'erreur est la plus forte aux niveaux d'entrée les
  plus fréquents en entraînement : −35 à −25 dBFS (ESR 0,030 à 0,038, contre 47 % des trames
  d'entraînement). Elle est plus faible aux niveaux rares : 0,014 à 0,036 sous −40 dBFS, 0,008 à
  0,010 de −20 à −10 dBFS. Les six modèles ont le même profil (ρ de +0,37 à +0,59).
- **O2 (établi) : sens opposé à H_mémoire.** À niveau courant moyen, l'erreur est environ quatre
  fois plus faible dans les 50–200 ms qui suivent une trame forte (ESR ≈ 0,010) que plus d'une
  seconde après (≈ 0,045). Rapport de 0,22 à 0,27 sur les six modèles.
- **O3 (établi)** : le désaccord des graines prédit l'erreur modérément pour A48 (0,52), faiblement
  pour C (0,27). Aucun des deux bras n'atteint 0,5 avec l'autre.
- Les règles préinscrites classent O1 et O2 en « intermédiaire » : elles ne prévoyaient pas
  d'inversion de signe. Le classement est rapporté tel quel ; l'inversion est le fait marquant.

**Supposé, non mesuré ici.** Une explication commune possible à O1 et O2 : la part du bruit propre
de l'ampli, imprévisible, dans l'énergie de la cible. Le bruit de la cible pèse plus là où la
sortie est faible pour un niveau d'entrée donné, loin de tout événement fort. Autre piste, non
exclusive : les niveaux moyens tombent dans le coude de saturation, où la forme exacte de la courbe
compte le plus. Mesures qui départageraient, **à tester** :
- ESR par classe de niveau de la cible, comparée à la part du bruit mesuré de l'appareil
  (`demo/capture/valvestate/noise.json`) ;
- la même lecture O1 sur les sinus glissants à trois niveaux, sans bruit de jeu.

**Conséquence pour la campagne (supposé)** : rien ne désigne les niveaux rares ni la mémoire
après un fort comme le manque principal. La section « niveaux » de la source au micro (extrait de
−22 dBFS environ, à −12, −6 et +6 dB) couvre déjà la zone −35 à −16 dBFS où l'erreur se concentre.
Les salves de récupération restent utiles comme contrôle, pas comme priorité.

## Test du bruit de l'appareil (prédiction écrite avant la mesure, 2026-10-06)

Mesure : `noise_share.py`, modèles `vs100r_gc_film_joint_s925_r2` et `vs100r48fast_film_s42`, mêmes
prises et trames. Les trames sont classées par niveau de la **cible** (classes de 5 dB). Pour
chaque classe, on compare l'ESR observée à un plancher de bruit, qu'aucun modèle nourri du
bouclage ne peut prédire :

ESR_bruit = (1 − part cohérente) × puissance de bruit de la prise / puissance moyenne de la cible
dans la classe.

Le bruit est `fx_noise_fullband_dbfs` et la part cohérente `coherent_share`
(`demo/capture/valvestate/noise.json`, par prise).
- **H_bruit** : dans les classes où l'ESR dépasse 0,02, ESR_bruit vaut au moins la moitié de l'ESR
  (médiane des classes). Le bruit explique alors l'essentiel de l'erreur là où elle est forte.
- **Contraire** : ESR_bruit sous un dixième de l'ESR dans ces classes.
- Complément pour O2 (descriptif) : écart de niveau moyen de la cible entre les trames moyennes à
  50–200 ms d'un fort et celles à plus d'une seconde.

### Résultat du test du bruit (`noise_share.json`)

- **H_bruit : contraire (établi).** Dans les classes où l'ESR dépasse 0,02, le plancher de bruit en
  explique 1,3 % en médiane (0,7 et 1,8 % selon le modèle). Même aux cibles les plus faibles
  (−65 à −55 dBFS), il n'en explique que 4 à 17 %.
- **L'erreur relative monte fortement quand la cible faiblit (établi).** ESR par classe de niveau
  de cible, modèle A48 graine 42 : 0,015 à −20 dBFS, 0,041 à −25, 0,11 à −40, 0,78 à −60,
  2,2 à −65 dBFS. Le bras C (graine 42) est environ deux fois moins bon que A48 sur les trames
  faibles : 0,34 contre 0,14 à −50 dBFS, 5,0 contre 2,2 à −65. Il est équivalent au-dessus de
  −30 dBFS.
- **O2 n'est pas un effet de niveau de cible (établi)** : les trames moyennes à 50–200 ms d'un fort
  et celles à plus d'une seconde ont la même puissance moyenne de cible (écart 0,1 dB).

**À tester** (nés de cette mesure) :
- l'erreur propre du modèle à sortie faible : énergie de sa sortie sur la section `silence` de la
  source, face au bruit de l'appareil, et son spectre (résidu continu, ronflement, souffle) ;
- pourquoi le bras C est moins bon à faible niveau. L'entraînement sans TF32 et à la FFT de 288 000
  a changé quelque chose que l'ESR globale (dominée par les trames fortes) ne voyait pas.

## Trames faibles : bras C face à A48, et part d'une composante continue (prédictions écrites avant la mesure, 2026-10-06)

Mesure : `quiet_frames.py`, les 6 modèles (A48 et C, graines 42, 43 et 44), mêmes prises et trames
que `noise_share.py`. Trames faibles : niveau de cible sous −40 dBFS.
- **Q1, bras C moins bon à faible niveau** : ESR des trames faibles, rapport C / A48 par graine.
  Soutenue si le rapport vaut au moins 1,5 pour les trois graines ; contraire s'il est entre 0,67
  et 1,5 pour au moins deux graines.
- **Q2, composante continue** : écart de moyenne (sortie − cible) par prise, sur la région notée.
  Son énergie par trame, longueur de trame × écart², est rapportée à l'énergie d'erreur des trames
  faibles. Soutenue si elle en explique au moins la moitié (médiane des modèles) ; contraire si
  moins d'un dixième.
- Descriptif : puissance de la sortie de chaque modèle sur la section `silence` (entrée quasi
  nulle), face au bruit de l'appareil (`noise.json`).

### Résultats des trames faibles (`quiet_frames.json`)

| modèle | ESR des trames faibles | part du décalage continu | sortie sur le silence (dBFS) |
|---|---|---|---|
| A48 s42 | 0,15 | 18 % | −60 |
| A48 s43 | 0,29 | 2 % | −70 |
| A48 s44 | 0,17 | 2 % | −71 |
| C s42 | 0,38 | 36 % | −55 |
| C s43 | 0,23 | 68 % | −54 |
| C s44 | 0,60 | 55 % | −51 |

Bruit de l'appareil, même section : −67 dBFS (médiane des prises).

- **Q1 : intermédiaire (établi).** Rapports C / A48 de 2,5, 0,8 et 3,6 selon la graine : C est nettement
  moins bon à faible niveau pour deux graines sur trois, meilleur pour la graine 43.
- **Q2 : intermédiaire (établi)**, médiane 0,27. Le partage est net entre bras : le décalage continu
  explique 2 à 18 % de l'erreur faible de A48, mais 36 à 68 % de celle de C. Sur une entrée presque
  nulle, les trois modèles C sortent 12 à 16 dB au-dessus du bruit de l'appareil. Pour A48, seule la
  graine 42 le dépasse (de 7 dB).
- **Supposé** : le bras C (FFT de 288 000 sans TF32, adopté pour les nouveaux prototypes) apprend un
  décalage continu de sortie que A48 n'apprend pas ou peu. Ni la lecture P, dominée par les trames
  fortes, ni l'oreille (une composante continue ne s'entend pas) ne le voient. Le cas de graine 42 A48
  montre que A48 n'en est pas exempt ; trois graines par bras ne permettent pas d'attribuer l'écart au
  réglage plutôt qu'au hasard.
- **Lien avec la piste impaire** : le modèle impair n'a aucun biais et ne peut donc pas avoir de
  décalage continu à entrée nulle. Sa comparaison au bras C, préinscrite et inchangée, inclura cet
  avantage ; la lecture des trames faibles sera à faire en descriptif quand ses runs existeront.
- **À tester** : retirer la moyenne de sortie (ou un passe-haut très bas) des modèles existants et
  relire l'ESR des trames faibles et la lecture P. Si l'écart C / A48 disparaît, le décalage est
  toute l'histoire ; le correctif serait alors un passe-haut de sortie, à préinscrire.

## Sans composante continue (prédictions écrites avant la mesure, 2026-10-06)

Mesure : `dc_removed.py`, les 6 modèles. Un même passe-haut causal à un pôle (R = 0,9995, coupure
vers 3,8 Hz à 48 kHz) est appliqué à la sortie et à la cible, avant la région notée. On relit
l'ESR des trames faibles (cible sous −40 dBFS) et la lecture P (moyenne par couple des ln ESR,
couple non vu exclu, 35 prises).
- **D1** : sans composante continue, le rapport C / A48 des trames faibles revient entre 0,67 et 1,5
  pour au moins deux graines sur trois. L'écart de C à faible niveau est alors son décalage continu.
  Contraire : il reste au-dessus de 1,5 pour au moins deux graines.
- **D2** : l'écart de lecture P entre C et A48 (moyenne des trois graines) change de moins de 0,02,
  car P est dominée par les trames fortes. Contraire : il change de plus de 0,05.
