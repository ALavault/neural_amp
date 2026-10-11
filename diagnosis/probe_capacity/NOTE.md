# Capacité des sondes sur Nolly : six pistes (exploration, D-0017)

Régime **exploration** : aucune configuration gelée ni aucune porte. La rigueur vient du dispositif :
B n'entre dans aucun ajustement, les sélections portent sur V, les oracles ajustés sur B sont
étiquetés. En revanche, plusieurs choix expérimentaux ont été faits après avoir vu B (section 2). Chaque affirmation
porte l'une de ces deux marques :
- **établie** : mesure citée, recalculée par un vérificateur indépendant ;
- **supposée** : interprétation non départagée.

Les pistes ont été lancées sur demande de l'utilisateur (« Suit toutes les pistes », research-memory
D-0017). La piste 6 est une exception exploratoire à « aucun réseau appris dans une métrique ».
Aucun résultat ici ne modifie la lecture figée de `diagnosis/probe_metrics/NOLLY_READING.md`.

Fichiers :
- banc commun `harness.py` ;
- une paire `leadN_*.py` / `leadN_*.json` par piste ;
- vérifications adversariales `verify/leadN_verify.py` / `.json` et `verify/crosslead_verify.*` ;
- référence des sondes figées `baseline.json` ;
- rapports structurés des agents : `leads_reports.json` (pistes) et `verify/verifications.json`
  (vérifications), chemins absolus retirés.

Les sections audio en cache (5,9 Go) se régénèrent par `harness.py build` ; elles ne sont pas
commitées.

## 1. Écart

- **Métrique** : capacité H2 = RMS(y_B − q_B) / RMS(y_B). C'est la part de la sortie, **en
  amplitude**, que la sonde laisse inexpliquée. Elle est ajustée sur A = `di_train` et lue sur
  B = `di_test`, à 48 kHz, avec SKIP 2 400 et un plancher de 1e-4. Seuil gelé : 0,3.
- **Valeurs** (`baseline.json`, identiques à `nolly_reading.json` à 1,4e-15 près pour H5 et 3,9e-15 pour VL3), prises 30, 16, 20,
  6 et 7 :

  | Gain | 0 | 2,5 | 5 | 7,5 | 10 |
  |---|---|---|---|---|---|
  | FIR 256 | 0,511 | 0,825 | 0,831 | 0,830 | 0,800 |
  | Hammerstein 5 | 0,476 | 0,794 | 0,803 | 0,800 | 0,747 |
  | Volterra–Laguerre 3 | 0,497 | 0,786 | 0,793 | 0,790 | 0,754 |

- **Dispersion** : l'écart entre périodes de multisinus vaut au plus 3,9e-5 de la puissance de sortie
  sur les 34 prises utilisables (5,7e-5 sur les 42). Il est négligeable pour la capacité, mais ce
  n'est pas une mesure de répétabilité entre rendus : la piste 5 le lit comme une non-périodicité.
  L'écart à 0,3 dépasse de loin ce niveau.

## 2. Comparabilité

- **Instrument établi** : `harness.py` redonne la lecture figée à 1,4e-15 près (H5) et 3,9e-15 près
  (VL3). Il redonne aussi les 34 capacités et les 19 parts de différence, et la linéarité en y des
  5 sondes figées (au plus 1,2e-12). Le contrôle croisé le confirme indépendamment.
- **Unités** : les capacités et parts résiduelles sont en amplitude, sauf celles de la piste 4, en
  **parts de puissance** (capacité²). Les ESR sont des rapports de puissance. Les mêmes chiffres n'ont donc pas le même sens : 0,612 de la piste 3 (amplitude) n'est
  pas 0,612 de la piste 4 (puissance, soit 0,78 en amplitude).
- **V n'est pas une réserve indépendante** : `di_validation` reprend les enregistrements de
  `di_train` (NOLLY_READING.md). Une sélection sur V n'est pas une fuite de B, mais elle est
  optimiste par rapport à une partition indépendante.
- **B a déjà été vu** avant les pistes, dans une étude exploratoire hors dépôt :
  - **la coupure de 300 Hz** (piste 3) a été choisie par un balayage qui lisait B sur la prise 20 ;
  - **les pentes tanh** viennent de la même étude ; la provenance des nœuds et retards de la
    piste 3 n'est pas documentée ;
  - **les références « Wiener-HP300 » et « Wiener-BLA »** citées par la piste 2 normalisaient les
    caractéristiques par le RMS de x_B, une normalisation tirée de B. Recalculée dans le protocole,
    Wiener-HP300 vaut 0,270 au gain 0 et 0,634 au gain 5, au lieu de 0,520 et 0,692. Wiener-BLA
    apprend en outre son préfiltre sur la sortie cible : procédure non linéaire en y ;
  - **8 paires y_ref** (référence 1 ; évaluations 2, 4, 7, 28, 30, 31, 35, 36) avaient été lues sur
    B avant que la piste 1 écrive ses seuils, et non 6 comme elle le déclare.

  Pendant les pistes, trois choix ont aussi suivi une lecture de B, sans que B entre dans un
  ajustement :
  - la variante `yh5_1024` de la piste 1 a été ajoutée après un essai lisant B sur trois paires ;
  - la règle mesurée de la piste 4 a pris sa forme après la lecture de B de variantes antérieures ;
  - la fenêtre A' de la piste 6 a été déplacée après avoir observé H5 = 4,55 sur B (prise 1).

  Toutes les lectures de B ci-dessous sont donc exploratoires.
- **Fuites** : les sept vérificateurs n'ont trouvé aucun chemin de B vers un ajustement, une ridge,
  une normalisation ou une marge dans le code des pistes. Les oracles ajustés sur B sont étiquetés.

## 3. Hypothèses (familles différentes) et mesure discriminante

| Famille | Hypothèse | Mesure discriminante | Piste |
|---|---|---|---|
| Structure | Il manque une saturation entre des filtres, en cascade, pas du degré ou de la mémoire | Bases saturantes après préfiltre ; cascade apprise puis figée, de 1 à 3 étages | 3, 2 |
| État lent | Un état plus lent que la mémoire des sondes porte une grande part de la sortie | Relaxation sans modèle (multisinus, échelons, clics) contre deux appareils synthétiques étalonnés | 5 |
| Plafond de l'appareil | Le plafond est une propriété de l'appareil, mesurable sur multisinus | Part non-BLA D des multisinus contre la FIR conditionnée au niveau sur la DI, à RMS égal | 4 |
| Référence | La différence entre deux sorties demande moins de capacité que la sortie entière | Sonde dont les caractéristiques viennent de y_ref, lue sur la différence d | 1 |
| Base apprise | Une base adaptée à l'appareil représente ce que les bases génériques ratent | Lecture linéaire de caractéristiques figées du bras C, contre une initialisation aléatoire | 6 |

## 4. Mesures et verdicts

### Piste 3, bases de prédistorsion RF (verdict : partielle, confirmé par la vérification)

- **établie** : après un passe-haut fixe à 300 Hz, passer d'une base polynomiale à une base
  saturante (tanh+signe, CPWL, DVR) baisse la capacité d'environ 0,16 au gain 5 :
  - polynôme h5(u)+x : 0,792 ;
  - tanh+signe : 0,634 ;
  - union : 0,612 ;
  - H5 figé : 0,803.

  Les valeurs tanh+signe brutes et hp300, à tous les gains, se recalculent à 1e-15.
- **établie** : la coupure de 300 Hz, contaminée par B, ne coûte rien au gain 5. Une sélection sur V
  seul parmi 100, 200, 300, 500 et 800 Hz retient 300 Hz aux gains 2,5, 5 et 7,5, et B reste à
  0,6336. Aux autres gains, V choisit une autre coupure :
  - **gain 10** : 100 Hz, B = 0,612, soit −0,135 par rapport à H5 (et non −0,09 à 300 Hz) ;
  - **gain 0** : 200 Hz, B = 0,236.
- **établie** : au gain 0, des sondes figées linéaires en y franchissent 0,3 :
  - hp300 avec bases saturantes : 0,260 à 0,280 ;
  - recalcul dans le protocole : 0,270 ;
  - coupure choisie sur V : 0,236.

  Les pistes ne l'avaient pas énoncé. Le critère de la piste, sous 0,6 de A vers B, portait sur le
  gain 5 : il n'y est pas atteint (union 0,612), et seul l'oracle ajusté sur B y descend (0,579). Au
  gain 2,5, union donne 0,545 ; aux gains 7,5 et 10, 0,626 et 0,654.
- **établie, avec sa portée** : la structure propre au RF ajoute au plus 0,022 sur tanh+signe, pour
  la capacité de sortie hp300. Sur la différence 39→30, elle ajoute 0,028 ; sur l'oracle, 0,036. La
  famille est saturée : « all » égale « union » à 0,0007 près sur les sorties.
- **établie** : la BLA de la prise est un mauvais préfiltre (union 0,716, tanh 0,748). Elle est apprise
  sur la sortie cible, donc ces lignes ne sont pas linéaires en y.
- **supposée** : l'écart de 0,03 entre l'oracle et A→B viendrait d'une différence de matériau entre
  A et B. Rien ne sépare la variance d'estimation d'un déplacement de distribution.
- **établie** : la dépendance au niveau est réduite, sans disparaître. Au gain 5, l'étendue entre
  quintiles passe de 0,156 (H5) à 0,034–0,048 (bases saturantes). **supposée** : « aucune compression
  dynamique n'est nécessaire » ; une saturation statique après préfiltre suffit à aplatir les
  quintiles, ce qui ne prouve pas qu'il n'y en a pas.
- **JVM410H** : sur la sortie préampli à G050, tanh hp300 donne 0,486 contre 0,707 pour H5. Contrôle
  plus faible : un seul fichier coupé en A et B disjoints, une seule séance.

### Piste 2, cascade Wiener–Hammerstein apprise puis figée (verdict : partielle)

- **établie** : la cascade apprise sur la prise 20 (G5), lue sur B de la même prise, donne :
  - 1 étage : 0,382–0,395 ;
  - 2 étages : 0,298–0,342 ;
  - 3 étages : 0,260–0,287 (3 graines).

  Ce sont des modèles ajustés à leur cible, donc pas des sondes figées. Les pas d'arrêt et les
  variantes sont choisis sur V seul (code vérifié). Le B de chaque run a été calculé avant la
  sélection : c'est déclaré, et aucun sélecteur ne le lit.
- **établie** : figées depuis la prise 20, avec une lecture linéaire (linéarité en y 3,97e-14 sur une
  paire indépendante), les caractéristiques WH3 battent H5 aux quatre autres gains :

  | Gain | 0 | 2,5 | 7,5 | 10 |
  |---|---|---|---|---|
  | WH3 figée | 0,328 | 0,392 | 0,357 | 0,507 |
  | H5 | 0,476 | 0,794 | 0,800 | 0,747 |
- **établie, avec sa composition** : sur les autres prises G5, la capacité va de **0,236 à 0,308**
  (prise 14 au-dessus de 0,3 ; 0,236 est la prise 42, avec baffle). La « médiane 0,281, 18 prises
  sur 31 sous 0,3 » reflète la composition : 17 des 31 prises sont au gain d'apprentissage.
  **Aucune des 12 prises hors G5 et sans baffle n'est sous 0,3.**
- **établie** : le refus tient aux autres gains. Au gain 10, WH3 figée donne 0,507 et la meilleure
  banque de drives 0,489.
- **établie** : mettre x à l'échelle avant le modèle figé (banque de drives) ne gagne que 0,02 à 0,04.
  **supposée** : le gain agirait comme un drive entre les étages. D'autres explications donnent le
  même résultat : une calibration locale inadéquate, des filtres qui dépendent du réglage, un
  déplacement de biais.
- **supposée** :
  - « les boutons après la non-linéarité sont linéaires » : la capacité de sortie y est peu
    sensible, et les parts de différence de tonalité et de présence laissent encore 0,26 à 0,52 ;
  - « la mémoire et le degré ne sont pas la limite » : chaque étage ajoute aussi 127 échantillons de
    mémoire, et une somme de tanh a un degré infini.
- **reformulée** : la phrase « le refus H2 est une propriété de la classe de sondes, pas de
  l'appareil » s'appuyait sur des modèles ajustés prise par prise, donc des émulateurs. Au gain 7,5,
  le modèle WH3 prise par prise, une seule graine, est lui-même à 0,3005. Forme retenue : *les
  cascades par blocs ajustées à chaque prise descendent à 0,09–0,30 ; ce n'est pas une lecture H2,
  qui porte sur des sondes figées.*
- **statut de règle** : cette cascade est une représentation **apprise** (mélanges, filtres et tanh
  appris), même figée. Elle relève de « aucun réseau appris dans une métrique », comme la piste 6.

### Piste 1, sonde rapportée à la sortie (verdict : partielle)

- **établie** : part non représentée de la différence d = y_eval − y_ref, sur B :

  | Paires | FIR sur y_ref | x5_yh5 (choisie sur V) | H5 depuis x |
  |---|---|---|---|
  | Tonalité | 0,359 | 0,267 | 0,829 |
  | Gain | 0,681 | 0,653 | 0,933 |
  | Baffle | 0,154 (0,046 avec FIR 1024) | 0,144 | — |
  | Répétitions | 1,000 | 1,000 | — |

  Ces nombres sont des **parts de différence**. Elles égalent exactement la couverture
  RMS(r_eval − r_ref)/RMS(d) : ce ne sont pas des capacités de sortie qui profiteraient de y_ref.
- **établie** : la variante x5_yh5 a été choisie sur V seul (part médiane sur V de 0,3242 sur 31
  paires). Elle est aussi la meilleure sur B.
- **établie** : la sonde est linéaire en y (écarts de 1e-14 à 3e-13), mais **conditionnellement à
  y_ref**. Elle reproduit y_ref presque exactement (r_ref/RMS = 2,2e-4), donc la borne figée
  |d_audio − d_probe| ≤ r_ref + r_eval est vide pour elle. La couverture de la différence est la
  seule lecture informative.
- **établie** : les non-linéarités de y_ref aident la tonalité (0,359 → 0,267), pas le gain (au plus
  0,022). La prédiction secondaire est réfutée.
- **reformulée** : la « capacité conditionnelle » 0,415 contre 0,771 pour H5 sur les paires de gain
  n'est pas une lecture H2. Elle obtient gratuitement la référence identité q = y_ref, qui laisse
  déjà 0,644. La lecture informative reste la part de différence, 0,653 pour le gain.
- **supposée, affaiblie** : la tonalité agirait en partie avant la saturation. Le profil par quintile
  est en U (0,292 / 0,244 / 0,229 / 0,261 / 0,323), donc l'excès au quintile le plus fort ne
  départage pas.
- **établie, VS100R** : les sondes depuis x, ajustées sur V, laissent 0,86 à 1,32 de l'erreur de
  chaque modèle. La sonde sur y_ref, avec 16 coefficients anticipatifs (non causale), expose leurs
  décalages temporels :
  - **bras C** : en avance d'environ 1,0 à 1,2 échantillon à 48 kHz (médiane −1,108 sur V), environ
    1,0 sous 2 kHz ;
  - **F3** :
    - s42 : retard de +2,03 (V) / +2,00 (B) ;
    - s44 : retard de +3,64 (V) / +3,61 (B), deux retards presque purs ;
    - s43_r2 : −0,93 en bande pleine, mais −0,75 / −0,64 sous 2 kHz, donc en partie une
      différence de phase ;
  - **L s43** : retard de +0,7 à +2,0 ; **GREY s44** : retard d'environ +1,2.
- **conséquence établie pour la boîte grise** : la compensation qui aligne F3 s44 vaut environ
  −3,7 échantillons, appliquée au modèle. Elle sort de la grille de la co-lecture figée (−2 à +3,
  `coreading_delay.py`), du côté −2. Sur 4 prises (V), l'ESR médiane de F3 s44 (rapport de
  puissance) vaut :
  - 0,228 brute ;
  - 0,106 au bord de la grille ;
  - 0,033 alignée.

  Pour F3 s42, la compensation commune vaut −1,8, dans la grille. Selon la prise, son retard (1,57
  à 2,45) peut toutefois dépasser le bord. La règle figée ne bouge pas ; ce point sera signalé dans
  le bilan de `demo/greybox_deep/predictions.md`.
- **correctif mineur** : `lead1_output_referenced.py:283-289` range la paire inversée 23→20 parmi les
  paires avant exclues (n = 6 au lieu de 5). La ligne « tonalité, porte 4 exclue » vaut 0,352 / 0,348
  / 0,335 / 0,336 sur les seules paires avant. Aucun nombre principal ne change.

### Piste 4, plafond mesuré sur multisinus (verdict : partielle, resserrée)

- **établie** : le multisinus ne donne pas de plafond pour la DI. À RMS d'entrée égal, la FIR
  conditionnée au niveau, sur la DI, laisse en moyenne 0,226 / 0,183 / 0,109 de **puissance** de plus
  que la part non-BLA D des multisinus (−30 / −24 / −18 dBFS, gains d'au moins 2,5). Il y a des
  exceptions au gain 10 (prise 19 : −0,128 à −18 dBFS).
- **établie** : le classement des prises s'inverse. Spearman −0,622 à −18 dBFS (p = 1,4e-4) et
  −0,514 au global, robuste au jackknife et au bootstrap. Le global pondère les courbes de
  multisinus par la puissance des sorties de B (lecture descriptive) ; pondéré par V, il vaut
  −0,527. À −30 et −24 dBFS, l'effet est marginal :
  −0,368 et −0,330, intervalles bootstrap qui touchent 0.
- **établie** : D vaut 0,40–0,48 sur les quatre prises de l'échelle H2, et de 0,26 à 0,58 sur les 32
  prises à G ≥ 2,5.
- **retirée** : « la FIR ordonnancée non causale bat sa version causale (0,612 contre 0,732),
  signe d'une dépendance à l'enveloppe à venir ». La version dite causale utilisait le niveau de la
  fenêtre précédente, périmé de 50 à 100 ms. Un niveau causal glissant fait aussi bien (0,618 sur
  50 ms, 0,598 sur 10 ms, en puissance, au gain 5).
- **abaissée à supposée** : la règle mesurée « les branches non linéaires de H5 et VL3 captent de
  façon reproductible 7 à 13 % de ce que la FIR laisse ». Le signe est reproductible (V1, V2 et B),
  mais :
  - la valeur dépend de la section (4 à 13,5 %, en puissance) ;
  - la marge (n = 2) n'est pas étalonnée ;
  - une FIR causale ordonnancée par le niveau, linéaire en y, fait mieux que les deux sondes.
- **supposée** : l'inversion du classement viendrait de la pondération spectrale. Le multisinus met
  75 % de sa puissance au-dessus de 5 kHz, la DI 94 à 99 % sous 1 kHz. Au gain 0, l'essentiel de
  l'écart vient plutôt de la longueur de la FIR : avec 1 024 coefficients, le résidu à −30 dBFS passe
  de 0,083 à 0,024 sur B.

### Piste 5, état et rétroaction (verdict : réfutée pour un état lent ; portée resserrée)

- **établie** : un état lent existe, mais il ne porte qu'une petite part de la sortie. Différence
  entre la période 1 et la période 3 d'un multisinus après 0,5 s de silence, pire cas sur les gains
  de 2,5 à 10 :

  | Fenêtre | Écart (part en amplitude) |
  |---|---|
  | 10–30 ms | −14,4 dB (0,19) |
  | 30–100 ms | −27,8 dB (0,041) |
  | 100–300 ms | −40,5 dB (0,0094) |
  | Plancher, période 2 contre 3 | −42,0 dB |

  L'appareil synthétique S1, avec un affaissement de 0,5 s, reste à −19,0 dB sur 100–300 ms.
- **établie** : sur ces transitions (silence puis multisinus), la part dépendante de l'histoire vaut
  au plus 0,19 au-delà de 10 ms. Pour passer de 0,80 à 0,30, il faudrait une composante d'au moins
  0,45–0,50 de la sortie, ou 0,68–0,74 si elle est orthogonale au reste.

  **supposée, conditionnelle** : l'exclusion d'un état lent comme cause du refus sur la DI. Elle
  suppose que ces transitions représentent la DI. Or 25,5 % des fenêtres de B dépassent le niveau
  maximal testé, et l'effet d'un réajustement de A vers B n'est pas mesuré.

  La plage 5,3–10 ms n'est pas couverte. Juste après la mémoire des sondes, la différence à froid
  vaut 0,39 à 0,58, mais un Hammerstein plus long ne la récupère pas (au plus 0,007). **supposée** :
  c'est de la mémoire avant la non-linéarité, plutôt qu'un état.
- **établie** : la règle LPV, déclarée préinscrite (antériorité non vérifiable, les fichiers n'étant
  pas versionnés), rend « indécidable » : baisses de 0,036 à 0,043 pour un seuil de 0,04. Le tau
  choisi sur V est le plus court de la grille, 10 ms. L'enveloppe anticausale fait aussi bien, et le
  placebo ne fait jamais mieux. Aux taus de 100 ms et plus, les baisses causales valent 0,003 à
  0,021. **supposée** : le gain viendrait du niveau court terme comme caractéristique, plutôt que de
  l'histoire ; l'égalité causale/anticausale n'exclut pas un effet de l'histoire.
- **établie** : deux clics identiques à 0,5 s d'écart diffèrent de −20,4 dB en médiane, surtout dans
  les 5 premières millisecondes. **supposées** :
  - « relaxé en 1 s » (histoires d'excitation différentes) ;
  - « au plus 0,1 sur la capacité » (le clic n'est pas une borne sur la DI).
- **établie** : la sortie remonte dans le silence après une entrée forte à présence 10 et **gain 5**.
  Les six prises concernées sont toutes au gain 5 ; c'est le seul gain d'au moins 5 rendu à
  présence 10. **supposée** : une boucle instable. Deux réserves : la remontée ne se maintient pas
  sans excitation, et elle diffère de la réponse modifiée aux clics. Celle-ci concorde avec
  `nolly_marker_drift.json` (−16,2 à −7,5 dB), sans démontrer une oscillation additive commune.
- **correctif d'étiquette** : les planchers de contrôle de x et S0 viennent d'un artefact de bord
  (chaque section est décimée seule), concentré dans la dernière milliseconde.

### Piste 6, caractéristiques apprises du bras C (verdict : partielle)

Il s'agit d'une exception exploratoire à « aucun réseau appris dans une métrique ». Le dispositif :
- caractéristiques figées des réseaux VS100R `vs100r48fast_film_s42-44` ;
- lecture linéaire ajustée sur A' (les 30 dernières secondes de `di_train`) ;
- ridge choisie sur A' seul.

- **établie** : B (`di_test`) de chaque prise utilisée n'a jamais servi à entraîner ces réseaux
  (manifeste du jeu, positions dans le fichier). En revanche, A' fait partie de leurs données
  d'entraînement pour les prises des réglages entraînés : leurs lignes sont des **caractéristiques
  entraînées sur la cible**.
- **établie** : sur la VS100R, la lecture `fir_pen` (FIR 256 sur l'avant-dernière couche) aux
  boutons du centre, aveugle au réglage, donne en moyenne sur 3 graines :
  - prises 1, 39 et 37 : 0,044 / 0,035 / 0,031 ;
  - prises 26, 16, 9 et 30 : 0,295 à 0,340 ;
  - sondes figées sur les mêmes prises : 0,468 à 0,774.

  Aux boutons de chaque prise, les moyennes de `fir_pen` restent sous 0,167 ; par graine, elles
  montent jusqu'à 0,239. Avec d'autres lectures, c'est plus haut : `fir_out` atteint 0,309 en moyenne
  sur la prise 16.
- **établie, portée resserrée** : trois initialisations aléatoires non entraînées donnent 0,70 à 0,77
  sur la prise 1, contre 0,04 entraînées. L'apprentissage compte donc. En revanche, un réservoir
  réglé (rayon spectral, échelle d'entrée) n'a pas été testé : « pas de la dynamique de
  l'architecture » n'est pas établi.
- **retirée comme preuve** : « la prise 16, couple jamais entraîné, se comporte comme une prise
  entraînée ». Sa sortie sur B ne diffère de la prise entraînée 26 que de 0,100 en RMS relatif, moins
  que la répétition 1→39 (0,138). Elle ne prouve donc pas une généralisation hors des réglages
  appris.
- **établie** : hors de son domaine, sur Nolly, la lecture moyenne aux boutons du centre (fir_pen)
  donne :

  | Gain | 0 | 2,5 | 5 | 7,5 | 10 |
  |---|---|---|---|---|---|
  | fir_pen | 0,488 (0,369 à 0,575) | 0,666 | 0,702 | 0,719 | 0,765 (0,717 à 0,821) |
  | H5 | 0,476 | 0,794 | 0,803 | 0,800 | 0,747 |

  Elle bat H5 aux gains 2,5 à 7,5, fait à peu près jeu égal aux gains 0 et 10, et reste au-dessus de
  0,3 partout (au plus près au gain 0). Sur les 19 différences Nolly d'un seul bouton, la FIR sur
  y_ref la bat toujours.
- **établie** : les lectures sont linéaires en y (3,4e-11 à ridge 1e-8, 9,5e-13 à 1e-4 ; l'écart vient
  du conditionnement). Le 2,616 de static_pen à Nolly G2,5 est une instabilité de la lecture peu
  régularisée : 90 % de l'énergie résiduelle tient dans 1 % des échantillons. Ce n'est ni un effet du
  bord de la grille ni une extrapolation d'amplitude.
- **limites** :
  - aucune décomposition par branche ni par ordre ;
  - circularité : une métrique fondée sur les caractéristiques du bras C favoriserait les modèles
    semblables au bras C.

## 5. Ce qui ressort

**Établi** sur ces rendus :
1. Sur les prises de l'échelle H2 à G ≥ 2,5, le refus tient pour les sondes testées dont les
   caractéristiques n'ont pas été apprises sur la cible évaluée. Les meilleures lectures :
   - banque de drives WH3 : 0,350 / 0,329 / 0,489 (G2,5 / G7,5 / G10) ;
   - WH3 figée transférée au gain 5 : 0,346 à 0,637 selon la source des caractéristiques (le minimum
     0,346 est une sélection sur B parmi quatre sources) ;
   - bases saturantes hp300 : 0,545 à 0,654.
2. Au gain 0, le seuil est franchi par des bases saturantes après un passe-haut fixe (0,236 à 0,280).
3. Après un même préfiltre, une base saturante bat une base polynomiale d'environ 0,16 au gain 5.
   Les cascades apprises descendent plus bas (1 à 3 étages : 0,39 → 0,27 sur la prise
   d'apprentissage).
4. Sur des transitions contrôlées, aucune part dépendante de l'histoire au-delà d'environ 10 ms
   n'atteint ce qu'il faudrait pour expliquer le refus. L'extension à la DI reste supposée.
5. Un plafond mesuré sur multisinus n'est pas un plafond pour la DI.
6. Une sonde rapportée à la sortie couvre bien les différences de tonalité (0,267) et de baffle
   (0,046), mal celles de gain (0,653). Elle révèle les décalages temporels des modèles VS100R,
   dont un, F3 s44, sort de la grille de la co-lecture figée.
7. Des caractéristiques apprises sur un appareil représentent très bien cet appareil, mais se
   transfèrent mal à un autre. Sur Nolly, celles de la VS100R ne battent H5 que d'environ 0,1 aux
   gains 2,5 à 7,5.

**Supposé**, à départager :
- la forme compterait plus que la mémoire et le degré : les contrastes mesurés n'isolent pas la
  forme, la mémoire, le degré et le budget de paramètres ;
- le gain du plugin agirait comme un drive entre étages ;
- la tonalité agirait en partie avant la saturation ;
- l'inversion du classement multisinus contre DI viendrait du spectre ;
- la plage 5,3–10 ms serait de la mémoire avant la non-linéarité.

## 6. Non expliqué et à tester

- **Différences de gain** (0,511 à 0,974 sur les transferts admissibles de la piste 2 ; 0,653 pour
  la sonde rapportée à la sortie) : le point dur. Pistes
  issues de la discussion avec Codex, non mesurées, toutes sur la cascade en cache :
  - ajouter ses dérivées par rapport à ses paramètres internes (gains entre étages, seuils), ce qui
    garde la sonde linéaire en y ;
  - comparer une dérivée interne à une dérivée de drive externe ;
  - comparer une dérivée à une différence finie.
- **Histoire ancienne** : l'ajouter après les caractéristiques rapides de la cascade, avec un
  contrôle anticausal. Pari de Codex : moins de 0,01.
- **Plage 5,3–10 ms** : non couverte par les mesures sans modèle.
- **Contraste « filtre entre étages contre sans filtre »** à budget égal : non fait. Profondeur,
  mémoire et paramètres ont changé ensemble.
- **Franchissement au gain 0** : la sélection de la coupure sur V (200 Hz, 0,236 sur B) est un
  contrôle exploratoire déjà réalisé. Relire B ne lèverait pas son exposition antérieure. Une
  confirmation exige un jeu resté fermé et un protocole figé avant sa lecture.
- Toute nouvelle métrique tirée de ces pistes demande un protocole neuf, figé sur le jeu de
  développement avant toute lecture.
