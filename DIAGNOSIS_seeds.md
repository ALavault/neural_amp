# DIAGNOSIS_seeds — dispersion inter-graines sur ToneTwist Big Muff

Scripts et tables : `diagnosis/seeds/` (hypothèses : `hypotheses.md`,
`hypotheses_nested.md` ; mesures : `measure.py`, `validation_loss.py`,
`global_factor.py`, avec leurs sorties). Lecture seule sur les runs de
`demo/nablafx_bench/`.

## 1. Écart

Métrique : ESR de test du protocole NablAFx, moyenne des ESR de douze segments de
5 s, dernier checkpoint. Trois graines (42, 43, 44) par condition, douze runs.

| Condition | ESR par graine | Moyenne | Écart-type | CV |
|---|---|---|---|---|
| SSM-WaveNet 11,3k, ZOH + exemption + garde | 0,0358 / 0,1277 / 0,0832 | 0,0822 | 0,0460 | 56 % |
| S4-TF-L-16 70,2k, mêmes changements | 0,0888 / 0,1354 / 0,1568 | 0,1270 | 0,0348 | 27 % |
| S4-TF-L-16 70,2k, entraînement publié | 0,1373 / 0,1569 / 0,1171 | 0,1371 | 0,0199 | 15 % |
| SSM-WaveNet 12,4k, `b` appris, entraînement publié | 0,2363 / 0,0738 / 0,1656 | 0,1586 | 0,0815 | 51 % |

Référence : l'étude publiée ne rapporte qu'un run par modèle, donc aucune
dispersion de référence. L'écart SSM/S4 avec changements (0,0822 contre 0,1270)
est du même ordre que l'écart-type intra-condition : SSM est plus bas sur les
trois graines appariées, mais un run unique ne le résout pas. L'objet du
diagnostic est la dispersion elle-même.

## 2. Comparabilité

- **Jeu de test** : les douze mêmes segments de 5 s pour les douze runs
  (`test_segments.json`, champ `segment_rms` commun).
- **Données, métrique, protocole** : même archive Zenodo (sha256 dans
  `paper/icassp2027/README.md`), même rééchantillonnage, `auraloss` ESR par
  segment puis moyenne, dernier checkpoint, même code
  (`scripts/product_nablafx_bench.py`) avec le commit enregistré par run.
- **Budget réel** : plafond de 15k pas jamais atteint ; arrêt entre 4 137 et
  8 764 pas ; 11 329 / 12 353 / 70 193 paramètres.
- **Taille du jeu d'évaluation** : la moyenne sur douze segments a une erreur
  type de 19 à 24 % de sa valeur (`spread.txt`). Les runs partageant les mêmes
  segments, cette erreur n'explique pas les différences entre runs, mais elle
  interdit de comparer un run unique à une valeur publiée à mieux que ±20 %.

Aucun défaut de comparabilité : le diagnostic porte sur l'entraînement.

## 3. Hypothèses

Écrites avant toute mesure (`diagnosis/seeds/hypotheses.md`, commit `4f7809e`).

- **H1, point d'arrêt (optimisation)** : l'arrêt anticipé et le palier de
  learning rate s'appuient sur douze segments ; les runs s'arrêtent à des
  maturités très différentes. *Prédiction* : ρ(pas, ESR) ≤ −0,8 par condition.
- **H2, partition train/val (données)** : la graine choisit les douze segments
  retirés de l'entraînement, donc le signal d'arrêt et la matière retirée.
  *Prédiction* : l'ESR de la moitié calme suit une statistique de niveau de la
  partition.
- **H3, tirage des pôles (capacité)** : 512 pôles contre 4 096 pour S4 ; un
  tirage pauvre en modes lents n'est pas rattrapable. *Prédiction* : la meilleure
  graine a nettement plus de pôles à τ > 10 ms que la pire.
- **H4, définition de la métrique (mesure)** : la moyenne par segment est
  dominée par quelques segments. *Prédiction* : un ESR pondéré par l'énergie
  réduit le CV d'au moins 30 % en relatif.

## 4. Mesures et verdict

- **H1** — corrélation de rang pas/ESR : −1,00 dans les deux conditions « avec
  changements », **+1,00** dans les deux conditions « entraînement publié »,
  +0,09 sur les douze runs. Le signe s'inverse d'une condition à l'autre.
  **Indécidable** : trois points par condition ne séparent pas ±1 du hasard.
- **H2** — reconstruction des partitions : le RMS minimal du sous-ensemble de
  validation ordonne exactement l'ESR de test dans les deux conditions gardées,
  six runs sur six (SSM : 0,0308 → 0,036 ; 0,0201 → 0,083 ; 0,0002 → 0,128 ;
  S4 : 0,0284 → 0,089 ; 0,0201 → 0,135 ; 0,0127 → 0,157). Mais le mécanisme
  proposé est faux : le segment quasi silencieux de la graine 43, d'ESR 7,04,
  ne pèse que 6,3 % de la perte de validation. **Corrélation soutenue, mécanisme
  réfuté** ; statistique retenue après coup parmi cinq.
- **H3** — constantes de temps apprises : les runs avec exemption de weight decay
  ont 99 à 118 pôles au-delà de 10 ms contre 4 à 21 pour l'ablation, ce qui
  confirme l'effet du weight decay *entre* conditions ; à l'intérieur d'une
  condition, le nombre de pôles longs n'ordonne pas l'ESR (graine 44 : 118 pôles
  longs et ESR moyen ; graine 42 : 99 et meilleur ESR). **Réfutée** pour la
  dispersion intra-condition.
- **H4** — ESR pondéré par l'énergie : CV 50 % contre 56 % (SSM), 29 % contre
  27 % (S4), soit 11 % en relatif au lieu des 30 % prédits. **Réfutée.**

## 5. Contribution retenue

**Révision du 2026-09-21 : la contribution est identifiée, et elle n'est pas dans
l'optimisation.** La graine tire trois choses ensemble — initialisation, ordre des lots,
partition train/validation. En rendant la chaîne de données commune à six graines
(`diagnosis/seeds/pilot_E_split.md`), l'écart-type de l'effet run tombe de **0,375 à 0,191**,
soit le plancher du non-déterminisme GPU seul, 0,18. Une fois la partition et l'ordre des lots
fixés, l'initialisation ne contribue plus rien de mesurable.

*Prédicat* : la dispersion inter-graines de l'ESR de test sur ce dispositif est portée par le
tirage des données, non par la trajectoire d'optimisation. *Mécanisme* : le jeu de validation de
douze segments fixe à la fois ce que le modèle ne voit pas et ce sur quoi toutes ses décisions
sont prises ; son écart-type d'effet segment vaut 1,0 en log, un facteur 2,7 entre segments
faciles et difficiles. *Prédiction chiffrée qu'un correctif devra satisfaire* : un protocole qui
fixe la partition pour toutes les graines doit ramener l'écart-type de l'effet run sous **0,25**,
et il ne doit pas améliorer la moyenne — la partition figée ici donne 0,0831 contre 0,068 pour la
moyenne des partitions tirées, parce que fixer n'est pas choisir.

**Voie écartée, mesurée : l'ordonnanceur.** Le détecteur de plateau se déclenche bien sur du
bruit — seuil 44 à 88 fois sous la fluctuation qu'il observe, marge de déclenchement souvent
inférieure à celle-ci (`plateau_margin.md`). Mais le correctif est faux : remonter le seuil ne
réduit pas la dispersion (0,344 pour un critère à 0,25) et dégrade la qualité d'un facteur 2,4
(`pilot_F_threshold.md`) ; et aucun seuil fixe ne rend le déclenchement reproductible, les seuls
minima étant dégénérés (`plateau_simulation.md`).

**Versions précédentes de cette section**, dont celle du 2026-09-16 qui a établi le facteur
global et celle, réfutée, qui plaçait l'écart sur les segments calmes :
`diagnosis/seeds/revisions.md`.

## 6. Non expliqué, non vérifié

- **Part non attribuée** : le facteur global reste sans mécanisme. On sait maintenant *d'où* il
  vient — le tirage des données — mais pas *comment* une partition produit un facteur commun à
  tous les segments de test.
- **Partition et ordre des lots** restent inséparés : tirés du même générateur au même instant.
  Les distinguer demanderait de toucher au module de données de nablafx.
- **H2 sans mécanisme** : la corrélation entre RMS minimal de la partition de validation et ESR
  de test tient 6 fois sur 6, mais la statistique a été choisie après coup.
- **S4-TF-L-16** (`s4_nested.md`) : aucun effet de graine détectable, F(2,3) = 0,29 ; l'écart
  intra-graine, 0,173 sur deux paires vérifiées comparables, égale ou dépasse celui entre graines
  — l'inverse de SSM-WaveNet. La troisième paire est inutilisable, son run initial datant d'un
  état antérieur du script.
- **Toute affirmation d'ordre est fragile** (`variance_sources.md`) : un autre tirage de douze
  segments de test réordonnerait les six runs du banc dans 41 % des cas.
- **Hors de cause, vérifié** : la règle d'arrêt (`best` et `last` donnent le même ESR à 0,0005
  près) et le réglage de l'ordonnanceur (section 5). Le non-déterminisme GPU seul vaut 0,18 en
  log et n'est pas supprimable sans le mode déterministe, qui coûte 5 à 10 %.
- Le détail des conclusions retirées en route est dans `diagnosis/seeds/revisions.md`.

Correctif : voir skill controlled-fix.
