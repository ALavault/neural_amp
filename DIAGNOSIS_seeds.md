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

**Révision du 2026-09-16 (après coup, `global_factor.py`, `global_factor.txt`).** La graine
multiplie l'erreur de tous les segments par un même facteur. Rapport pire/meilleure
graine, moitié calme contre moitié forte : ×1,69 contre ×1,63 (S4, changements),
×1,35 contre ×1,44 (S4 publié), ×2,98 contre ×2,73 (ablation SSM). Seul SSM avec
changements s'écarte, avec ×4,00 contre ×2,03. Sur le log de l'ESR, l'effet run,
commun aux douze segments, domine l'interaction run × segment : F de 9 à 36.
Les 77–81 % observés sur la moitié calme relèvent de l'arithmétique. Un facteur
unique y placerait 73 à 83 % de l'écart, puisque ces segments ont déjà un ESR
2,6 à 4,8 fois plus grand. Le critère absolu n'est pas en cause non plus :
l'ESR pondéré par l'énergie, dominé par le matériel fort, varie autant (H4).
Contribution révisée : la graine fixe un facteur global d'erreur, d'écart-type
0,53 en log de l'ESR pour SSM-WaveNet (×1,7), 0,27 et 0,16 pour S4, de mécanisme
inconnu. La question suivante, graine ou non-déterminisme, a ses prédictions
écrites avant mesure (`hypotheses_nested.md`).

**Version initiale, réfutée (texte complet au commit `6025b26`).** *Prédicat* :
77 à 81 % de l'écart entre meilleure et pire graine sur les six segments calmes.
*Mécanisme* : un critère d'entraînement et de sélection absolu face à une
métrique relative laisserait le bas niveau à la graine. *Prédiction* : une
validation stratifiée par niveau ramènerait le CV de SSM sous 25 % et le rapport
calme/fort sous 3,0. Elle vise un mécanisme faux et n'est plus un critère.

## 6. Non expliqué, non vérifié

- La corrélation H2 reste sans mécanisme : six runs, statistique choisie après
  coup. Trois graines de plus par condition trancheraient.
- Part non attribuée : toute la dispersion, le facteur global étant sans mécanisme.
- **Variance à graine constante : une répétition, conclusion retirée.**
  L'entraînement n'est pas déterministe sur GPU (cuDNN benchmark,
  `use_deterministic_algorithms` à False, comme dans NablAFx). La répétition de la
  graine 42 (`ssmzoh_guard_seed42_repeat`) donne 0,0470 contre 0,0358, soit un
  décalage global de +0,23 en log (écart-type par run estimé à 0,16). C'est le
  niveau de S4 publié entre graines, et un tiers de celui de SSM, avec un seul
  degré de liberté. La conclusion « huit fois plus petit que l'écart entre graines,
  n'explique pas la dispersion » est retirée : elle comparait, en échelle
  linéaire, une différence isolée prise sur la meilleure graine à l'étendue de
  trois graines. Répétitions en cours (graines 43 et 44 pour SSM, 42 à 44 pour S4
  publié).
- Non séparés : initialisation, partition et ordre des lots changent ensemble
  avec la graine.

Correctif : voir skill controlled-fix.
