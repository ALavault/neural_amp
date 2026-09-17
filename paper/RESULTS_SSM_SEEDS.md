# Résultats pour la rédaction : SSM-WaveNet, sensibilité à la graine, pilote « effet papillon »

État au 2026-09-17. Chaque chiffre renvoie à un fichier du dépôt ; ne rien citer
d'autre sans le recalculer. Le brouillon ICASSP 2027 (`paper/icassp2027/`, tag
`icassp2027-draft`) est abandonné et contient des affirmations retirées (section 3).
Ce fichier ne concerne pas `paper/claims.md` ni `paper/outline.md`, qui portent sur
FSSR contre NAM A2.

**Conventions.** ESR de test du protocole NablAFx : moyenne des ESR de 12 segments de
5 s, dernier checkpoint. Comparaisons et dispersions sur le log de l'ESR, moyennes
géométriques, n indiqué partout, tous les runs rapportés, répétitions comprises.
Auteur : Antoine Lavault, LISTIC, Université Savoie Mont Blanc, ORCiD
0009-0001-9266-786X. Aucune venue n'est choisie.

## 1. Établi

**Protocole et pièges** (Big Muff S050_V100 de ToneTwist AFx, NablAFx commit `045db6e`,
`scripts/product_nablafx_bench.py`, méthode détaillée dans `paper/icassp2027/main.tex`,
section Setup) :
- AdamW, learning rate 0,01, lot de 16, L1 + 0,1·MR-STFT, division du learning rate après
  20 époques sans amélioration, arrêt après 50, 15 000 pas au plus. Partition 118/12
  tirée par la graine. Clipping à 1 comme dans l'article, alors que le YAML publié met 10.
- Le code publié place tous les paramètres dans un seul groupe AdamW (weight decay 0,01)
  et ignore l'exemption que demandent les couches DSSM. Changement 1 : exemption.
- Le terme MR-STFT ne voit pas le signe de la sortie. Un run SSM-WaveNet a convergé vers
  une sortie inversée : ESR 3,95 au pas 5 100, 0,052 une fois la sortie négée. Trois
  modèles publiés (TCN-2500-S-16, LSTM-32, GCN-TF-2500-L-16) ont un ESR supérieur à 1 et
  un L1 de 0,0293 à 0,0295, proche de 2E|y| = 0,0299 : compatible avec une inversion,
  non confirmé faute de checkpoints. Changement 2 : garde de polarité. Sur les 9 runs
  gardés du banc, 21 bascules, jamais après l'époque 20.

**Comparaison** (`demo/nablafx_bench/*.json` ; valeurs publiées :
`paper/icassp2027/data/published_bigmuff.json`, annexes de Comunità et al. 2025) :

| Modèle | Paramètres | n | ESR, graines 42 / 43 / 44 | Moyenne géom. |
|---|---:|---|---|---:|
| SSM-WaveNet, ZOH, deux changements | 11 329 | 3 graines × 2 | 0,0358 / 0,1277 / 0,0832 ; répétitions 0,0470 / 0,0823 / 0,0659 | 0,068 |
| S4-TF-L-16, deux changements | 70 193 | 3 | 0,0888 / 0,1354 / 0,1568 | 0,124 |
| S4-TF-L-16, entraînement publié, réexécuté | 70 193 | 3 | 0,1373 / 0,1569 / 0,1171 | 0,136 |
| SSM-WaveNet, première version (ablation) | 12 353 | 3 | 0,2363 / 0,0738 / 0,1656 | 0,142 |
| S4-TF-L-16 publié (un run, meilleure de 4 pondérations de perte) | 70 193 | 1 | 0,108 | — |

- **Formulation autorisée** : « SSM-WaveNet a un ESR plus bas que S4-TF-L-16 à chacune
  des trois graines, répétitions comprises, avec six fois moins de paramètres (rapport
  des moyennes géométriques 1,8 à conditions d'entraînement égales). Avec trois graines
  par modèle, aucun test ne peut descendre sous 5 % (Mann-Whitney bilatéral : p minimal
  0,10), et l'écart entre graines est du même ordre que l'écart entre architectures. »
- **Interdit** : « SSM-WaveNet bat / surpasse S4-TF-L-16 », « état de l'art ».
- Les graines ne forment pas des paires : la partition est tirée après
  l'initialisation, qui consomme un nombre de tirages différent selon le modèle.
- Un run unique ne se compare pas à une valeur publiée à mieux que ±20 % : l'erreur
  type de la moyenne sur 12 segments vaut 19 à 24 % de sa valeur
  (`diagnosis/seeds/spread.txt`).

**Sensibilité à la graine** (`DIAGNOSIS_seeds.md`, `diagnosis/seeds/global_factor.txt`) :
- La graine multiplie l'erreur de tous les segments de test par un même facteur : sur le
  log de l'ESR, l'effet run domine l'interaction run × segment (F de 9 à 36).
- Écart-type de l'effet run sur les premiers runs des 3 graines : 0,53 pour SSM-WaveNet,
  0,27 pour S4-TF-L-16 avec les deux changements, 0,16 pour S4-TF-L-16 publié.
- Analyse emboîtée SSM-WaveNet, 3 graines × 2 runs, prédictions écrites avant
  (`diagnosis/seeds/hypotheses_nested.md`) : écart-type intra-graine 0,18, composante
  graine 0,37, F(2,3) = 8,8, p ≈ 0,06. La graine porte environ 80 % de la variance
  estimée. Le non-déterminisme GPU seul a déplacé un run d'un facteur 1,55. Aucune des
  deux hypothèses pré-enregistrées (non-déterminisme dominant, graine dominante) ne
  tient telle qu'écrite.
- Mécanismes réfutés : localisation sur les segments calmes (effet arithmétique d'un
  facteur global), critère de validation absolu, nombre de pôles lents, pondération
  par l'énergie. Corrélation sans mécanisme, choisie après coup : RMS minimal de la
  partition de validation et ESR de test (6 runs sur 6).

**Déterminisme** (`diagnosis/butterfly/determinism.json`, `diagnosis/butterfly/pilot_A.md`) :
- Algorithmes CUDA déterministes, et padding par réflexion reconstruit par découpage :
  le backward CUDA de `reflection_pad1d`, utilisé par `torch.stft` dans la MR-STFT, n'a
  pas de version déterministe.
- Lot de 2, SSM-WaveNet et S4-TF-L-16 : deux runs de 200 pas identiques bit à bit
  (poids et moments d'Adam), deux reprises identiques, deux runs non déterministes
  différents.
- Lot de 16, SSM-WaveNet : parent et jumeau identiques bit à bit aux époques 5 et 100,
  pour un coût de 5 à 10 % en vitesse.

## 2. Préliminaire : ne pas rédiger comme résultat

Pilote A (`diagnosis/butterfly/pilot_A.md`, prédictions commitées avant les runs,
`bb08337`). Parent SSM-WaveNet déterministe, graine 42, fourches aux époques 5 et 100.
Enfants : chaque poids déplacé d'un pas float32 (k = 1 à 4). Bras « décide » : décisions
propres. Bras « rejoue » : learning rate de chaque époque et pas d'arrêt du témoin k = 0,
sans arrêt anticipé.

- Contrôles passés : jumeaux identiques ; « rejoue » k = 0 identique bit à bit au
  témoin, et 923 pertes de validation identiques.
- Fourche 100, écart de log ESR au témoin (ESR 0,0341) : « décide » +0,405, +0,102,
  +0,112 (k = 1, 2, 3) ; « rejoue » −0,007, −0,015 (k = 1, 2). Écart de sortie au témoin
  (ESR entre sorties, sur l'ensemble du test) : « rejoue » 5,3·10⁻⁴ et 4,2·10⁻⁴,
  « décide » 7,2·10⁻³, 9,8·10⁻⁴ et 1,6·10⁻³, pour une erreur du témoin de 0,0245 sur la
  même mesure.
- Courbes de validation : les enfants « rejoue » rejoignent le témoin vers l'époque 300 ;
  pour « décide » k = 1 et 2, le déficit existe à époque égale avant l'arrêt anticipé,
  avec une première division du learning rate aux époques 35 et 98 contre 113.
- Le verdict pré-enregistré attend « rejoue » k = 3, les deux bras à k = 4 et le
  contrôle positif P0 (fourche 5).

## 3. Retiré ou réfuté, encore présent dans le tag `icassp2027-draft`

- Paragraphe « Where the error sits » de `main.tex` : 77 à 81 % de l'écart entre graines
  sur les six segments calmes (`\SpreadQuietShare`), mécanisme « perte de validation
  absolue contre métrique relative », validation stratifiée comme correctif. Réfuté :
  remplacé par le facteur global. `\ValBlindEsr` et `\ValBlindShare` sont exacts mais ne
  soutiennent plus aucun mécanisme.
- Répétition de la graine 42 présentée comme petite (`\RepeatDelta` = 0,011, en
  linéaire) et « non-déterminisme huit fois plus petit que l'écart entre graines » :
  retirés. En log, +0,27 ; en emboîté, 0,18 contre 0,37.
- `\GuardFlips` (10) et `\GuardLastFlipEpoch` (11) : 6 runs, périmés ; désormais 21
  bascules sur 9 runs, jamais après l'époque 20.
- `\SSMesrMeanStd` (0,082 ± 0,046, n = 3, moyenne arithmétique) : périmé ; 6 runs,
  moyenne géométrique.
- `results.tex` est généré par `scripts/product_paper_results.py` : ne pas l'éditer à la
  main, et ses macros datent d'avant les répétitions.

## 4. En attente

- Fourche 100 : « rejoue » k = 3, puis k = 4 dans les deux bras. Ensuite le verdict A.
- Fourche 5 : « décide » k = 0 à 4 (P0) ; si P0 tient, « rejoue » k = 0 à 4
  (`diagnosis/butterfly/pilot_A_fork5_replay.md`). Ensuite le verdict B.
- Pilote C, huit graines sous un calendrier commun fixe, seulement si B tient
  (`diagnosis/butterfly/pilot_C_common_schedule.md`).
- S4-TF-L-16 publié : répétitions à graine égale, puis verdict emboîté pour S4.

**Citations** : clés dans `paper/icassp2027/refs.bib`. Le modèle S4-TF-L-16 est défini
dans Comunità, Steinmetz et Reiss, *Frontiers in Signal Processing* 5, art. 1580395,
2025, doi:10.3389/frsip.2025.1580395 (vérifié le 2026-09-17). NablAFx : arXiv:2502.11668.
S4 : Gu et al., ICLR 2022. TFiLM : Birnbaum et al., NeurIPS 2019 ; Comunità et al.,
ICASSP 2023.
