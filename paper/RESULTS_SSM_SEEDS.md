# Résultats pour la rédaction : SSM-WaveNet, sensibilité à la graine, pilote « effet papillon »

État au 2026-09-18. Chaque chiffre renvoie à un fichier du dépôt ; ne rien citer
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
- La graine multiplie l'erreur de tous les segments de test par un facteur largement
  commun : sur le log de l'ESR, l'effet run domine l'interaction run × segment (F de 9
  à 36). **Nuance mesurée depuis** (`diagnosis/butterfly/where_audible.md`) : le rapport
  d'ESR segment par segment entre deux graines varie d'un facteur 10 (0,83 à 8,82,
  écart-type du log 0,629) autour d'une médiane de 3,30. Le facteur global domine
  (1,19 en log contre 0,63 de résidu) mais c'est une approximation, pas une égalité :
  écrire « largement commun », jamais « identique ».
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

**Pilote A : les décisions pilotées par la validation amplifient une perturbation
minimale** (`diagnosis/butterfly/pilot_A.md`, prédictions commitées avant tout run à
`bb08337` ; verdict dans `diagnosis/butterfly/pilot_A_results.json`).

Parent SSM-WaveNet déterministe, graine 42, fourche à l'époque 100. Enfants : chaque
poids déplacé d'un pas float32, direction tirée par la graine 1000 + k. Bras « décide » :
décisions propres. Bras « rejoue » : learning rate de chaque époque et pas d'arrêt du
témoin, sans arrêt anticipé.

- **Verdict : soutenu.** s(décide) = 0,156 ≥ 0,10 ; s(rejoue) = 0,009 ≤ 0,078 ; rapport
  des moyennes des |Δ| = 13,2 ≥ 2. Les trois critères pré-enregistrés sont satisfaits.
- **Contrôle positif P0** (fourche 5, bras décide) : s = 0,207 pour un seuil de 0,10. La
  faible dispersion du bras « rejoue » n'est donc pas une insensibilité du dispositif.
- Contrôles passés : parent et jumeau identiques ; « rejoue » k = 0 identique bit à bit
  au témoin, et 923 pertes de validation identiques.
- Fourche 100, ESR : décide 0,0341 / 0,0512 / 0,0378 / 0,0382 / 0,0365 ; rejoue 0,0341 /
  0,0339 / 0,0336 / 0,0339 / 0,0334. Écart de sortie au témoin : décide 8,7·10⁻⁴ à
  7,2·10⁻³, rejoue 4,2·10⁻⁴ à 7,3·10⁻⁴, pour une erreur du témoin de 0,0245 sur la même
  mesure.
- **Formulation autorisée** : « une perturbation d'un pas float32 sur chaque poids,
  appliquée à l'époque 100, change l'ESR final de 0,156 en log lorsque le run prend ses
  propres décisions de validation, et de 0,009 lorsqu'il rejoue le calendrier du témoin ;
  5 runs par bras, une graine, un appareil, une architecture ».
- **Interdit** : présenter ce pilote comme valant pour l'entraînement en général, ou
  omettre que les seuils sont de qualité pilote.

**Mesures sur les enfants de la fourche 100** (lecture seule, CPU, sans nouveau run) :

- *Quand le sort d'un run se joue* (`diagnosis/butterfly/decided_when.md`). Sur cinq runs
  partageant le découpage, lecture paire par paire de l'époque à partir de laquelle le
  minimum courant de la perte de validation conserve le bon ordre : les quatre paires
  d'écart ≥ 0,29 en log sont tranchées entre les époques 98 et 191 sur environ 900 ; les
  écarts ≤ 0,11 le sont tard (269 à 652) ou jamais. **Piège à signaler** : l'ESR est
  quadratique en amplitude d'erreur là où la perte L1 + 0,1·MR-STFT est linéaire, donc un
  facteur 2 entre les deux dispersions en log est imposé par les définitions. Au-delà de
  ce facteur, le passage validation → test ajoute ×2,5 (décide) et ×1,9 (rejoue) ; les
  décisions, elles, dilatent l'écart de validation d'un facteur 16.
- *Connectivité linéaire des modes* (`diagnosis/butterfly/mode_connectivity.md`).
  L'enfant le plus divergent franchit une barrière réelle : ESR 0,0567 au milieu du
  chemin contre 0,0341 et 0,0510 aux extrémités, présente dans 12 segments sur 12. Mais
  la hauteur de barrière suit la distance L2 parcourue (ρ = +0,90, p = 0,002, n = 8) et
  non le bras, ce qui est l'hypothèse nulle d'un bassin unique mais courbe : sept enfants
  sur huit y sont compatibles, et seul le plus divergent y échappe (barrière 5,2 fois
  celle d'un enfant pourtant plus éloigné). **Ne pas écrire** que la connectivité sépare
  les deux bras.
- *Ce que l'écart n'est pas* (`diagnosis/butterfly/is_it_eq.md`). Le meilleur filtre
  linéaire de la sortie d'un enfant vers celle du témoin ne retire que 1 % de l'écart
  (ESR 0,0072 → 0,0071) et son module tient entre −0,04 et +0,02 dB par tiers d'octave
  de 99 Hz à 3,2 kHz. L'écart n'est donc ni un niveau, ni une égalisation, ni un retard
  constant : il dépend du programme.
- *Où il se loge* (`diagnosis/butterfly/when_audible.md`). Images de 20 ms classées par
  la pente de l'enveloppe : pendant les chutes d'enveloppe l'écart se tient 5,3 dB sous
  le signal sur le segment le plus divergent et 2,3 dB sur le suivant, contre 16,0 et
  25,3 dB pendant les tenues. Les attaques ne portent rien de particulier.
- *Ce qui ne le prédit pas* (`diagnosis/butterfly/where_audible.md`). Aucune corrélation
  de rang entre la part d'erreur prise par l'écart et le RMS (−0,12), le facteur de crête
  (+0,12), le centroïde spectral (+0,43) ou la proportion de passages calmes (+0,19),
  n = 12 segments.

## 2. Préliminaire : ne pas rédiger comme résultat

**Écoute** (`diagnosis/butterfly/listening.md`, page `demo/listening/butterfly.html`,
générateur `scripts/product_butterfly_listening.py`). Un auditeur, sans relevé d'essais
chiffré, sur des extraits choisis au point de divergence maximale : aucune différence
entendue entre le témoin et son enfant divergent, **ni entre le témoin et l'appareil
réel**, et aucune préférence en comparaison A/B non masquée. Le texte écrit d'avance
prévoit exactement ce cas : sans contrôle positif de la chaîne d'écoute, un nul sur
« réel contre témoin » signifie que l'épreuve manque de sensibilité et ne conclut rien
sur les enfants. Une échelle d'audibilité a été ajoutée à la page pour lever
l'ambiguïté ; elle n'a pas encore été parcourue. **Formulation autorisée si rien de
plus n'arrive** : « aucune évaluation perceptive n'étaye ces écarts d'ESR ».
**Interdit** : « la divergence est inaudible » sans le contrôle positif.

**Fourche 5, bras « rejoue »** (`diagnosis/butterfly/pilot_A_fork5_replay.md`, en cours).
Trois enfants sur cinq mesurés : Δ = +0,750 (k = 1), −0,129 (k = 2). Le bras ne répond
pas à la question qu'il posait, parce que la garde de polarité y reste libre : le parent
bascule aux époques 7, 8 et 9, donc une fourche placée à l'époque 5 laisse chaque enfant
redécouvrir ses propres bascules, et celles-ci sont elles-mêmes des décisions de
validation. Nuance : l'historique de bascules est identique entre les deux bras pour un
même k, les bascules tombant bien avant la première division du learning rate, donc la
comparaison entre bras reste interprétable ; c'est la lecture de s(rejoue) comme « ce qui
reste quand on retire les décisions » qui ne tient pas.

**Pilote D, fourche 25** (`diagnosis/butterfly/pilot_D_fork25.md`, pré-enregistré, runs
non commencés). Fourche placée après la dernière bascule du parent (époque 9) et avant sa
première division du learning rate (époque 113), pour poser à la phase précoce la
question que la fourche 100 a tranchée.

## 3. Retiré ou réfuté, encore présent dans le tag `icassp2027-draft`

- Paragraphe « Where the error sits » de `main.tex` : 77 à 81 % de l'écart entre graines
  sur les six segments calmes (`\SpreadQuietShare`), mécanisme « perte de validation
  absolue contre métrique relative », validation stratifiée comme correctif. Réfuté :
  remplacé par le facteur global. `\ValBlindEsr` et `\ValBlindShare` sont exacts mais ne
  soutiennent plus aucun mécanisme. La localisation sur les segments calmes a été
  re-testée en 2026-09-18 pour l'écart entre deux enfants d'une même graine, et reste
  réfutée : deux segments de même proportion de passages calmes (50 %) portent des parts
  d'écart de 0,93 et 0,10.
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

- Fourche 5 : « rejoue » k = 3 et k = 4, puis lecture du bras avec la réserve ci-dessus.
- Pilote D, fourche 25 : parent rejoué avec porte au bit près, puis dix enfants.
- Pilote C, huit graines sous un calendrier commun fixe, conditionné à B
  (`diagnosis/butterfly/pilot_C_common_schedule.md`).
- S4-TF-L-16 publié : répétitions à graine égale, puis verdict emboîté pour S4.
- Écoute : parcourir l'échelle d'audibilité, qui donnerait la marge de la divergence sous
  le seuil, dans l'unité de la divergence elle-même.
- Question ouverte, sans expérience prévue : la direction de la perturbation. C'est k = 1
  qui décroche aux deux fourches déjà mesurées, alors que sa direction est tirée avec la
  même graine 1000 + k sur deux parents différents. Une chance sur quatre d'être fortuit.

**Citations** : clés dans `paper/icassp2027/refs.bib`. Le modèle S4-TF-L-16 est défini
dans Comunità, Steinmetz et Reiss, *Frontiers in Signal Processing* 5, art. 1580395,
2025, doi:10.3389/frsip.2025.1580395 (vérifié le 2026-09-17). NablAFx : arXiv:2502.11668.
S4 : Gu et al., ICLR 2022. TFiLM : Birnbaum et al., NeurIPS 2019 ; Comunità et al.,
ICASSP 2023. Connectivité linéaire des modes : Frankle, Dziugaite, Roy et Carbin, ICML
2020.
