# G2 : des réglages qui déplacent les pôles (Marshall JVM410H OD1)

Voie produit (D-PRODUCT-001), aucune revendication scientifique. **Pré-enregistré le 2026-09-24,
avant tout run G2**, préflight compris. Les seuils ci-dessous ne sont plus modifiés après les
chiffres. Code : `cond`, `cond_head` et `cond_hidden` de `src/fssr_nam/models/ssm_wavenet.py`, et
tests dans `tests/numerical/test_ssm_poles.py`. Verdict calculé par `demo/param/poles_verdict.py`,
écrit avant les runs.

## Question

Le FiLM de SSM-WaveNet agit sur les couches 1×1 : il est sans mémoire, donc un bouton ne peut pas
y déplacer une fréquence de coupure. Or un tone stack passif est un filtre dont les boutons
déplacent les pôles. **Conditionner les pôles des SSM, et non les activations, généralise-t-il
mieux aux réglages jamais vus ?**

Seconde question, de structure : les réglages d'entraînement font varier **un bouton à la fois**
autour de 5 (plus tout-à-0 et tout-à-10) ; les neuf réglages de test sont des **combinaisons**.
Une tête additive (somme de fonctions d'un seul bouton) est exactement ce que ces données
identifient ; une tête jointe doit extrapoler des interactions qu'elle n'a presque pas vues.

## Bras (plan 2×2) et parité de paramètres

Tous les bras : SSM-WaveNet, 8 blocs, 16 canaux, 4 modes, ZOH, exemption de weight decay,
graine 42, `--split-seed 925`, mode déterministe, L1 0,5 + MR-STFT 0,5, 15 000 pas au plus, arrêt
anticipé actif (patience 50 époques), comme pour A, chez qui il ne s'est pas déclenché. Les
drapeaux sont ceux de `param_ssm_wavenet_marshall`, seul le conditionnement change.

| bras | `--cond` | `--cond-head` | `--cond-hidden` | paramètres | écart au FiLM joint |
|---|---|---|---|---|---|
| **A** FiLM joint (existe : `param_ssm_wavenet_marshall`) | film | joint | 32 | 28 385 | 0 |
| **B** FiLM additif | film | additive | 8 8 8 8 | 28 289 | −0,34 % |
| **C** pôles joint | poles | joint | 25 | 28 094 | **−1,03 %** |
| **D** pôles additif | poles | additive | 7 7 6 6 | 28 661 | +0,97 % |

- Le bras A est réutilisé sans relance. La porte de reproductibilité du 2026-09-24 a montré son
  chemin identique au bit au code actuel (`diagnosis/seeds/repro_gate.md`).
- Écart déclaré : le plan visait ±1 %. Pour C, la largeur entière la plus proche donne −1,03 %
  (7 paramètres de moins que la borne) et 26 donnerait +1,25 %.
- Tête « pôles » : s·tanh(u), s = 2,3. Elle produit, par bloc, δlog_dt par canal et δlog_A_real
  par mode. A_imag n'est pas conditionné. Couche de sortie initialisée à zéro : au départ, le
  modèle est le modèle non conditionné.
- Noyaux propres à chaque exemple, sous `torch.utils.checkpoint`.
- **Même tronc dans les quatre bras.** À graine égale, les bras B, C et D consomment d'abord
  exactement les tirages de la tête de A, puis construisent leur tête sur un générateur dérivé.
  Tous les paramètres hors tête sont donc identiques à ceux de A, et les têtes étant nulles au
  départ, les quatre bras sont le même modèle au pas 0. Sans cela, chaque forme de tête
  consommait un nombre différent de tirages, et chaque bras partait d'un tronc différent : un
  facteur confondant que les partitions de G2.2 n'auraient pas moyenné (relevé à la relecture,
  avant tout run ; test `test_every_arm_starts_from_the_same_trunk`).

## Mesures

- **Réglages non vus** : les neuf réglages de test, ESR et MR-STFT par réglage (dernier
  checkpoint, lots de 8 comme le banc).
  - **Étiquette corrigée** : `B030_M070_T100_G050` est évalué avec treble = 0,1, valeur primaire
    (`demo/param/label_check.md`), et avec 1,0 en secondaire. Le bras A est réévalué de la même
    façon, par le même script.
- **Réglages vus** : `val_last_by_setting` des enregistrements, sur 28 réglages, mesure
  optimiste.
- **Effet apparié** par réglage : d = ln(ESR_A / ESR_bras) ; positif = le bras fait mieux que le
  FiLM joint.
- **Bruit inconnu** : on ne connaît pas l'écart-type d'une run Marshall seule d'une graine ou
  d'une partition à l'autre. Sur Big Muff, il valait 0,27 à 0,53 en log entre graines non
  appariées, et 0,066 en résidu apparié entre architectures. Aucun des chiffres de G2.1 n'est donc
  lu comme significatif ; les seuils ci-dessous classent des effets, ils ne testent rien.

## Prédictions, pour les bras « pôles » C et D (chacun contre A)

- **H1, combinaison extrême** `B100_M000_T000_G050` (le pire réglage non vu du FiLM après
  l'étiquette fausse) : d ≥ ln 1,5 → **soutenue** ; d ≤ 0 → **réfutée** ; entre les deux →
  intermédiaire.
- **H2, médiane des 9 non vus** (étiquette corrigée), non-infériorité, marge ln 1,15 :
  médiane(d) ≥ −ln 1,15 → **non inférieur** ; sinon **inférieur**.
- **H3, MR-STFT moyenne des non vus** : ln(MR_A / MR_bras) ≥ ln 1,15 → **soutenue** ; ≤ 0 →
  **réfutée** ; entre les deux → intermédiaire. Déplacer les pôles doit d'abord corriger la forme
  spectrale.
- **H4, rapport extrême / interpolation**, R = ESR(B100_M000_T000) / ESR(B010_M030_T070). Pour A,
  R = 3,64. R_bras ≤ 0,6 × 3,64 = 2,18 → **soutenue** ; R_bras ≥ 3,64 → **réfutée** ; entre les
  deux → intermédiaire. En G2.2, la référence est le rapport de A aux mêmes partitions, et non la
  constante 3,64.
- **H5, réglages vus**, non-infériorité : médiane sur les 28 réglages vus de ln(ESR_A / ESR_bras)
  ≥ −ln 1,15 → **non inférieur** ; sinon **inférieur**.
- **H6, diagnostic, pôles figés au réglage central** : les décalages de pôles sont calculés avec
  tous les boutons à 0,5, pour tous les exemples. Le modèle se réduit alors à sa réponse au
  réglage central. Rapport médian sur les non vus ESR(figés) / ESR(mobiles) ≥ 1,5 → **le modèle
  se sert du déplacement des pôles** ; ≤ 1,15 → **il ne s'en sert pas** (les boutons sont alors
  presque ignorés) ; entre les deux → intermédiaire.

## Question de structure, sans prédiction dirigée

Pour chaque famille (FiLM : B contre A ; pôles : D contre C), médiane sur les 9 non vus de
ln(ESR_joint / ESR_additif) :

- ≥ ln 1,15 → l'additif généralise mieux ;
- ≤ −ln 1,15 → le joint généralise mieux ;
- entre les deux → indifférent à ce niveau.

L'interaction du plan 2×2 (l'additif aide-t-il plus sous « pôles » que sous FiLM ?) est
rapportée en descriptif seulement.

## Préflight et portes

- **Préflight** (`--no-record`, 200 pas, bras C, puis bras D à 200 pas).
  - Porte mémoire : pic de mémoire **réservée** (`max_memory_reserved`, cache de l'allocateur
    compris) ≤ 21,5 Gio pour tous les deux. Chaque run attend ensuite ce pic plus 1 Gio libres.
  - Le préflight attend 23 000 Mio libres (22,5 Gio, presque tout le GPU au repos) : un dépassement de mémoire y
    est alors celui du modèle, pas celui d'un autre projet. Le registre distingue « dépassement
    de mémoire » de « pic mesuré au-dessus du seuil ». Un préflight qui dépasse la mémoire ou
    sort en erreur échoue la porte, quels que soient les chiffres imprimés avant. Amendement du
    2026-09-24, avant tout run.
- **Une run G2.1 en échec arrête la file** : elle est inscrite comme échec et relancée à la main
  sous un nouveau run_id (`_r2`), sans que les bras suivants partent entre-temps.
  - Porte temps : durée extrapolée à 15 000 pas ≤ 12 h.
  - Si une porte échoue, **aucune run G2.1 ne part** : la file s'arrête et le problème est
    rapporté. Aucun réglage n'est changé en silence (ni le lot, ni la longueur des segments, ni
    le checkpointing).
- **Porte de run** : ESR finie sur tous les réglages, et aucune bascule de garde de polarité
  après l'époque 20.

## G2.1 puis G2.2, et la règle d'arrêt

**G2.1** (partition 925) : bras B, C et D, trois runs, dans cet ordre : C, D, B. Les « pôles »
passent d'abord.

**Règle d'arrêt, pré-enregistrée, qui ne se déclenche que sur un grand effet** (le bruit d'une run
seule étant inconnu). Un bras « pôles » est **arrêté** et ne passe pas à G2.2 si sa médiane(d) sur
les non vus est ≤ −ln 2 (deux fois pire que le FiLM joint en médiane), ou s'il échoue la porte de
run.

**G2.2** (partitions 736 et 688, les suivantes du tirage déclaré dans `paired_protocol.md`) :

- toujours le bras A, qui n'existe qu'à 925 ;
- le meilleur bras « pôles » non arrêté, par médiane(d) à 925. Si les deux sont non arrêtés et à
  moins de ln 1,15 l'un de l'autre, les deux passent ;
- le bras B, seulement si la question de structure a donné « l'additif généralise mieux » pour le
  FiLM.

Verdict G2.2 : les mêmes H1 à H5, sur la moyenne des d des trois partitions. Avant de juger un
chiffre proche d'un seuil, l'écart-type apparié de d **par réglage** entre partitions est estimé et
rapporté (deux degrés de liberté, donc très incertain). Un d moyen dont la valeur absolue est
inférieure à cet écart-type est rapporté comme « indistinct ». Précision de mise en œuvre, écrite
avant tout run : d est moyenné par réglage sur les partitions avant toute médiane. L'écart-type
de comparaison est celui du réglage extrême pour H1, la médiane des écarts-types par réglage pour
H2 (non vus) et H5 (vus), et celui de la quantité par partition pour H3. La mention « indistinct »
s'ajoute au libellé sans le remplacer. G2.2 ne conclut que si tous les bras retenus existent aux
trois partitions.

**Bras témoin « résidus »** (réglages appliqués aux seuls résidus C des SSM) : pas lancé en G2.1.
Il n'est écrit et lancé que si un bras « pôles » soutient H1 et H2 en G2.2, pour séparer « déplacer
les pôles » de « donner une mémoire au conditionnement ».

## Ce que G2 ne pourra pas dire

- Une graine, une à trois partitions : c'est un pilote de voie produit.
- Le Marshall de ToneTwist est capturé à 44,1 kHz et rééchantillonné par nablafx. Les réglages de
  test sont sur un DI différent de l'entraînement, ce qui mêle généralisation de réglage et de
  matériau.
- « Pôles » ne teste que les modes réels : A_imag n'est pas conditionné, par choix.

## Verdict G2.1 (2026-09-26, partition 925, `poles_verdict.py`, `poles_verdict_925.json`)

Les quatre runs passent la porte de run : ESR finies sur les réglages vus et non vus, une seule
bascule de polarité, à l'époque 0, 15 000 pas sans arrêt anticipé. Préflights : 13,4 Gio réservés,
9,1 h extrapolées.

ESR de test par réglage non vu (étiquette corrigée) :

| réglage | A FiLM joint | B FiLM additif | C pôles joint | D pôles additif |
|---|---|---|---|---|
| B000_M000_T100 | 0,121 | 0,101 | 0,122 | **0,074** |
| B000_M100_T000 | 0,042 | 0,059 | 0,057 | **0,041** |
| B000_M100_T100 | 0,076 | 0,050 | 0,122 | **0,028** |
| B010_M030_T070 (interpolation) | 0,048 | 0,046 | 0,060 | **0,025** |
| B030_M070_T010 (étiquette corrigée) | 0,018 | 0,017 | 0,029 | **0,010** |
| B065_M085_T035 | 0,015 | 0,019 | 0,030 | **0,011** |
| **B100_M000_T000 (extrême)** | 0,173 | 0,250 | **0,101** | 0,141 |
| B100_M000_T100 | 0,044 | 0,082 | 0,069 | **0,034** |
| B100_M100_T000 | 0,018 | 0,028 | 0,060 | **0,013** |
| moyenne / médiane | 0,062 / 0,044 | 0,073 / 0,050 | 0,072 / 0,060 | **0,042 / 0,028** |

| | C pôles joint | D pôles additif |
|---|---|---|
| H1, extrême (d) | **soutenue** (+0,54) | intermédiaire (+0,21) |
| H2, médiane des non vus (d) | **inférieur** (−0,44) | **non inférieur** (+0,35) |
| H3, MR-STFT (d) | intermédiaire (+0,05) | intermédiaire (+0,04) |
| H4, extrême / interpolation (A : 3,64) | **soutenue** (1,67) | **réfutée** (5,63) |
| H5, médiane des vus (d) | **inférieur** (−0,81) | **non inférieur** (+0,14) |
| H6, pôles figés / mobiles | se sert des pôles (×5,5) | se sert des pôles (×12,4) |

**Question de structure** :

- FiLM : B contre A, médiane −0,22, « le joint généralise mieux » ;
- pôles : D contre C, médiane +0,88, « l'additif généralise mieux ».

Descriptif, sans seuil : l'interaction du plan 2×2 est nette. La tête additive aide sous « pôles »
et nuit sous FiLM.

*Correction du 2026-10-02* : cette interaction ne survit pas au retrait du décalage propre à chaque
run. Si l'on retranche de chaque bras sa médiane de d sur les réglages vus, l'excès « non vus moins
vus » vaut +0,16 (B), +0,37 (C) et +0,21 (D) (`poles_verdict_925.json`) : les trois bras font mieux
que A sur les non vus relativement aux vus, et sous « pôles » l'additif ne gagne pas plus que le
joint. Une run par cellule ; l'interaction brute peut n'être qu'un effet de run.

**Règle d'arrêt et G2.2** : aucun bras « pôles » n'est arrêté (médianes −0,44 et +0,35, au-dessus
de −ln 2). D est le meilleur, et C en est à plus de ln 1,15 : seul D passe. Le FiLM additif n'est
pas retenu. **G2.2 : bras A et D, partitions 736 et 688.**

**Lecture, une partition, bruit inconnu.**

- Déplacer les pôles par une somme de fonctions d'un seul bouton (D) donne la meilleure
  généralisation aux combinaisons jamais vues, sur huit réglages sur neuf. Elle améliore
  davantage l'interpolation que l'extrême, d'où H4 réfutée.
- La tête jointe (C) fait l'inverse : c'est elle qui gagne sur la combinaison extrême, mais elle
  perd partout ailleurs, réglages vus compris.
- Les deux bras « pôles » dépendent fortement de leurs pôles : figés au réglage central, leur ESR
  est multipliée par 5 à 12.
- Rien de cela n'est encore une conclusion : c'est l'objet de G2.2, qui estimera aussi l'écart
  apparié d'une partition à l'autre.

## Écart de G2.2 (2026-09-28, écrit avant la relance)

- **Échec.** `g2_poles_additive_s688` (bras D, partition 688) s'est arrêté sur « NaN loss »
  vers le pas 9 385, sans écrire de résultat. L'échec est consigné dans `demo/RUNS.jsonl`.
- **Cause, diagnostiquée sur les checkpoints** (commit `2d49ac7`) : le noyau calcule
  `a ** k = exp(k · log a)`. Quand un mode rapide, accéléré encore par son décalage de réglage,
  fait tomber le pôle `a = exp(λ·dt)` exactement à 0 en float32 (Re(λ·dt) < −104 environ), a⁰
  donne NaN.
  - Au pas 9 300, le bloc 7 atteignait −98,9 pour un réglage d'entraînement.
  - Les gradients étaient stables (norme de 1 à 4,5).
  - Le lien avec le pas exact du NaN est inféré, pas démontré.
- **Correctif** : Re(λ·dt) est bornée à −80 avant l'exponentielle, et le poids FOH prend sa limite
  dt/z². Les bits sont inchangés au-dessus du seuil, vérifié sur 200 000 valeurs et par les
  tests au bit près. Un test permanent reproduit le bug.
- **Relance** : `g2_poles_additive_s688_r2`, drapeaux identiques, code avec le correctif.
  - En mode déterministe, elle suit le run d'origine jusqu'au premier pas où un mode passe sous
    −80, puis s'en écarte.
  - Les autres bras, faits avant le correctif, ne sont pas relancés.
  - `poles_verdict.py` prend `_r2` quand le premier essai n'a pas de résultat. Il réévalue les
    checkpoints avec le code courant : un mode sous −80 à l'évaluation y donnerait une valeur
    très légèrement différente, au lieu d'un NaN.
- **Limites connues, non corrigées** : les conversions hors ligne `zoh_to_foh` et
  `two_tap_kernel` ne sont pas protégées ; sous le seuil, la vitesse du mode ne reçoit plus de
  gradient par le pôle `a` (relecture Codex, lecture seule).

## Lecture à vérifier au verdict G2.2 (écrite le 2026-09-28, avant le verdict, sans changer les seuils)

`pole_speed.py`, `pole_speed.json`, théorie dans `pole_speed_theory.md`. On note s = −Re(λ·dt),
la décroissance par échantillon, sur les 28 réglages d'entraînement. Au-delà de s ≈ 10, un mode
n'a plus de mémoire et agit comme un gain instantané.

| run | sans mémoire pour ≥ 1 réglage | pour tous | commutés (s < 1 pour un réglage, > 10 pour un autre) |
|---|---|---|---|
| D, partition 925 | 9 | 2 | 4 |
| D, partition 736 | 4 | 1 | 1 |
| D, partition 688 (pas 9 300) | 1 | 1 | 0 |
| C, partition 925 | 5 (dont s = 101,7, à environ 2,3 de l'annulation en float32, s ≈ 103,97) | 2 | 1 |
| FiLM (A 925, 736, 688 ; B 925) | 0 | 0 | 0 |

**Lecture à vérifier** : dans le régime sans mémoire, le décalage de α agit asymptotiquement sur un
gain qui dépend du réglage. Les bras « pôles » ont donc un moyen de **régler des gains par la
vitesse des pôles**, en plus de déplacer des dynamiques comme un correcteur. Les comptes décrivent
des changements de régime, pas leur effet sur la sortie : « commuté » est le nom d'un critère
numérique. Un avantage de D, s'il se confirme, ne
prouverait pas à lui seul que les réglages déplacent des coupures. Au verdict, ces comptes seront
rapportés à côté de H1–H6. Une comparaison qui aiderait à départager (modes normalisés en gain, avec et sans tête explicite
de gain) est décrite dans `pole_speed_theory.md`, section 6. Elle n'est pas lancée.

## Verdict G2.2 (2026-09-28, partitions 925, 736 et 688, `poles_verdict.py --splits 925 736 688`)

Sortie : `poles_verdict_925_736_688.json`. Bras A et D, retenus par G2.1. Pour D à 688, le script
prend `g2_poles_additive_s688_r2`, la relance avec le plancher des pôles. Les seuils ne sont pas
touchés, cette section les applique.

**Portes.** Les six runs passent la porte de run : ESR finies sur les réglages vus et non vus, une
seule bascule de polarité par run, à l'époque 0. La partie 925 redonne au bit les valeurs de
`poles_verdict_925.json` (ESR non vues de A et de D, pôles figés de D, réglages vus). C'est un
contrôle sur CPU que le plancher de 2d49ac7 ne change pas ces deux checkpoints ; ce n'est pas un
contrôle du pilote GPU.

ESR de test par réglage non vu (étiquette corrigée), par partition 925 / 736 / 688. d est la
moyenne sur les trois partitions de ln(ESR_A / ESR_D), et sd son écart-type apparié entre
partitions (2 ddl) :

| réglage | A FiLM joint | D pôles additif | d moyen | sd |
|---|---|---|---|---|
| B000_M000_T100 | 0,121 / 0,220 / 0,089 | 0,074 / 0,128 / 0,126 | +0,228 | 0,503 |
| B000_M100_T000 | 0,042 / 0,286 / 0,059 | 0,041 / 0,074 / 0,075 | +0,372 | 0,860 |
| B000_M100_T100 | 0,076 / 0,235 / 0,112 | 0,028 / 0,055 / 0,082 | +0,919 | 0,574 |
| B010_M030_T070 (interpolation) | 0,047 / 0,059 / 0,058 | 0,025 / 0,057 / 0,064 | +0,194 | 0,392 |
| B030_M070_T010 (étiquette corrigée) | 0,018 / 0,022 / 0,030 | 0,010 / 0,015 / 0,022 | +0,433 | 0,154 |
| B065_M085_T035 | 0,015 / 0,022 / 0,028 | 0,011 / 0,028 / 0,033 | −0,016 | 0,300 |
| **B100_M000_T000 (extrême)** | 0,173 / 0,195 / 0,245 | 0,141 / 0,077 / 0,114 | **+0,633** | 0,379 |
| B100_M000_T100 | 0,044 / 0,067 / 0,051 | 0,034 / 0,067 / 0,079 | −0,054 | 0,357 |
| B100_M100_T000 | 0,018 / 0,019 / 0,027 | 0,012 / 0,020 / 0,023 | +0,159 | 0,193 |

| | D pôles additif, trois partitions | indistinct |
|---|---|---|
| H1, extrême (d) | **soutenue** (+0,63 ≥ ln 1,5) | non (sd 0,38) |
| H2, médiane des non vus (d) | **non inférieur** (+0,23) | **oui** (sd typique 0,38) |
| H3, MR-STFT (d) | intermédiaire (+0,05) | non |
| H4, extrême / interpolation | intermédiaire (2,92 ; A aux mêmes partitions : 3,71, seuil de soutien 2,23) | — |
| H5, médiane des vus (d) | **non inférieur** (+0,08) | **oui** |
| H6, pôles figés / mobiles | se sert des pôles (×9,1) | — |

Médiane de d par partition, descriptive : +0,35 à 925, +0,37 à 736, **−0,09 à 688**. L'avantage
de D ne tient pas à 688 en médiane de d. En moyenne des neuf réglages, D y reste
plus bas que A (0,069 contre 0,078), et aussi sur l'extrême (0,114 contre 0,245).

**Conséquence pré-enregistrée.** D soutient H1 et est non inférieur pour H2. Or la règle ouvre le
bras témoin « résidus » quand un bras « pôles » « soutient H1 et H2 » en G2.2. H2 est une
non-infériorité et porte la mention « indistinct ». La condition est donc remplie à la lettre, mais
sur un H2 qui ne se distingue pas du bruit entre partitions. Le témoin n'est ni écrit ni lancé :
la décision revient à l'utilisateur.

### Lecture à vérifier, confrontée (post hoc, descriptive)

`pole_speed.json`, relancé avec `g2_poles_additive_s688_r2`. Les entrées déjà présentes sont
identiques au bit.

| run D | sans mémoire pour ≥ 1 réglage | pour tous | commutés | médiane de d |
|---|---|---|---|---|
| 925 | 9 | 2 | 4 | +0,35 |
| 736 | 4 | 1 | 1 | +0,37 |
| 688 `_r2` | 1 | 1 | 0 | −0,09 |

- À 688 `_r2`, le seul mode rapide (bloc 7, canal 11, mode 2) est à s = 160–220 sur les 28
  réglages : Re(λ·dt) = −s y est partout sous −80, donc écrêtée. Le gradient par cette partie
  réelle est nul (limite connue de 2d49ac7) ; la phase et le coefficient d'entrée (a − 1)/λ
  restent différentiables. Le mode agit comme un gain quasi instantané.
- La partition sans mode commuté est celle où D ne bat pas A en médiane de d. Trois points ne
  permettent pas d'en tirer un lien, et la variation de A d'une partition à l'autre (moyenne des
  non vus 0,062, 0,125 et 0,078) contribue aussi aux différences observées. Conformément à la lecture écrite avant le verdict, un avantage de D
  ne prouve pas que les réglages déplacent des coupures. Une comparaison qui aiderait à départager (modes
  normalisés en gain, `pole_speed_theory.md` section 6) n'est pas lancée.

### Ce qui ne peut pas être conclu

- Une graine, trois partitions, deux ddl pour chaque écart-type. H2 et H5 sont indistincts.
- Le FiLM joint (A) varie fortement d'une partition à l'autre (0,286 contre 0,042 et 0,059 sur
  B000_M100_T000). Une part de d vient de là, pas de D.
- D à 688 est une relance avec un code différent au-dessous du seuil de −80, et un mode y vit sous
  ce seuil.

## Élargissement de G2.2 aux partitions 744 et 369 (pré-enregistré le 2026-09-30, avant les runs)

**Pourquoi.** Au verdict G2.2, D soutient H1, mais H2 (non-infériorité sur la médiane des non
vus) et H5 sont « indistincts », et la médiane de d s'inverse à 688 (−0,09). La condition
d'ouverture du bras témoin « résidus » n'est remplie qu'à la lettre. Lancer un témoin pour
expliquer un effet pas encore établi serait prématuré. Deux partitions de plus portent les
écarts-types entre partitions à 4 degrés de liberté. Sur Big Muff, l'appariement a été validé
sur cinq partitions, avec r₅ = 0,048 (`diagnosis/seeds/paired_protocol.md`). Décision de
l'utilisateur du 2026-09-30.

**Runs.**

- Bras A et D aux partitions 744 et 369, les deux suivantes du tirage déclaré dans
  `paired_protocol.md`. Nouveaux run_id `g2_film_joint_s{744,369}` et
  `g2_poles_additive_s{744,369}`.
- Drapeaux exacts de `scripts/product_g22_queue.sh`, entrelacés par partition (A puis D) :
  une file arrêtée laisse ainsi des partitions complètes. Une run en échec arrête la file, sans
  relance automatique.
- Code du runner `neural_amp_runner_g22r2` @ 561eef4, qui contient le plancher des pôles de
  2d49ac7. Il est identique à HEAD pour `src/` et le banc. File :
  `scripts/product_g22_ext_queue.sh`.

**Porte avant les runs, au bit.** Les runs existantes ont été entraînées à d'autres commits et
sous le pilote 595.84. Le chemin « pôles » n'a pas été porté sous le nouveau pilote (595.91.07).
Deux relances courtes, `--no-record`, comparées par `repro_gate.compare` :

- chemin D contre `g2_poles_additive_s925` (ac7274a), 300 pas ;
- chemin A contre `param_ssm_wavenet_marshall` (29a869c), 300 pas.

Au premier écart, la file s'arrête. Elle s'arrête aussi si le banc sort en erreur ou si la
relance n'atteint pas ses 300 pas. Le plancher de 2d49ac7 ne change aucun bit au-dessus de −80
(vérifié sur les checkpoints à 925, section du verdict G2.2). Un écart désignerait donc le pilote
ou un autre changement de code, et il serait rapporté avant toute run.

**Hétérogénéité connue, non neutralisée.** Les nouveaux D ont le plancher des pôles, comme
`g2_poles_additive_s688_r2`. Les anciens D, à 925 et 736, ne l'avaient pas. Leurs modes
finissent sous le plancher (s maximal 72,4 et 77,6, `pole_speed.json`), qui n'y aurait donc rien
changé à la fin de l'entraînement ; pendant l'entraînement, ce n'est pas vérifié. La porte de
300 pas ne peut pas exclure qu'un nouveau D passe sous −80 plus tard. Les comptes de
`pole_speed.py` le diront.

**Verdict.** `poles_verdict.py --splits 925 736 688 744 369`, sans modification du script :

- mêmes H1 à H6 et mêmes seuils, d moyenné par réglage sur les cinq partitions ;
- mêmes règles « indistinct », avec des écarts-types à 4 degrés de liberté ;
- D à 688 reste `g2_poles_additive_s688_r2`.

Les comptes de `pole_speed.py` (modes sans mémoire et commutés) sont ajoutés pour les deux
nouveaux D, en descriptif.

**Écart déclaré : règle d'ouverture du bras témoin « résidus ».** Cette règle est décidée après
le verdict à trois partitions, donc en connaissant ses chiffres, et avant les deux nouvelles
partitions. Le témoin ne s'ouvre que si, sur les cinq partitions, D soutient H1 et est non
inférieur pour H2, **sans la mention « indistinct »** pour H1 ni pour H2. Elle remplace la
règle d'origine pour cette décision ; les seuils de H1 à H6 ne changent pas. Elle est plus
stricte, donc elle ne peut pas rendre l'ouverture plus facile. Le verdict à trois partitions
reste tel qu'il est écrit plus haut ; il n'est pas réécrit.

**Ce que l'élargissement ne dira pas.** Il reste une seule graine d'initialisation. Le résidu
apparié de 0,048 vaut pour deux architectures sur Big Muff, pas pour ces bras sur Marshall.
Durée attendue : environ 27 h de GPU (A environ 4,7 h et D environ 8,7 h par partition, d'après
les runs existantes), sans compter les attentes du GPU partagé.

## Verdict de l'élargissement (2026-10-02, cinq partitions, `poles_verdict.py --splits 925 736 688 744 369`)

Sortie : `poles_verdict_925_736_688_744_369.json`. Bras A et D aux partitions 925, 736, 688, 744 et
369 ; à 688, D est `g2_poles_additive_s688_r2`. Les seuils et la règle du témoin sont ceux écrits
avant les runs ; cette section les applique.

**Portes.**
- Les deux portes au bit de la file sont franchies : chemin « pôles » 366 valeurs, chemin FiLM 348,
  aucun écart, 300 pas atteints.
- Les quatre nouvelles runs passent la porte de run : ESR finies, une seule bascule de polarité
  chacune, à l'époque 0, 15 000 pas.
- Les entrées des partitions 925, 736 et 688 sont identiques au bit au verdict à trois partitions.
- Incident d'exécution : la file a été tuée à 17:00:55 le 30/09 (`demo/RUNS.jsonl`). D à 744 a
  continué orphelin, sans relance, et A et D à 369 sont partis d'une continuation, mêmes drapeaux et
  même runner.
- Hétérogénéité déclarée du plancher des pôles : en fin d'entraînement, les nouveaux D n'ont aucun
  mode sous le plancher (s maximal 42,7 à 744 ; aucun au-delà de 30 à 369, `pole_speed.json`). Un
  passage sous −80 pendant l'entraînement n'est pas exclu.

d est la moyenne sur les cinq partitions de ln(ESR_A / ESR_D) par réglage non vu, et sd son
écart-type apparié entre partitions (4 ddl) :

| réglage | d moyen | sd |
|---|---|---|
| B000_M000_T100 | +0,206 | 0,423 |
| B000_M100_T000 | +0,047 | 1,097 |
| B000_M100_T100 | +0,998 | 0,490 |
| B010_M030_T070 (interpolation) | +0,227 | 0,349 |
| B030_M070_T010 (étiquette corrigée) | +0,366 | 0,165 |
| B065_M085_T035 | +0,099 | 0,354 |
| **B100_M000_T000 (extrême)** | **+0,728** | 0,330 |
| B100_M000_T100 | −0,007 | 0,282 |
| B100_M100_T000 | +0,060 | 0,287 |

| | D pôles additif, cinq partitions | indistinct |
|---|---|---|
| H1, extrême (d) | **soutenue** (+0,73 ≥ ln 1,5) | non (sd 0,33) |
| H2, médiane des non vus (d) | **non inférieur** (+0,21) | **oui** (sd typique 0,35) |
| H3, MR-STFT (d) | intermédiaire (+0,04) | non |
| H4, extrême / interpolation | intermédiaire (2,97 ; A aux mêmes partitions : 4,51, seuil de soutien 2,70) | — |
| H5, médiane des vus (d) | **non inférieur** (+0,19) | **oui** |
| H6, pôles figés / mobiles | se sert des pôles (×9,4) | — |

**Règle du témoin « résidus », écart déclaré avant les runs : le témoin ne s'ouvre pas.** H1 est
soutenue sans mention « indistinct », mais H2 la porte encore. La règle stricte exigeait les deux
sans cette mention.

**Descriptif, par partition.**

| partition | médiane de d | ESR moyenne des non vus, A / D | modes commutés de D |
|---|---|---|---|
| 925 | +0,35 | 0,062 / 0,042 | 4 |
| 736 | +0,37 | 0,125 / 0,058 | 1 |
| 688 | −0,09 | 0,078 / 0,069 | 0 |
| 744 | −0,02 | 0,056 / 0,046 | 2 |
| 369 | +0,57 | 0,093 / 0,049 | 2 |

- D a une ESR moyenne des non vus plus basse que A aux cinq partitions. En médiane de d, l'avantage
  ne tient qu'à trois partitions sur cinq.
- L'effet le plus net et le plus stable porte sur la combinaison extrême (H1). L'avantage en
  médiane, qui a motivé G2, reste du même ordre que l'écart entre partitions.
- Les modes commutés ne suivent pas la médiane de d (2 à 744, où d ≈ 0). Lecture post hoc, tirée
  d'une relecture critique sur les checkpoints et non consignée ailleurs : ces modes semblent
  commandés surtout par le bouton Gain, qui vaut 5 sur tous les réglages de test. Si c'est le cas,
  ils n'expliquent pas les écarts d'un réglage non vu à l'autre, mais ils peuvent encore contribuer
  à la sortie et à l'avantage global de D.

### Ce qui ne peut pas être conclu

- Une seule graine d'initialisation. L'écart-type entre partitions (0,35 en médiane) est celui de
  ces deux bras sur le Marshall.
- Les réglages non vus sont joués sur un autre DI que l'entraînement : généralisation de réglage et
  de matériau restent mêlées.
- Le bras « pôles » garde C et D indépendants des boutons : tout gain réglable y passe par des modes
  rapides, et tout déplacement de zéro par un déplacement de pôles. « Conditionner les pôles » ne
  teste donc pas, à lui seul, « déplacer les fréquences de coupure ».
