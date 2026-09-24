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
  deux → intermédiaire.
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
  - Porte mémoire : pic ≤ 21,5 Gio pour tous les deux.
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
inférieure à cet écart-type est rapporté comme « indistinct ».

**Bras témoin « résidus »** (réglages appliqués aux seuls résidus C des SSM) : pas lancé en G2.1.
Il n'est écrit et lancé que si un bras « pôles » soutient H1 et H2 en G2.2, pour séparer « déplacer
les pôles » de « donner une mémoire au conditionnement ».

## Ce que G2 ne pourra pas dire

- Une graine, une à trois partitions : c'est un pilote de voie produit.
- Le Marshall de ToneTwist est capturé à 44,1 kHz et rééchantillonné par nablafx. Les réglages de
  test sont sur un DI différent de l'entraînement, ce qui mêle généralisation de réglage et de
  matériau.
- « Pôles » ne teste que les modes réels : A_imag n'est pas conditionné, par choix.
