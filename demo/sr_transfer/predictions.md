# Transfert de fréquence par rééchelonnement de Δ — prédictions

Voie produit (D-PRODUCT-001) : mesure produit sur le jeu de test, aucune revendication
scientifique. Écrit le 2026-09-22, **avant** tout chiffre inter-fréquences ; seuls les
contrôles ci-dessous étaient connus. Script : `scripts/product_sr_transfer.py`.

## Objet

Un checkpoint du banc Big Muff (entraîné à 48 kHz sur la capture ToneTwist 44,1 kHz
suréchantillonnée) tourne à une autre fréquence fs en ajoutant log(48000/fs) à chaque
`log_dt`. Toute la mémoire de SSM-WaveNet passe par des SSM diagonaux en `zoh`, le reste agit
échantillon par échantillon : c'est le cas favorable. S4-TF-L-16 (nablafx) garde en plus un
TFiLM à blocs de 128 échantillons : contraste « partiellement indépendant ».

Checkpoints : `last` des runs appariés `paired_ssm_wavenet_s{925,736,688}` et
`paired_s4_tf_l_16_s{925,736}` (s688 S4 encore en entraînement ; ajouté s'il termine, même
règle). Aucun checkpoint écarté.

## Contrôles passés avant d'écrire ce fichier

- (a) Décalage nul de `log_dt` : sortie bit à bit identique (couche synthétique, float64).
- (b) Chemin d'évaluation CPU contre l'ESR enregistré sur GPU à 48 kHz :

  | run | ESR ici | ESR enregistré |
  |---|---|---|
  | ssm s925 | 0,043890 | 0,043852 |
  | ssm s736 | 0,049987 | 0,049989 |
  | ssm s688 | 0,042464 | 0,042448 |
  | s4 s925 | 0,084980 | 0,085012 |
  | s4 s736 | 0,081118 | 0,081174 |

- (c) Couche `zoh` synthétique : la réponse indicielle à 96 kHz (dt/2) prise aux indices
  2n+1 coïncide avec celle à 48 kHz (écart max 7e-15). Sur entrée tenue, idem à 1,8e-14.
  Ce contrôle a fait apparaître un mécanisme : la sortie C h[n] intègre l'entrée tenue
  jusqu'à t_n + dt, donc chaque branche SSM **devance** son passage direct D d'un échantillon,
  soit 1/fs secondes. L'entraînement a figé cette avance à 20,8 µs ; elle vaut 22,7 µs à
  44,1 kHz, 10,4 µs à 96 kHz, 5,2 µs à 192 kHz. D'où le bras exploratoire `*_lead`, qui
  multiplie C par exp(λ·dt′·(fs/48000 − 1)) pour restaurer l'avance apprise (au-dessus de
  48 kHz, il ignore l'entrée future que cette avance réclamerait).
- (d) Aucun checkpoint n'a de vecteur d'entrée appris (`zoh` confirmé par les paramètres,
  pas par le nom).

Covariable, modes avec |Im λ|·dt > π (repliés) :

| run | 48 kHz | 44,1 kHz | 96 kHz |
|---|---|---|---|
| ssm s925 | 2/512 | 3/512 | 0/512 |
| ssm s736 | 0/512 | 0/512 | 0/512 |
| ssm s688 | 0/512 | 0/512 | 0/512 |
| s4 s925 | 531/4096 | 580/4096 | 156/4096 |
| s4 s736 | 517/4096 | 575/4096 | 148/4096 |

Environ 13 % des modes de S4 sont déjà repliés à la fréquence d'entraînement : leur
équivalent continu est ambigu, et les rééchelonner les place sur la fréquence dépliée, pas
forcément sur ce que le modèle a effectivement appris.

## Bras

À 44,1 kHz, contre la capture native de l'appareil : `naive` (dt inchangé), `delta`,
`delta_lead`, `around` (rééchantillonnage vers 48 kHz, modèle, retour ; hors ligne et non
causal, borne haute), `up2_delta[_lead]` et `up4_delta[_lead]` (modèle à 88,2 / 176,4 kHz
puis retour à 44,1 kHz).
À 88,2 / 96 / 192 kHz, contre `around` à cette fréquence (cohérence avec le modèle à
48 kHz, pas fidélité) : `naive`, `delta`, `delta_lead`.

## Prédictions (SSM-WaveNet, chacune doit tenir sur les trois graines)

- **P1** `44k1_around` à ±5 % de l'ESR à 48 kHz du même checkpoint.
- **P2** `44k1_delta` ≤ 1,05 × `44k1_around`. **Règle produit** : si P2 tient sur les trois
  graines, le Δ rééchelonné suffit à 44,1 kHz sans rééchantillonneur.
- **P3** `44k1_naive` > `44k1_delta` : dt inchangé allonge toutes les constantes de temps de
  8,8 %. Ampleur attendue : faible (Wilczek et al., DAFx22 : 0,2 à 0,6 dB de SDR pour un
  LSTM entre 44,1 et 48 kHz), donc naive/delta entre 1,02 et 1,5.
- **P4** `44k1_delta_lead` à ±2 % de `44k1_delta` : l'écart d'avance à 44,1 kHz n'est que de
  1,9 µs.
- **P5** Montée, `delta_vs_around` croît avec fs (88,2 < 96 < 192), parce que l'écart
  d'avance croît (10,4 puis 15,6 µs). Ordre de grandeur : un retard de 10 µs sur une branche
  vaut 0,13 rad à 2 kHz, soit une erreur de l'ordre du pourcent. Prédiction : ESR de 0,005 à
  0,05 à 96 kHz.
- **P6** `delta_lead_vs_around` < `delta_vs_around` à 96 et 192 kHz, réduction d'au moins
  30 %. Si P6 échoue, le mécanisme d'avance n'est pas la source dominante de l'écart.
- **P7** `naive_vs_around` ≥ 5 × `delta_vs_around` à 96 et 192 kHz (dt inchangé divise
  toutes les constantes de temps par 2 ou 4).
- **P8** `up2_delta` et `up4_delta` contre l'appareil : pas de sens prédit. Le
  suréchantillonnage réduit le repliement des non-linéarités, mais l'écart d'avance et le
  fait que l'entraînement a vu le repliement à 48 kHz jouent en sens inverse. On rapporte
  le rapport à `44k1_around`.

## Contraste S4-TF-L-16

- **P9** Le gain relatif de naive à delta à 44,1 kHz est plus petit que pour SSM-WaveNet,
  et `delta_vs_around` à 96 kHz vaut au moins 3 × celui de SSM-WaveNet (TFiLM en
  échantillons et modes repliés).

## Réfutations

Chaque prédiction est jugée graine par graine ; un échec sur une graine est rapporté comme
tel, sans moyenne qui le masque. Les mesures MR-STFT ne se comparent qu'à fréquence égale.
