<!-- Copie telle quelle du scratchpad de la session du 2026-09-22 (revue_sr_independence.md). -->

# Traitement audio neuronal à fréquence d'échantillonnage variable — revue

Convention : **[V]** = vérifié par lecture du texte primaire (PDF/HTML extrait et lu) ;
**[NV]** = affirmation générale de ma part ou reprise d'un résumé de recherche non recoupé
sur la source.

---

## 0. Ce que dit la littérature en une phrase

Le problème est reconnu et nommé (« sample rate independence ») depuis 2022 (billet de
Chowdhury) et formalisé depuis 2024 par un seul groupe, celui d'Édimbourg (Carson, Wright,
Bilbao) avec Välimäki (Aalto) et Chowdhury. **Toute la littérature « effets audio » porte sur
les RNN.** Il n'existe, à ma connaissance après recherche, **aucun travail publié qui mesure
ou exploite le rééchelonnement du pas Δ d'un SSM pour un modèle d'ampli/effet à une autre
fréquence** (§3). Les solutions par temps continu et par filtres viennent d'ailleurs :
séparation de sources, classification, synthèse de piano, vision.

---

## 1. Pourquoi le problème existe, formellement

### 1.1 Le partage net : mémoire ou pas de mémoire

**[V]** DAFx24 §2 pose la distinction proprement. Pour un RNN `hⁿ = f(hⁿ⁻¹, xⁿ)`,
`yⁿ = g(hⁿ, xⁿ)`, les auteurs écrivent : *« Here we take g(·) to be a fully-connected
memoryless affine layer, which is by definition SR independent, therefore do not consider
further its influence when operating at different SRs. »*

**[V]** Chowdhury (2022) formule la même chose côté pratique : *« memoryless networks made
up of only fully-connected layers are already sample-rate agnostic »*, alors que pour les
réseaux « stateful » convolutifs ou récurrents, *« processing signal at a different sample
rate than the training data will lead to a wildly different result »*.

Donc : **toute couche sans mémoire (MLP, activation ponctuelle, gain) est invariante par
construction ; tout ce qui porte une mémoire — récurrence, convolution à noyau, état — ne
l'est pas.** Le conditionnement par paramètres de contrôle n'est pas concerné.

### 1.2 Convolutions dilatées (WaveNet / TCN)

Le champ réceptif d'une pile dilatée est fixé **en échantillons**. À f_s′ = M·f_s, il vaut
donc `RF/(M·f_s)` secondes au lieu de `RF/f_s` : la mémoire du modèle est divisée par M.
Deuxième désaccord, distinct et souvent oublié : les poids d'un noyau convolutif **sont** une
réponse impulsionnelle FIR échantillonnée ; sa réponse en fréquence est fixée en fréquence
*normalisée*, donc chaque filtre appris se translate d'un facteur M en hertz. **[NV pour la
formulation ; V pour la prémisse]** — la prémisse « un layer de convolution = une collection
de filtres FIR renversés dans le temps » est exactement l'argument de Saito et al. (§2.6) :
*« we can interpret a convolution layer as a collection of time-reversed digital finite
impulse response (FIR) filters »* **[V]**.

**[V]** La correction naturelle (multiplier les dilatations par M pour conserver
l'espacement physique des prises) est documentée non pas dans un article mais dans une
implémentation, NAM-Oversampler : `_dilation = _base_dilation * _dilation_scale`, justifié
par `Δt = d_base·S / (S·F_s) = d_base/F_s`. C'est l'analogue convolutif exact du « delay-line
trick » des RNN. Aucune mesure publiée n'accompagne cette implémentation (§3.4).

### 1.3 Récurrences (LSTM, GRU)

**[V]** DAFx24 §3.2 donne la formulation propre. Le RNN approxime un système à temps continu
avec un retard fixe T = 1/F_s :

    h(t) = f(h(t − T), x(t))                              (Eq. 5)

Échantillonné à T′ = T/M, le système **correct** est

    h(nT′) = f(h((n − M)T′), x(nT′))   →   h̃ⁿ = f(h̃ⁿ⁻ᴹ, x̃ⁿ)    (Eq. 6–7)

Conserver un retard d'**un** échantillon (méthode « naïve ») revient donc à raccourcir la
constante de temps de la boucle d'un facteur M. Sur le proxy linéaire à un pôle
`hⁿ = A hⁿ⁻¹ + xⁿ` avec `A = e^{−2πf_c/F_s}` **[V]**, la réponse naïve est
`H_NAIVE(ω) = 1/(1 − A e^{−jωT′})` (Eq. 17a) et les auteurs concluent : *« The naive method
simply stretches the function along the frequency axis, causing significant frequency warping
below the original Nyquist limit. »* **[V]** — c'est-à-dire : la fréquence de coupure apprise
se déplace proportionnellement à f_s. Tous les pôles appris migrent.

**[V]** Lecture ODE alternative (DAFx24 §3.1) : en réécrivant `hⁿ = hⁿ⁻¹ + r(hⁿ⁻¹, xⁿ)`, on
lit le RNN comme un Euler explicite de `ḣ = F_s · r(h, x)` (Eq. 3), et changer f_s revient à
diviser le résidu par M (méthode « STN »). Cette lecture converge mais, disent les auteurs,
*« it converges to the incorrect response therefore represents a different system to the
original »* — la dégradation empire quand f_s augmente.

### 1.4 Modèles d'état (S4/S4D/S5, Mamba)

**[V]** La discrétisation ZOH d'un SSM diagonal (S5, Eq. 6) :

    Λ̄ = e^{ΛΔ},   B̄ = Λ⁻¹(Λ̄ − I)B̃,   C̄ = C̃,   D̄ = D

Les paramètres appris `Λ, B̃, C̃` sont **continus** ; seul Δ porte la fréquence. D'où la
recette **[V, S5 §6.2]** : *« as [S4 and S5] are parameterized in continuous-time, these
models can be applied to datasets with different sampling rates without the need for
re-training, simply by globally re-scaling the timescale parameter Δ by the ratio between the
new and old sampling rates. »* Et **[V, S4ND §4]** : *« scaling Δ by the relative change in
frequencies (i.e., if the input resolution is doubled, halve the SSM's Δ parameter). »*

Deux réserves que la littérature énonce explicitement :

1. **[V, S4ND]** Le rééchelonnage n'est exact que si le noyau continu est à bande limitée à
   la nouvelle Nyquist : *« While S4ND can be used at any resolution, performance suffers when
   moving between resolutions due to aliasing artifacts in the kernel »*. Les valeurs propres
   dont la partie imaginaire dépasse la nouvelle Nyquist replient.
2. **[V, Piano-SSM]** Formulation audio directe : *« Of special concern are those frequencies
   f (imaginary parts of eigenvalues) violating the Nyquist criteria f > f_synth/2 »*, avec
   mesure de corrélation (§2.3).

**Mamba** : le Δ y est `Δ = softplus(linear(x))`, donc **dépendant de l'entrée** et sans
référence explicite à f_s. Je n'ai trouvé **aucun résultat publié** de transfert zéro-shot de
fréquence pour un Mamba/SSM sélectif (§3.5). C'est une piste vide, pas un acquis.

### 1.5 Non-linéarités

Ponctuellement, `tanh`, `σ`, ReLU sont invariantes : la sortie à l'instant n ne dépend que de
l'entrée à l'instant n. **Mais** les harmoniques qu'elles créent replient différemment selon
f_s. **[V, Carson DAFx25]** : *« Non-linear activation functions create both the desired
harmonic distortion and unwanted aliasing distortion as the bandwidth of the signal is
expanded beyond the Nyquist frequency. »* C'est le point de jonction avec §4.

### 1.6 Un biais rarement formulé

**[NV — mon analyse, non trouvée telle quelle dans la littérature]** Le modèle a appris la
chaîne d'acquisition à f_s_train, filtre anti-repliement compris. À 2·f_s, l'appareil réel
produirait du contenu au-dessus de l'ancienne Nyquist que le modèle n'a jamais observé. Même
une conversion de fréquence *parfaite* ne restitue donc pas le comportement haute fréquence de
la cible : elle restitue le comportement du **modèle à 48 kHz**, extrapolé. C'est exactement
ce que mesurent Carson et al. : leur référence est *la sortie du modèle de base*, pas
l'appareil analogique. **[V]** — DAFx24 : *« the signal-to-noise ratio (SNR) computed relative
to the baseline model output at 44.1 kHz »*.

---

## 2. Les solutions publiées

### 2.1 Rééchantillonner autour du modèle

**Carson, Välimäki, Wright, Bilbao — « Resampling Filter Design for Multirate Neural Audio
Effect Processing », arXiv:2501.18470 (v2, mai 2025).** Le seul travail qui chiffre coût,
latence et qualité pour le cas 44,1 ↔ 48 kHz (L/M = 160/147). **[V, Table I lue dans le PDF] :**

| Filtre | Ordre N | Latence (éch. au taux intermédiaire L·F_s) | Latence (ms) | Opérations/éch. @44,1 kHz |
|---|---|---|---|---|
| NB-Kaiser | 3318 | 1659 | 0,235 | 44,07 |
| NB-Remez | 2254 | 1277 | 0,181 | 33,67 |
| WB-Kaiser | 916 | 458 | 0,065 | 23,87 |
| WB-Remez | 698 | 349 | 0,049 | 17,93 |
| HB-IIR (elliptique ordre 13, 119,7 dB) | 13 | 0 | 0 | 18 |

**Attention à l'unité** : la colonne « Samples » compte les échantillons **au taux
intermédiaire** L·F_s = 160 × 44,1 kHz ≈ 7,06 MHz, pas à 44,1 kHz (1659 / 7,056e6 = 0,235 ms ;
458 / 7,056e6 = 0,065 ms). Lire 1659 échantillons comme 37,6 ms serait un contresens d'un
facteur 160. La légende précise aussi que pour les méthodes deux étages *« the relevant rows
must be summed for the total computational load and latency »*. Le corps du texte estime
N = 2544 pour NB-Remez via la formule (9) alors que la Table I donne 2254 ; je retiens la
valeur de la table.

Comparaison de bout en bout (Table II, S = 80 états, L/M = 160/147) **[V]** :
NB-Kaiser 88,14 op./éch. et 0,47 ms ; HB-IIR + WB-Remez 71,86 op./éch. et 0,10 ms ; contre les
méthodes « delay-line » : CIDL 609,52 op./éch. et 0 ms, LIDL 261,22 op./éch. et 0 ms. La
conception deux étages réduit la latence de *« over 70 % »* par rapport au simple étage.

Qualité **[V]** : *« the ESR is considerably lower for the single-stage resampling methods
than the nonlinear phase two-stage methods »* (propriété de phase linéaire des FIR) ;
NB-Kaiser donne le meilleur ESR moyen, CIDL le second. Sur le repliement, *« The LIDL and
CIDL methods generally produce the least aliasing out of all the methods, and in some cases
less than the original model »*.

Verdict des auteurs **[V]** : *« the resampling method is likely preferable for most
applications … unless zero phase delay and zero latency are required—in which case the model
adjustment method may be a useful alternative »*, argument décisif étant la stabilité :
*« The resampling method avoids instability since the model feedback loop is unaltered,
resulting in no such outliers. »*

Piège de conception mesuré **[V]** : les designs **Remez** (équiripple) augmentent le
repliement pour les entrées basse fréquence — *« we conclude that a uniform stopband
attenuation of 120 dB is not sufficient and instead the monotonically decreasing ripple
height of the Kaiser window method is preferable for aliasing suppression »*.

**Ce que j'ai effectivement vérifié dans le dépôt local** : `third_party/NeuralAmpModelerCore`
stocke une fréquence attendue (`mExpectedSampleRate`, `GetExpectedSampleRate()`) et **ne
rééchelonne rien** — ni les dilatations WaveNet, ni le pas de la récurrence ; `container.cpp`
se contente de vérifier la cohérence des sous-modèles et de passer `mExternalSampleRate` à
`Reset()`. Un `ResamplingContainer` avec `LanczosResampler` est fourni **dans les dépendances**
(`AudioDSPTools`, commentaire du fichier : *« Need to resample 48k to variable rate with a
small window »*). **Le câblage côté plugin n'a pas été vérifié ici** : je constate la présence
de l'outil, pas son usage.

### 2.2 Modifier le RNN : retard fractionnaire dans la boucle d'état (SRIRNN)

Origine **[V, via référence]** : J. Chowdhury, *« Sample-rate agnostic recurrent neural
networks »*, Medium, avril 2022. Limite énoncée dès l'origine : ne marche que si f_s′ ≥
f_s_train, sinon on crée une boucle sans retard.

**Carson, Wright, Chowdhury, Välimäki, Bilbao — DAFx24.** 18 modèles LSTM pré-entraînés de la
GuitarML Tone Library, base 44,1 kHz, **sur-échantillonnage uniquement** (*« SR conversion
from a higher to a lower rate is left to further work »* **[V]**). Ratios testés : M = 1,0884
(48 kHz), 2 (88,2), 2,177 (96), plus 4 et 4,444 dans l'étude de cas. Quatre méthodes :

- **STN** (Parker et al.) : `h̃ⁿ = (1 − 1/M)h̃ⁿ⁻¹ + (1/M)f(h̃ⁿ⁻¹, x̃ⁿ)` — effet passe-bas,
  se dégrade quand M croît.
- **LIDL** : interpolation linéaire entre les deux états voisins (Chowdhury).
- **APDL** : passe-tout du 1er ordre, `η = (1 − Δ)/(1 + Δ)` — proposé ici.
- **CIDL** : Lagrange cubique (K = 3) — proposé ici.

Résultats **[V]** :
- M entier : les trois méthodes à retard se réduisent à un retard pur et sont **exactes**.
  *« for integer sampling ratios the SNRH is around 100 dB – meaning the harmonic amplitudes
  match the baseline to near machine precision (operating at single-precision) »*. Gain sur
  STN à M = 2 : *« a minimum of 18 dB (“MesaMiniRec”) and up to 50 dB increase
  (“DumbleLowG”) in SNR »*.
- M non entier (44,1 → 48 kHz) : **APDL échoue catastrophiquement** sur 3 modèles sur 18
  (6505Plus, MesaIICplus, MesaMiniRec) — *« the APDL method fails catastrophically giving
  nearly 0 dB SNRH »*, attribué à l'annulation pôle-zéro quand Δ < 0,1, avec oscillations à
  Nyquist. *« this is a concerning result given that guitar and bass have harmonic content in
  this range »*. CIDL ne présente pas ces anomalies.
- Coût (RTNeural, GRU 24→72, M1 MacBook Pro, variation du facteur temps réel) **[V]** :
  Delay −0,56 %, STN −0,57 %, LIDL −2,13 %, APDL −3,24 %, CIDL −3,17 %. **Négligeable.**

**Carson, Wright, Bilbao — ICASSP 2025 (arXiv:2409.15884).** 160 LSTM GuitarML (hidden 40,
H = 80), ratios 160/147 (44,1 → 48 kHz) et 147/160 (44,1 → 40,5 kHz), filtres Lagrange-K et
minimax-K, K = 1…5. Résultats **[V]** :
- Sur-échantillonnage : Lagrange-5 meilleur dans **58,1 %** des cas (jusqu'à **80 dB** de
  SNR) mais **pire que la méthode naïve dans 8,1 %** des cas, *« with a SNR as low as −5 dB »*.
  Lagrange-3 comme compromis unique : SNR de **19 à 73 dB**.
- Sous-échantillonnage (avance fractionnaire par extrapolation) : *« the minimum SNR for all
  methods is less than 0 dB »* ; dans **3,1 %** des cas la méthode naïve est la meilleure.
- **Le diagnostic a priori** : linéarisation autour d'un point fixe, `vₙ = A vₙ₋₁ + b` avec
  `A = [[l ⊗ Jₐ], [I, 0]]`, `Jₐ = ∇f(a)ᵀ` ; stabilité ⟺ `max|eig(A)| ≤ 1`. Sur
  3200 combinaisons modèle×filtre, la prédiction binaire est correcte dans **97,8 %**
  ((2868 + 261)/3200) des cas. Étude de cas MatchlessSC30 : l'extrapolation linéaire fait
  sortir une paire de pôles conjugués du cercle unité et le modèle auto-oscille à ≈ 0,22·F_s.

C'est, à ma connaissance, **le seul critère publié permettant de savoir à l'avance si un
modèle donné supportera un changement de fréquence.**

### 2.3 Temps continu : neural ODE

**Wilczek, Wright, Välimäki, Habets — DAFx22 (« Virtual Analog Modeling of Distortion
Circuits Using Neural Ordinary Differential Equations »).** Résumé **[V]** : *« We show that
this approach does not require oversampling and allows to increase the sampling rate after the
training has completed, which results in increased accuracy. »* Astuce numérique **[V]** :
l'ODE est mise à l'échelle par f_s pour que le pas valle 1.

Ce papier contient **la mesure la plus directe de « ne rien faire »** que j'aie trouvée pour
un LSTM (SDR en dB, entraînement à 44,1 kHz, circuits SPICE) :

*Table 2 — diode clipper 1er ordre* **[V]**

| Modèle | 44,1 | 22,05 | 48 | 192 |
|---|---|---|---|---|
| LSTM8 | 30,9 | 18,5 | 30,7 | **19,6** |
| STN 3×4 | 20,4 | −6,6 | 21,3 | 21,5 |
| ODENet9-FE | 21,3 | 2,6 | 21,7 | 23,6 |
| ODENet9-IA | 26,4 | −0,8 | 27,3 | 27,7 |

*Table 4 — diode clipper 2nd ordre* **[V]**

| Modèle | 44,1 | 22,05 | 48 | 192 |
|---|---|---|---|---|
| LSTM16 | 21,2 | 13,5 | 20,6 | **7,9** |
| STN 2×30 | 16,9 | 15,7 | 16,8 | 16,3 |
| ODENet20-RK4 | 14,3 | 9,0 | 14,2 | 14,1 |
| ODENet30-FE | 11,6 | 6,4 | 11,5 | 11,0 |
| ODENet30-TR | 15,5 | 8,2 | 15,4 | 13,9 |
| ODENet30-RK4 | 15,7 | 14,7 | 15,6 | 15,7 |

Lecture : le LSTM naïf perd **0,2–0,6 dB à 48 kHz** (ratio 1,088) mais **11,3 puis 13,3 dB à
192 kHz** ; les modèles informés de la fréquence sont plats voire meilleurs à 192 kHz. Et
symétriquement, à 22,05 kHz ce sont les modèles à temps continu qui **s'effondrent**
(STN −6,6 dB, ODENet9-IA −0,8 dB) : *« For 22.05 kHz test sampling rate, STN and ODENet models
produced aliased output … This is to be expected because these models were trained with a
higher sampling rate. »* **[V]** C'est la démonstration expérimentale la plus nette que
descendre en fréquence est un problème de repliement, pas de discrétisation.

Réserves : circuits jouets (diode clippers simulés SPICE), réseaux minuscules (542 à 1250
paramètres), métrique SDR, pas de test d'écoute. **Ce n'est pas un ampli.**

Note bibliographique utile **[V]** : sur la State-Trajectory Network de Parker, Esqueda,
Bergner (DAFx19), Wilczek et al. écrivent *« STN can be used at sampling rates different than
the sampling rate of the training set by properly scaling the residual but the authors did not
provide any results concerning this feature. »* Le DAFx24 confirme l'autre limite : la STN
*« requires additional state information to be recorded from the target device »*.

### 2.4 Temps continu : modèles d'état

**Ce qui est mesuré, et sur quelles tâches** (aucune n'est une régression d'effet audio) :

*S4 (Gu, Goel, Ré — ICLR 2022, arXiv:2111.00396), Table 5, Speech Commands 10* **[V]** :

| Modèle | MFCC | Brut 16 kHz | 0,5× (8 kHz, zéro-shot) |
|---|---|---|---|
| CKConv | 95,3 | 71,66 | 65,96 |
| Performer | 80,85 | 30,77 | 30,68 |
| **S4** | 93,96 | **98,32** | **96,30** |

*« As a continuous-time model, S4 automatically adapts to data sampled at different rates …
Without re-training, S4 achieves 96.3 % accuracy at 0.5× the frequency on Speech Commands 10,
simply by changing its internal step size Δ. »* **[V]**

**Réserve que les auteurs eux-mêmes imposent** : la légende de la Table 5 porte
*« Please read Appendix D.5 before citing this table »*, et l'annexe D.5 précise **[V]** que
SC10 est un sous-ensemble à 10 classes désormais abandonné, et que *« To cite S4 as a baseline
for Speech Commands, please use Table 11 from [19] instead of Table 5 from this paper »*. Le
chiffre « 96,3 % en zéro-shot » le plus cité est donc celui que les auteurs demandent de ne
plus citer. Les valeurs à retenir sont celles sur les 35 classes (table S5 ci-dessous) :
S4-LegS y passe de 96,08 à **91,32 %**, soit −4,76 points et non −2.

*S5 (Smith, Warrington, Linderman — ICLR 2023, arXiv:2208.04933), Table 2, SC35, 16 → 8 kHz
par décimation naïve* **[V]** :

| Modèle | Param. | 16 kHz | 8 kHz zéro-shot |
|---|---|---|---|
| InceptionNet | 481 K | 61,24 | **05,18** |
| ResNet-1 | 216 K | 77,86 | **08,74** |
| XResNet-50 | 904 K | 83,01 | **07,72** |
| ConvNet | 26,2 M | 95,51 | **07,26** |
| S4-LegS | 307 K | 96,08 | 91,32 |
| S4D-LegS | 306 K | 95,83 | 91,08 |
| Liquid-S4 | 224 K | 96,78 | 90,00 |
| S5 | 280 K | 96,52 | **94,53** |

C'est le contraste le plus violent de toute la revue : les CNN passent sous le hasard
(7 % pour 35 classes), les SSM perdent 2 à 7 points. À noter aussi : **Liquid-S4, dont le
noyau dépend de l'entrée, est le SSM qui perd le plus** (−6,78 points) — indice défavorable
pour les variantes sélectives type Mamba.

*S4ND (Nguyen, Goel, Gu et al. — NeurIPS 2022, arXiv:2210.06583)* : la correction du
repliement de **noyau**. Mécanisme **[V]** : pour un A diagonal d'éléments aₙ, chaque
fonction de base est `Kₙ(t) = e^{t aₙ} Bₙ`, la fréquence étant portée par Im(aₙ) ; on
**masque le coefficient Cₙ** correspondant au-delà d'un seuil réglé par α, *« theoretically,
α = 1.0 corresponds to the Nyquist cutoff if the basis functions are pure sinusoids. However,
due to the decay e^{Re(aₙ)} arising from the real part as well as approximations arising from
using finite-state SSMs, α often has to be set lower empirically. »* Sans ce masquage,
*« ablations showing absolute performance degradation of up to 20 %+ »* **[V]**. Le même
papier énonce la nuance importante : *« While S4 was capable of transferring between different
resolutions on audio data, visual data presents a greater challenge »*.

*Piano-SSM (Dallinger et al. — DAFx25, paper 57)* : **le travail le plus proche du besoin**,
parce qu'il est en audio brut, qu'il re-discrétise réellement un SSM diagonal à une autre
fréquence, et qu'il relie la perte à la Nyquist. Table 3 **[V]** :

| Config. | Train (kHz) | Test (kHz) | Valeurs propres repliées | Erreur MSSL |
|---|---|---|---|---|
| XL | 44,1 | 24,0 | 13,09 % (134/1024) | +11,77 % |
| XL | 44,1 | 16,0 | 31,64 % (324/1024) | +16,55 % |
| XL | 24,0 | 16,0 | 11,23 % (115/1024) | +10,73 % |
| L | 44,1 | 24,0 | 6,64 % (34/512) | +6,77 % |
| L | 44,1 | 16,0 | 15,23 % (78/512) | +11,86 % |
| L | 24,0 | 16,0 | 6,64 % (34/512) | +7,21 % |
| S | 44,1 | 24,0 | 0,78 % (2/256) | +10,37 % |
| S | 44,1 | 16,0 | 2,34 % (6/256) | +13,38 % |
| S | 24,0 | 16,0 | 1,56 % (4/256) | +5,93 % |

Argument des auteurs **[V]** : *« we can investigate the learned state dynamics by analyzing
the eigenvalues … the number of aliased eigenvalues (imaginary parts violating Nyquist)
correlates with the error induced due to discretization »*. La corrélation est réelle mais
**lâche** : la config S perd +10,37 % avec 0,78 % de valeurs propres repliées, la config L
perd +6,77 % avec 6,64 % — c'est-à-dire que la proportion ne prédit pas la perte d'un modèle
à l'autre. Leur verdict prudent : *« The baselines have always lower MSSL, but the error
incorporated due to system downsampling is acceptable. »* **Et c'est de la descente en
fréquence ; la montée n'est pas testée.**

### 2.5 Conditionnement explicite sur f_s ou sur Δt

- **NU-Wave 2 (Han & Lee, Interspeech 2022, arXiv:2206.08545)** : conditionne sur la **bande
  passante de l'entrée** via la couche BSFT (γ ⊙ h + β, paramètres issus d'un one-hot de bande
  dans le domaine spectral), pour une sortie **toujours à 48 kHz**. **[NV — lu via résumé de
  recherche, pas le PDF]** C'est un recouvrement partiel : le conditionnement porte sur la
  bande du signal, pas sur le pas de temps du modèle, et la tâche est la super-résolution, pas
  le traitement à fréquence variable.
- **S5** : `Δt` fourni pas à pas pour des observations irrégulières (§6.3) **[V]** — c'est le
  mécanisme le plus proche d'un « conditionnement sur le pas de temps », mais évalué sur une
  régression de pendule.
- **Effets audio : rien.** Je n'ai trouvé aucun modèle d'ampli ou d'effet conditionné sur f_s.

### 2.6 Filtres et interpolation : rendre la convolution indépendante de f_s

**Saito, Nakamura, Yatabe, Koizumi, Saruwatari — « Sampling-Frequency-Independent Audio Source
Separation Using Convolution Layer Based on Impulse Invariant Method », arXiv:2105.04079
(EUSIPCO 2021).** L'idée la plus transférable de toute la revue pour une architecture
convolutive. Mécanisme **[V]** : on n'apprend pas des réponses impulsionnelles discrètes mais
des **filtres analogiques latents** (ici des gammatones multiphases,
`g(t) = a t^{p−1} e^{−2πbt} cos(2πft + φ)`, avec f et φ appris) ; à l'exécution, la méthode
d'invariance impulsionnelle échantillonne ces filtres continus à la f_s demandée et remplit
les poids du layer. *« Since steps (i) and (ii) depend only on the sampling frequency and the
continuous-time impulse responses, we only need to perform them once … whenever the sampling
frequency changes. »* Coût d'exécution : nul.

Résultat **[V, texte ; les valeurs de SDR ne sont qu'en figure]** : entraînement à 16 kHz
seulement, test de 8 à 48 kHz **sans rééchantillonner** — *« As the sampling frequency moved
away from 16 kHz, the SDRs of Conv-TasNet and T-MP-GTF greatly decreased and they failed to
separate the sources. By contrast, the proposed models … provided similar SDRs for the 12- to
48-kHz-sampled data and outperformed the other models by a large margin »*. Point de jonction
avec §4 : la variante « Proposed+ » ajoute une réduction de repliement qui n'est nécessaire
que **en descendant** vers 8 kHz.

**Extension 2026 [V pour les métadonnées, NV pour les chiffres] :** Nakamura, Nakata,
Imamura, Saito, *« Neural Audio Codec with Adjustable Token Temporal Resolution Using
Sampling-Frequency-Independent Convolutional Layers »*, arXiv:2607.01865, accepté à IWAENC
2026 — même mécanisme (noyaux générés depuis un jeu de paramètres partagé, taille et pas
ajustés), appliqué à la résolution temporelle des tokens d'un Descript Audio Codec.

**CKConv (Romero et al., ICLR 2022, arXiv:2102.02611)** : noyau convolutif = MLP continu
`MLPψ(t)`, évaluable à n'importe quel pas. Table 5 **[V]**, Speech Commands brut, entraîné à
la résolution 1 :

| Test → | 1 | 1/2 | 1/4 | 1/8 | 1/16 |
|---|---|---|---|---|---|
| SC_raw | 71,66 | 65,96 | 52,11 | 40,33 | 30,87 |

et sur CharacterTrajectories 1 → 1/2 : GRU-D 23,1, ODE-RNN 41,8, LMU 44,7, **NCDE 6,0**,
HiPPO 88,8, **CKCNN 99,30**. Deux leçons : (a) « temps continu » ne garantit rien — les neural
ODE/CDE s'effondrent ; (b) même un noyau continu perd 5,7 points à moitié fréquence sur
l'audio brut.

**FlexConv (Romero et al., ICLR 2022, arXiv:2110.08059)** **[NV — lu via résumé de recherche]**
ajoute exactement ce qui manque à CKConv : paramétrisation MAGNet (Gabor anisotrope
multiplicatif) dont la fréquence maximale est **dérivable analytiquement** et pénalisée
pendant l'entraînement dès qu'elle dépasse la Nyquist de la résolution d'entraînement. C'est
l'analogue convolutif du bandlimiting de S4ND.

### 2.7 Entraînement multi-fréquence

**En parole, oui ; en effets, rien.** Le défi **URGENT** (Interspeech 2025, arXiv:2505.23212)
**[NV — lu via résumé de recherche]** impose un **modèle unique** couvrant 8, 16, 22,05, 24,
32, 44,1 et 48 kHz, et la recette dominante des soumissions est la **STFT « SFI »** : fenêtre
et saut fixés **en millisecondes** et non en échantillons, de sorte que la représentation
temps-fréquence est identique quelle que soit f_s. Cette astuce ne se transpose pas à un
modèle temporel échantillon par échantillon.

Je n'ai trouvé **aucun travail d'entraînement multi-fréquence pour la modélisation d'amplis
ou d'effets.** Avertissement : une de mes recherches web a produit un résumé affirmant que le
papier SRIRNN décrivait un entraînement simultané sur plusieurs fréquences, évalué sur
« virtual analog modeling, rhythm sequencing, and piano transcription ». **C'est faux** :
j'ai lu les deux papiers, ils n'entraînent rien, ils modifient des modèles déjà entraînés, et
n'évaluent que des LSTM d'effets guitare. Résumé de moteur de recherche hallucinée, à ne pas
propager.

---

## 3. Ce qui reste ouvert

### 3.1 Oui, il existe des mesures de « ne rien faire » — mais peu, et hors amplis

Récapitulatif de ce qui est **publié et chiffré** :

| Source | Famille | Ce qui est mesuré | Chiffre |
|---|---|---|---|
| Wilczek DAFx22 T.2 | LSTM8 | SDR, 44,1 → 192 kHz | 30,9 → 19,6 dB |
| Wilczek DAFx22 T.4 | LSTM16 | SDR, 44,1 → 192 kHz | 21,2 → 7,9 dB |
| Wilczek DAFx22 T.4 | LSTM16 | SDR, 44,1 → 48 kHz | 21,2 → 20,6 dB |
| Wilczek DAFx22 T.4 | LSTM16 | SDR, 44,1 → 22,05 kHz | 21,2 → 13,5 dB |
| S5 T.2 | ConvNet 26,2 M | Précision SC35, 16 → 8 kHz | 95,51 → 7,26 % |
| S4 T.5 | CKConv | Précision SC10, 16 → 8 kHz | 71,66 → 65,96 % |
| Piano-SSM T.3 | SSM diagonal | MSSL, 44,1 → 16 kHz (re-discrétisé) | +11,86 à +16,55 % |

Et ce qui manque : **dans les deux papiers de référence sur les effets (DAFx24, ICASSP25), la
méthode naïve n'apparaît que dans des figures** (Fig. 7 et Fig. 2 respectivement, un
« violin plot » et un diagramme SNR par modèle). Le texte ne donne qu'une borne indirecte :
Lagrange-5 est parfois *« worse than the naïve method with a SNR as low as −5 dB »* **[V]** —
donc le naïf est quelque part sous ce niveau — et *« in all cases, any of the methods outlined
in Section 3 provide a significant improvement in SNR compared to the naive method »* **[V]**.
**Il n'existe pas, à ma connaissance, de table publiée donnant le SNR/ESR naïf d'un LSTM
d'ampli à 48, 88,2, 96 et 192 kHz.** Les données brutes sont dans les dépôts
`a-carson/dafx24_sr_indie_rnn` et `a-carson/icassp25_srirnns`, pas dans les articles.

### 3.2 La descente en fréquence est un verrou ouvert

DAFx24 : *« SR conversion from a higher to a lower rate is left to further work »* **[V]**.
ICASSP25 mesure et conclut **[V]** : *« there is no single filter for undersampling that will
give reasonable results across all models »*, le minimum de SNR étant < 0 dB pour toutes les
méthodes. L'article de rééchantillonnage confirme **[V]** : *« for mid to high frequencies the
CEDL method fails, giving poorer quality by all metrics compared to the naive method »*, avec
des modèles rendus **instables**. Pour un modèle entraîné à 48 kHz et un hôte à 44,1 kHz, la
seule option robuste publiée est donc le rééchantillonnage externe — pas la modification du
modèle.

### 3.3 Pas de mesure perceptive du décalage de fréquence

Je n'ai trouvé **aucun test d'écoute** portant sur « le même modèle à 48 puis à 96 kHz ». Les
MUSHRA existants dans ce corpus portent sur autre chose : Carson DAFx25 fait écouter
teacher/student pour l'**anti-repliement** (16 participants, 12 retenus après screening
relâché à 80 % au lieu de 90 %) **[V]**. La seule ancre perceptive disponible dans la ligne
« fréquence variable » est le seuil NMR de −10 dB (§4).

### 3.4 Le côté convolutif est documenté par du code, pas par de la science

L'analogue convolutif du delay-line trick (multiplier les dilatations par M) est implémenté
(NAM-Oversampler, et un plugin LV2 qui affirme dans son README que *« If you run at a sample
rate that is an even multiple of the model sample rate, the model will be correctly
oversampled »*) mais **aucune mesure publiée** ne l'accompagne. NAM-Oversampler annonce
*« Exact Tone Preservation »* **sans aucun chiffre** — un dépôt qui contient pourtant
`run_null_tests.py` et `grid_search_results.csv`. À vérifier soi-même, pas à croire.
Les rares articles qui touchent aux CNN d'effets et à la fréquence (Carson DAFx25, avec un
TCN) ne changent pas la fréquence : ils combattent le repliement à fréquence fixe.

### 3.5 Pistes explicitement vides

Je le dis sans enrobage, ce sont des résultats de recherche négatifs :

1. **Aucun SSM d'effet audio testé à une autre fréquence.** J'ai lu Simionato & Fasciani,
   *« Comparative study of state-based neural networks for virtual analog audio effects
   modeling »* (EURASIP JASM 2025, arXiv:2405.04124) : l'expression « sampling rate » y
   apparaît **une seule fois**, pour dire que les données ont été enregistrées à 48 kHz. Aucune
   expérience inter-fréquences, aucune mention de l'invariance. Idem pour le modèle de
   compresseur à couches S4 (arXiv:2403.16331) : une seule mention, « 44.1 kHz sampling rate ».
   **La propriété la plus vantée des SSM n'a jamais été testée sur la tâche qui vous occupe.**
2. **Aucun résultat de rééchelonnage de Δ pour Mamba / SSM sélectif.** Le Δ y est
   `softplus(linear(x))`, fonction de l'entrée : rien n'y référence f_s. Le seul indice est
   indirect et négatif : Liquid-S4, autre variante à noyau dépendant de l'entrée, est celle qui
   perd le plus en zéro-shot 16 → 8 kHz (96,78 → 90,00) **[V, S5 T.2]**.
3. **Aucun modèle d'effet conditionné sur f_s**, ni aucun entraînement multi-fréquence pour
   les effets.
4. **Aucun article DAFx25 ou ICASSP 2026 sur l'indépendance à la fréquence** : j'ai parcouru
   l'archive DAFx 2025 ; le seul descendant direct de la ligne Carson y est l'anti-repliement
   par fine-tuning (paper 31). Piano-SSM (paper 57) est le seul autre à traiter d'une
   fréquence de synthèse variable. Je n'affirme pas l'exhaustivité d'une recherche bornée.
5. **Non recoupé** : aucune littérature trouvée sur la stabilité **long terme** d'un
   LSTM/SSM à fréquence changée dans un contexte rebouclé (feedback, larsen simulé).

---

## 4. Lien avec l'aliasing : où les deux questions se recoupent, où elles divergent

### 4.1 Le recouvrement : le sur-échantillonnage est la même opération

C'est explicite dès le cadrage de DAFx24 **[V]** : *« We would expect that oversampling
reduces aliasing artifacts caused by the non-linear activations – which could in fact improve
the perceived quality – therefore we seek an oversampling method that reduces aliasing without
otherwise affecting the original harmonic spectral content. »* Le même mécanisme — allonger le
retard d'état à M échantillons — résout le changement de fréquence **et** réduit le
repliement ; les deux sont mesurés ensemble via deux métriques séparées, SNRH (harmoniques) et
SNRA (repliement) **[V]**.

Mesures d'audibilité **[V, arXiv:2501.18470]** : *« at the training sample rate (M = 1) the
NMR at f₀ = 4186 Hz exceeds the −10 dB approximate threshold of audibility in 11 out of 14
models. Oversampling reduces the NMR progressively with M = 8 reducing it below −10 dB for all
models. »* Deux conséquences : le repliement est **déjà audible à la fréquence
d'entraînement** dans la majorité des modèles d'ampli testés ; et il faut ×8 pour le rendre
inaudible sur tous.

### 4.2 Le point où elles divergent : le spectre miroir

**[V, DAFx24 §4.1]** À M entier, les méthodes à retard reproduisent **exactement** la réponse
d'origine dans la bande, mais *« This comes at the cost of frequency aliasing above the
original SR … when operating at an oversampled rate of M = 2, the original spectrum is
mirrored about the original SR. »* La parade n'est pas un réglage du modèle mais une
hypothèse sur le signal : *« assuming the input signal, x, is bandlimited to the original
Nyquist limit then this high frequency behaviour will not be present in the output signal. »*

Autrement dit : **un modèle rendu indépendant de la fréquence par retard reste sourd
au-dessus de l'ancienne Nyquist.** Passer l'hôte à 96 kHz ne fait pas apparaître de contenu
que le modèle n'a pas appris ; cela déplace seulement le repliement hors de la bande audible.
Les deux questions sont donc liées par l'outil (sur-échantillonnage) mais **pas** par
l'objectif : l'une vise la fidélité au modèle de référence, l'autre la fidélité à l'appareil.

### 4.3 La même dualité côté SSM

Chez les SSM le repliement n'est pas dans le signal mais dans le **noyau** : ce sont les
valeurs propres dont Im(λ) dépasse la nouvelle Nyquist qui replient (§1.4). D'où le
bandlimiting de S4ND (masquer les Cₙ) **[V]** et l'analyse en valeurs propres de Piano-SSM
**[V]**. **Conséquence asymétrique et importante pour vous :** en *montant* de 48 à 96 kHz, la
Nyquist s'éloigne — aucune valeur propre apprise ne peut se mettre à replier, le
rééchelonnage de Δ est sûr de ce point de vue ; c'est en *descendant* que le problème apparaît.
La dégradation résiduelle en montant vient d'ailleurs : des non-linéarités et des couches GLU
intercalées entre les blocs SSM, qui restent ponctuelles et dont les harmoniques replient
différemment à chaque fréquence.

### 4.4 Le cadre formel générique (hors audio)

**StyleGAN3 / Alias-Free GAN (Karras et al., NeurIPS 2021, arXiv:2106.12423)** **[NV — lu via
résumé de recherche]** est la formulation la plus générale : interpréter **tout** signal
interne comme continu, et encadrer chaque non-linéarité ponctuelle par
**sur-échantillonner → non-linéarité → passe-bas → sous-échantillonner**, avec échantillonnage
non critique (coupure sous la demi-fréquence). C'est exactement la recette de l'anti-repliement
audio par sur-échantillonnage local, appliquée couche par couche. Le lien avec la fréquence
variable est direct : un réseau alias-free est **par construction** défini sur un signal
continu, donc rééchantillonnable.

### 4.5 L'anti-repliement sans changer de fréquence

Pour compléter le tableau, la branche qui traite le repliement **sans** toucher à f_s :
- **Carson, Wright, Bilbao — DAFx25 (arXiv:2505.11375)** : fine-tuning teacher/student contre
  un jeu de données sans repliement (sinusoïdes passées dans le modèle original, composantes
  non harmoniques retirées du spectre). *« In the majority of our case studies, the reduction
  in aliasing was greater than that achieved by two times oversampling »* **[V]**. Effet
  secondaire mesuré : les harmoniques désirées sont affectées aussi, et le compromis est
  **dépendant de l'architecture** — MUSHRA : LSTM student ≈ teacher (« Excellent »), TCN
  student dégradé de « Excellent » à « Very Good » **[V]**. Les auteurs notent aussi que
  l'ADAA (antiderivative antialiasing, Parker et al.) *« work[s] well for memoryless nonlinear
  functions … and virtual analog models with one or two states. However, the application of
  ADAA to larger state-space systems or neural networks is an open research question. »* **[V]**
- Vanhatalo, Legrand, Desainte-Catherine, Hanna, Pille, *« Evaluation of real-time aliasing
  reduction methods in neural networks for nonlinear audio effects modelling »*, JAES 72,
  pp. 114–122, mars 2024 **[NV — référence relevée dans DAFx25, non lue]** : compare données
  d'entraînement sur-échantillonnées, passe-bas entre couches, pertes spectrales et parcimonie ;
  d'après le résumé qu'en fait Carson, la parcimonie forcée marche pour un TCN, moins pour un
  LSTM, et au prix de la précision.
- Köper & Holters, *« Antialiased state trajectory neural networks for virtual analog
  modeling »*, DAFx23 **[NV — référence relevée, non lue]** : limite rapportée par Carson —
  entraînable uniquement sur données synthétiques.

---

## 5. Sources

**Ligne « effets audio », groupe Édimbourg / Aalto**
- Carson, Wright, Chowdhury, Välimäki, Bilbao, *Sample Rate Independent Recurrent Neural
  Networks for Audio Effects Processing*, **DAFx24** — [V, PDF lu intégralement]
  https://arxiv.org/abs/2406.06293 · https://dafx.de/paper-archive/2024/papers/DAFx24_paper_68.pdf
  · exemples audio : https://a-carson.github.io/dafx24_sr_indie_rnn/ ·
  https://a-carson.github.io/sr_indie_rnn/ · code : https://github.com/a-carson/dafx24_sr_indie_rnn
- Carson, Wright, Bilbao, *Interpolation Filter Design for Sample Rate Independent Audio
  Effect RNNs*, **ICASSP 2025** — [V, PDF lu intégralement]
  https://arxiv.org/abs/2409.15884 · exemples : https://a-carson.github.io/icassp25_srirnns/
- Carson, Välimäki, Wright, Bilbao, *Resampling Filter Design for Multirate Neural Audio
  Effect Processing*, préprint arXiv v2, mai 2025 — [V, HTML + PDF]
  https://arxiv.org/abs/2501.18470
- Carson, Wright, Bilbao, *Anti-aliasing of Neural Distortion Effects via Model Fine Tuning*,
  **DAFx25** — [V, PDF lu] https://arxiv.org/abs/2505.11375 ·
  https://dafx.de/paper-archive/2025/DAFx25_paper_31.pdf ·
  https://a-carson.github.io/dafx25_antialiasing_neural/
- Chowdhury, *Sample-Rate Agnostic Recurrent Neural Networks*, Medium, avril 2022 —
  [NV, contenu connu par les citations qu'en font DAFx24 et DAFx25 + résumé de recherche]
  https://jatinchowdhury18.medium.com/sample-rate-agnostic-recurrent-neural-networks-238731446b2
- RTNeural (bibliothèque d'inférence temps réel utilisée pour les mesures de coût) — [NV]
  https://github.com/jatinchowdhury18/RTNeural · https://arxiv.org/abs/2106.03037

**Temps continu, effets audio**
- Wilczek, Wright, Välimäki, Habets, *Virtual Analog Modeling of Distortion Circuits Using
  Neural Ordinary Differential Equations*, **DAFx22** — [V, PDF lu, Tables 2 et 4]
  https://www.dafx.de/paper-archive/2022/papers/DAFx20in22_paper_12.pdf ·
  https://thewolfsound.com/publications/dafx2022/
- Parker, Esqueda, Bergner, *Modelling of Nonlinear State-Space Systems Using a Deep Neural
  Network* (State Trajectory Network), **DAFx19** — [NV, connu par les citations de DAFx24 et
  DAFx22] https://dafx.de/paper-archive/2019/DAFx2019_paper_42.pdf

**SSM / temps continu, hors effets**
- Gu, Goel, Ré, *Efficiently Modeling Long Sequences with Structured State Spaces* (S4),
  ICLR 2022 — [V, PDF lu, Table 5 + Annexe D.5] https://arxiv.org/abs/2111.00396
- Smith, Warrington, Linderman, *Simplified State Space Layers for Sequence Modeling* (S5),
  ICLR 2023 — [V, PDF lu, Table 2 + §6.2/6.3] https://arxiv.org/abs/2208.04933
- Nguyen, Goel, Gu, Downs, Shah, Dao, Baccus, Ré, *S4ND: Modeling Images and Videos as
  Multidimensional Signals Using State Spaces*, NeurIPS 2022 — [V, PDF lu, §4 bandlimiting]
  https://arxiv.org/abs/2210.06583
- Dallinger et al., *Piano-SSM: Diagonal State Space Models for Efficient MIDI-to-Raw Audio
  Synthesis*, **DAFx25** — [V, PDF lu, §3.1, §5.2, Table 3]
  https://dafx25.dii.univpm.it/wp-content/uploads/2025/07/DAFx25_paper_57.pdf ·
  code : https://github.com/domdal/piano-ssm

**SSM appliqués aux effets — sans expérience inter-fréquences (résultat négatif)**
- Simionato, Fasciani, *Comparative Study of State-based Neural Networks for Virtual Analog
  Audio Effects Modeling*, EURASIP JASMP 2025 — [V, PDF lu et grep exhaustif : une seule
  occurrence de « sampling rate »] https://arxiv.org/abs/2405.04124 ·
  https://asmp-eurasipjournals.springeropen.com/articles/10.1186/s13636-025-00416-3
- Yin et al., *Modeling Analog Dynamic Range Compressors Using Deep Learning and State-space
  Models* — [V, PDF lu et grep : une seule occurrence de « sampling rate »]
  https://arxiv.org/abs/2403.16331

**Convolutions indépendantes de f_s**
- Saito, Nakamura, Yatabe, Koizumi, Saruwatari, *Sampling-Frequency-Independent Audio Source
  Separation Using Convolution Layer Based on Impulse Invariant Method*, EUSIPCO 2021 —
  [V, PDF lu ; résultats chiffrés uniquement en figure] https://arxiv.org/abs/2105.04079
- Nakamura, Nakata, Imamura, Saito, *Neural Audio Codec with Adjustable Token Temporal
  Resolution Using Sampling-Frequency-Independent Convolutional Layers*, IWAENC 2026 —
  [V pour métadonnées et mécanisme ; NV pour les chiffres] https://arxiv.org/abs/2607.01865
- Romero, Kuzina, Bekkers, Tomczak, Hoogendoorn, *CKConv: Continuous Kernel Convolution for
  Sequential Data*, ICLR 2022 — [V, PDF lu, Table 5] https://arxiv.org/abs/2102.02611
- Romero, Bruintjes, Tomczak, Bekkers, Hoogendoorn, van Gemert, *FlexConv: Continuous Kernel
  Convolutions with Differentiable Kernel Sizes*, ICLR 2022 — [NV, résumé de recherche]
  https://arxiv.org/abs/2110.08059 · https://github.com/rjbruin/flexconv

**Conditionnement sur la bande / entraînement multi-fréquence**
- Han, Lee, *NU-Wave 2: A General Neural Audio Upsampling Model for Various Sampling Rates*,
  Interspeech 2022 — [NV, résumé de recherche] https://arxiv.org/abs/2206.08545
- *Interspeech 2025 URGENT Speech Enhancement Challenge* — [NV, résumé de recherche]
  https://arxiv.org/abs/2505.23212 · https://urgent-challenge.github.io/urgent2025/

**Anti-repliement**
- Karras et al., *Alias-Free Generative Adversarial Networks* (StyleGAN3), NeurIPS 2021 —
  [NV, résumé de recherche] https://arxiv.org/abs/2106.12423 ·
  https://github.com/NVlabs/stylegan3
- Vanhatalo, Legrand, Desainte-Catherine, Hanna, Pille, *Evaluation of Real-Time Aliasing
  Reduction Methods in Neural Networks for Nonlinear Audio Effects Modelling*, JAES 72(3),
  pp. 114–122, mars 2024 — [NV, référence relevée dans DAFx25, non lue]
  https://www.aes.org/e-lib/browse.cfm?elib=22415
- Köper, Holters, *Antialiased State Trajectory Neural Networks for Virtual Analog Modeling*,
  DAFx23 — [NV, référence relevée dans DAFx25, non lue]
  https://dafx23.create.aau.dk/wp-content/uploads/2023/09/DAFx23_paper_63.pdf

**Implémentations (pratique, non validée scientifiquement)**
- NAM-Oversampler (« Time-Scaled Dilation Processing ») — [V pour le contenu du README ;
  aucune mesure publiée] https://github.com/DLC86/NAM-Oversampler
- neural-amp-modeler-lv2 (pas de rééchantillonnage interne) — [V pour le README]
  https://github.com/mikeoliphant/neural-amp-modeler-lv2
- NeuralAmpModelerCore — [V, code lu localement dans `third_party/`]
  https://github.com/sdatkinson/NeuralAmpModelerCore
