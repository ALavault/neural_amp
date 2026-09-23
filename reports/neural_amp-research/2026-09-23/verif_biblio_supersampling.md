# Vérification bibliographique : pistes A à F du brainstorming « super-résolution »

2026-09-23. Skill `literature-review`. Vérification ciblée de
`brainstorm_supersampling.md` (même dossier). Aucune expérience du dépôt, pas de GPU.
Fichiers de travail (PDF, txt, clones, script jouet) dans
`/fastdata/lavaulta/tmp/biblio_supersampling/`, hors dépôt.

Hors périmètre, traités par un autre agent : discrétisations SSM (ZOH, FOH, bilinéaire),
conditionnement des pôles, entraînement multi-fréquence.

Marquage : **[V]** = lu dans le texte primaire (ou le code), avec localisation.
**[NV]** = résumé, seconde main ou déduction. **[D]** = déduction de ma part, avec son
support.

## 0. Cadrage

- **Question.** Les pistes A à F sont-elles déjà faites, soutenues ou non documentées ?
- **Inclusion.** Articles, préprints, thèses et code public sur : l'anti-repliement des
  modèles neuronaux d'effets ; le sur-échantillonnage de modèles pré-entraînés ; l'ADAA
  dans des réseaux ; l'extension de bande neuronale ; l'intermodulation ultrasonore.
  Période : 2000 à 2026.
- **Exclusion.** L'image et la vision, sauf comme cadre (StyleGAN3). La parole seule,
  sauf pour l'extension de bande. Les sujets de l'autre agent.
- **Type.** Vérification ciblée, pas une revue systématique.

## 1. Constats principaux

1. **Piste A. La multiplication des dilatations (NAM-Oversampler) préserve déjà la
   réponse dans la bande.** Le brainstorming dit le contraire. Avec des dilatations × M,
   le réseau à M·Fs se sépare en M copies natives, une par phase polyphase. Le code du
   fork l'utilise explicitement **[V, code]**. La phase 0 reproduit donc la sortie native
   à l'échantillon près. Après passe-bas et décimation, on obtient la sortie native
   **moins son repliement**, avec des harmoniques identiques. Un jouet numérique le
   confirme **[D + jouet, §3.1]**. Dans ce même jouet, l'interpolation à bande limitée
   des noyaux (piste A) déplace les harmoniques dans la bande de 0,2 à 0,7 dB. **La
   piste A, telle qu'écrite, fait moins bien que ce qui existe pour préserver le
   modèle.**
2. **Correction de la revue du 22/09.** Carson et al. DAFx25 §4.4 mesurent justement
   la dilatation × 2 sur deux TCN NAM pré-entraînés (Table 1, NMR-S-OS2) **[V]**.
   `revue_sr_independence.md` §3.4 disait « aucune mesure publiée ». Elle disait aussi
   que Carson DAFx25 « ne change pas la fréquence ». Les deux affirmations sont fausses
   pour la comparaison ×2.
3. **Piste B déjà faite en grande partie.** Carson DAFx25 distille un enseignant natif
   (même fréquence, pas sur-échantillonné). La cible est l'enseignant resynthétisé sans
   ses composantes non harmoniques. L'élève tourne lui aussi à la fréquence native.
   Restent non trouvés : un élève à fréquence supérieure et des excitations multi-sons.
4. **Piste F : l'ADAA existe pour des RNN pré-entraînés, mais elle impose une fréquence
   supérieure.** Mikkonen et Werner : l'ADAA d'ordre o place le modèle à (o+1)·f_SR
   **[V]**. Le produit de Hadamard (les portes) est explicitement hors périmètre **[V]**.
   Pour un WaveNet à portes pré-entraîné : non trouvé.

## 2. Requêtes consignées (toutes le 2026-09-23)

Le fichier brut est `queries.log` dans le dossier de travail.

| Source | Requête | Résultats | Retenus |
|---|---|---|---|
| arXiv API | `all:aliasing AND all:neural AND (all:"audio effect" OR all:distortion OR all:amplifier)` | 16 | 3 (2505.11375, 2501.18470, 2505.04082) ; 2406.06293 déjà lu le 22/09 |
| arXiv API | `all:oversampling AND all:neural AND all:guitar` | 4 | 0 nouveau (tous lus le 22/09 ou ici) |
| arXiv API | `all:antiderivative AND all:antialiasing` | 0 | — |
| arXiv API | `all:"sampling frequency independent"` | 10 | 1 lu (2206.02124) ; lignée Saito déjà couverte |
| arXiv API | `all:aliasing AND all:"neural amp"` | 1 | 2505.04082 |
| arXiv API | `all:"alias-free" AND all:audio` | 1 | 0 (oscillateurs) |
| arXiv API | `abs:aliasing AND abs:distillation AND abs:audio` | 0 | — |
| arXiv API | `abs:multisine AND abs:neural AND (abs:nonlinear OR abs:distortion)` | 0 | — |
| arXiv API | `abs:"harmonic" AND abs:"aliasing" AND abs:"loss" AND abs:neural AND abs:audio` | 0 | — |
| arXiv API | `abs:"bandwidth extension" AND (abs:distortion OR abs:"audio effect" OR abs:guitar)` | 11 | 0 (parole, qualité perceptive) |
| arXiv API | `abs:"super-resolution" AND abs:audio AND (abs:guitar OR abs:"audio effect" OR abs:distortion)` | 4 | 1 (2306.11282, SR de musique) |
| arXiv API | `abs:ultrasonic AND abs:intermodulation AND abs:audio` | 1 | 0 (haut-parleur paramétrique) |
| arXiv API | `abs:intermodulation AND abs:"audio effect"` | 0 | — |
| arXiv API | `ti:AudioSR` | 1 | 2309.07314 |
| arXiv API | `abs:"pre-trained" AND abs:"sampling rate" AND abs:convolution AND abs:kernel AND abs:resampl` | 0 | — |
| arXiv API | `abs:"sampling frequency" AND abs:"pretrained" AND abs:"convolutional" AND abs:"without retraining"` | 0 | — |
| arXiv API | `abs:"sample rate" AND abs:"audio effect" AND abs:convolution` | 1 | 0 (réverbération à ressort) |
| DAFx25 | programme détaillé (dafx25.dii.univpm.it/detailed-program), grep aliasing/antiderivative/oversampling/upsampling | 70 articles liés, 6 titres pertinents | 4 lus (papers 31, 50, 61, 72) ; BBD/BLEP (29) et LUT-ADAA (30) exclus : non linéarités statiques ou puces BBD, pas des réseaux |
| Archive DAFx | recherche « Antialiased state trajectory » | 1 | DAFx23 paper 53 (Köper et Holters) |
| HAL API | `authLastName_s:Vanhatalo` | 15 | article JAES sans fichier ; thèse tel-04921220 lue (ch. 6) |
| Crossref | « Is High-Frequency Intermodulation Distortion a Significant Factor in High-Resolution Audio » | 3 | 1 (Stuart et al. 2019), résumé inaccessible |
| Semantic Scholar | recherche et citations de arXiv:2505.11375 et 2505.04082 | HTTP 429, trois fois | **chaînage avant non fait** |
| OpenAlex | citations de Carson DAFx25 | cited_by = 0 | jugé non fiable (article cité par Mikkonen et Werner) |
| WebSearch | intermodulation ultrasonore, chaînes de pédales | 9 | 0 citable (forums) ; piste vers Ashihara |
| WebSearch | Ashihara Kiryu, audibilité au-dessus de 22 kHz | 9 | Ashihara 2000 (PDF lu) ; Stuart 2019 [NV] |
| WebSearch | interpolation des noyaux d'un WaveNet pré-entraîné | 9 | 0 publication ; NAM-Oversampler (déjà connu) |
| WebSearch | effets non linéaires en cascade, DAFx | 9 | 0 nouveau |
| WebSearch | anti-repliement NAM 2026 | 9 | 1 (2512.20211, Pupu-Vocoder) |
| WebSearch | rééchantillonnage des noyaux d'un CNN pré-entraîné | 9 | 0 publication |
| GitHub | clones `a-carson/dafx25_antialiasing_neural` (468365e), `DLC86/NAM-Oversampler` (7d8df1b), sous-module `DLC86/NeuralAmpModelerCore` branche `nam-oversampler-time-scale` (bea723b) | — | code lu |

**Tri.** Environ 90 titres vus. 16 textes intégraux ou codes lus : 13 PDF et 3 dépôts de
code. Les résumés de WebSearch n'ont servi qu'à trouver des sources. Trois éléments
proposés par ces résumés ne sont **pas** cités, faute de texte primaire : une « méthode »
générique de rééchantillonnage de noyaux, les affirmations des forums sur
l'intermodulation, et arXiv 2601.14684.

## 3. Synthèse par question

### Q1 / piste A : enseignant de Carson, et interpolation des noyaux

**Carson, Wright, Bilbao, DAFx25 (arXiv:2505.11375v1).**
- **L'enseignant n'est pas sur-échantillonné** **[V, §2 et Fig. 1]**. C'est le modèle
  pré-entraîné lui-même, poids gelés, à sa fréquence native. Fs vaut 44,1 kHz par défaut.
  Les TCN NAM sont réglés à 48 kHz, « fine-tuning was implemented at this same rate »
  (§3.1). L'élève est une copie qui tourne à la même fréquence **[V, éq. 2]**.
- **La cible** : des sinus aléatoires passent dans l'enseignant **[V, §2.1]** : A ∼ U(0,1),
  φ ∼ U(0,2π), note MIDI l ∼ U(21,127), soit jusqu'à environ 12,5 kHz. On retire le
  transitoire. On applique une fenêtre de Chebyshev à 120 dB. On extrait les harmoniques
  avec correction du scalloping. On resynthétise par Fourier la partie harmonique plus
  le DC **[V, §2.2, éq. 7–12]**. Les bins non harmoniques sont donc **mis à zéro** dans
  la cible. La perte les pénalise, elle ne les ignore pas.
- **Perte** : L = ESR + λ·NMR avec λ = 1 **[V, éq. 17]**. Les deux signaux passent
  d'abord par un passe-bas (bande passante 12 kHz, coupure 16 kHz, 80 dB) **[V, §2.3]**.
  Paramètres de la NMR : N_FFT = 2048, recouvrement 50 %, 109 bandes critiques.
- **Modèles** : 2 LSTM GuitarML (Mesa, Goat), 2 TCN NAM (Vox, JCM) et 4 modèles
  entraînés par les auteurs (Broadcast et JHM8, en LSTM et en TCN) **[V, §3]**.
- **Chiffres** (dB, plus bas = mieux ; moyenne sur les f0 et les niveaux des sinus)
  **[V, Table 1]** :

| Modèle | NMR-S enseignant | NMR-S élève | NMR-S enseignant ×2 | ESR G+B élève (réf.) |
|---|---|---|---|---|
| Broadcast LSTM | 6,1 | −16,7 | −3,4 | −12,1 (enseignant −15,4) |
| Broadcast TCN | 6,5 | −18,2 | −2,1 | −7,0 (enseignant −12,1) |
| JHM8 LSTM | −8,9 | −22,7 | −24,8 | −20,1 (enseignant −24,1) |
| JHM8 TCN | −1,0 | −25,1 | −11,8 | −13,5 (enseignant −19,9) |
| Goat LSTM | 11,5 | −5,6 | 0,4 | −11,1 (réf. = enseignant) |
| Mesa LSTM | 19,3 | −14,8 | 4,2 | −12,7 (réf. = enseignant) |
| Vox TCN | 15,0 | −21,1 | 6,5 | −8,1 (réf. = enseignant) |
| JCM TCN | 13,4 | −12,5 | 3,0 | −10,0 (réf. = enseignant) |

  - L'élève bat le ×2 sauf pour JHM8 LSTM **[V, §4.4]**.
  - Le prix : l'ESR sur guitare et basse augmente dans tous les cas. Les harmoniques
    au-dessus de 12 kHz sont amorties, et c'est voulu **[V, §4.3]**. Au MUSHRA, l'élève
    TCN est jugé « low-passed » et passe de Excellent à Very Good **[V, §5]**.
- **Comment Carson sur-échantillonne un TCN** (comparaison ×2) : « by upsampling the
  convolutional kernels, i.e. increasing the convolution dilation of each layer by a
  factor of 2 » **[V, §4.4]**. C'est l'insertion de zéros (dilatation × 2), pas une
  interpolation à bande limitée. Les signaux d'entrée et de sortie sont rééchantillonnés
  dans le domaine fréquentiel.

**NAM-Oversampler (DLC86), lu dans le code.**
- `Conv1D::SetDilationScale` : `_dilation = _base_dilation * _dilation_scale`
  (`NAM/conv1d.cpp` l. 117–122, sous-module bea723b) **[V]**.
- `WaveNet::SetTimeScale` applique ce facteur aux couches, au conditionnement et à la
  tête (`NAM/wavenet/model.cpp` l. 663–676). Le chemin rapide A2 fait de même
  (`a2_fast.cpp` l. 346–365) **[V]**.
- **Aucune interpolation de noyau.** Les poids ne sont pas modifiés **[V]**.
- Le LSTM n'a pas de `SetTimeScale` propre. La valeur par défaut est une no-op
  (`NAM/dsp.h` l. 174 ; aucune occurrence dans `lstm.*`) **[V]**. Ce que le plugin fait
  d'un LSTM sur-échantillonné : **non vérifié**.
- Mode multi-cœur : `PolyphaseOversampledDSP` « decomposes an integer-oversampled
  WaveNet stream into independent native-rate phases ». Il fait tourner S copies aux
  dilatations natives (`NAM/polyphase.h` ; README) **[V]**.
- Rééchantillonnage : cascade de demi-bandes, avec les modes phase minimale, linéaire
  courte et linéaire longue (README) **[V pour le README]**.
- `run_null_tests.py` compare `input.wav` à `test_out.wav` (l. 538). C'est un
  aller-retour du rééchantillonneur. `grid_search_results.csv` règle ses filtres.
  **Aucune mesure du modèle** ne soutient « Exact Tone Preservation » **[V]**.
- Commits : dépôt 7d8df1b (2026-07-30), sous-module bea723b (2026-06-20).

#### 3.1 Déduction et jouet : la dilatation × M n'est pas ce que le brainstorming croit

**[D]** Avec des dilatations × M et des opérations sinon ponctuelles, chaque phase
polyphase ne voit que ses propres échantillons. C'est vrai pour les convolutions 1×1,
tanh, sigmoïde et le produit des portes. Le mode polyphase de NAM-Oversampler repose
exactement sur cette propriété **[V]**. Avec un suréchantillonneur de Nyquist, la phase 0
est donc la sortie native. Le signal à M·Fs est l'échantillonnage d'un réseau « à lignes
à retard » en temps continu. Les ultrasons y traversent les couches suivantes par la
réponse périodique H(z^M). Leur module est celui que le modèle natif applique à leurs
repliements. Un passe-bas suivi d'une décimation donne la sortie native **moins ses
composantes repliées**. Les harmoniques dans la bande restent identiques. La seule
limite est le repliement résiduel au-dessus de M·Fs/2. C'est ce que montrent les
NMR-S-OS2 encore positives de Vox et JCM à ×2.

**Jouet** (`toy.py`, CPU, FFT idéales, signaux périodiques) : trois couches dilatées
(1, 2, 4), tanh, poids aléatoires (graine 0), sinus d'amplitude 3, M = 4, f0 = k0·10 Hz
avec k0 premier avec N = 4800.
- Phase 0 contre natif : écart maximal 1,3·10⁻¹⁵.
- Harmoniques dans la bande au-dessus de −60 dB relatifs, écart maximal au natif :

| f0 | dilatation × 4 | noyaux interpolés (piste A) |
|---|---|---|
| 970 Hz | 0,000 dB | 0,000 dB |
| 2110 Hz | 0,000 dB | 0,207 dB |
| 4310 Hz | 0,000 dB | 0,485 dB |
| 7970 Hz | 0,000 dB | 0,737 dB |

- Rapport énergie non harmonique / harmonique à 4310 Hz : natif 8,1·10⁻⁴ ;
  dilatation × 4 : 7·10⁻¹⁶ ; piste A : 1·10⁻¹⁸.

**Limites du jouet.** Réseau jouet sans résidu, sans portes et sans données. Filtres
idéaux. Sinus stationnaire. Il établit le mécanisme, pas l'ampleur sur un vrai `.nam`.

**Autres travaux proches (Q1, noyaux interpolés) :**
- Paulus et Torcoli, « Sampling Frequency Independent Dialogue Separation »
  (arXiv:2206.02124) **[V, §2–3, grep]**. Le transfert de 8 à 48 kHz passe par un
  encodeur STFT dont la trame suit la fréquence. Ce n'est pas une interpolation de
  noyaux convolutifs appris.
- Saito et al. (SFI, 2021) et CKConv ont été lus le 22/09. Ce sont des filtres continus
  appris **à l'entraînement**. Rien ne convertit un modèle déjà entraîné.

**Conséquence piste A :** **non documentée** sous sa forme (interpolation à bande limitée
des noyaux d'un CNN pré-entraîné). Mais **dominée** par la dilatation × M, **déjà faite**
(Carson DAFx25 §4.4 à ×2, mesurée ; NAM-Oversampler jusqu'à ×32, non mesuré).
La dilatation × M préserve exactement les harmoniques de l'enseignant [D + jouet]. La
piste A les modifie, parce qu'elle coupe entre les couches les ultrasons que le modèle
natif « voit » sous forme repliée. L'hypothèse fragile de la piste A (« le modèle a
appris à compenser son repliement ») prend donc un sens précis : ce qu'apprend le réseau
natif, c'est justement la version à lignes à retard, et c'est elle que la dilatation × M
restitue.

### Q2 / piste B : distillation masquée et métriques de repliement

- **Carson DAFx25** : voir Q1. Excitation : un seul sinus par exemple, pas de multi-sons
  **[V, §2.1]**.
- **Métriques. Elles ne sont pas comparables entre articles.**
  - **NMR** (Carson DAFx25) : rapport bruit/masque par bandes critiques, calculé entre
    la STFT du signal et celle de sa référence harmonique (PEAQ / Kabal). En dB.
    Audibilité approximative à −10 dB. NMR-S = signal contre sa propre version à bande
    limitée **[V, éq. 14–16, §2.6]**.
  - **ASR** (Sato et Smith, DAFx25, arXiv:2505.04082v2) : E_A/E_H, avec
    E_A = E_Y − E_H **[V, §3.2]**. E_H somme |Y(m·k0)|² sur les harmoniques. DFT sans
    fenêtre ni zéros, N premier (48 017, une seconde à fs = 48 017 Hz), f0 = 1249 Hz.
    Rapport **linéaire**, pas en dB. Un seul sinus. L'amplitude du sinus n'est **pas
    trouvée dans le texte**. Le modèle entraîné à 48 kHz est évalué à 48 017 Hz.
  - **SNRA** (Mikkonen et Werner, après Carson DAFx24) : Σ|Y_bl|² / Σ(|Y′| − |Y_bl|)²
    sur les bins jusqu'à l'ancienne Nyquist **[V, éq. 67]**. Sinus de A0 à C8. Niveau
    réglé par modèle (−24 et −12 dBFS), parce que l'analyse-resynthèse « s'effondre »
    sinon **[V, §4.2]**.
  - **Vanhatalo** (thèse HAL tel-04921220, ch. 6, reprenant JAES 2024) : pas de métrique
    de repliement. MAE sur un sinus de test à 1244,5 Hz, plus inspection des spectres.
    L'auteur réclame une mesure objective et différentiable du repliement **[V, §6.7,
    conclusion]**. L'article JAES lui-même n'a pas été lu.
  - **AHR** (Pupu-Vocoder, arXiv:2512.20211) : « aliasing-to-harmonic ratio as proposed
    in [97] », définition « qui correspond à l'ASR » **[V pour la phrase]**. La
    référence [97] n'a pas été résolue **[NV]**.
- **Vanhatalo, entraîner sur données sans repliement ne suffit pas** : données du clipper
  à diode simulées dans LTspice à 705,6 kHz puis décimées à 22,05 kHz. « none of the
  models trained on anti-aliased data showed reduced amounts of aliasing » **[V, thèse
  §6.7.1]**. Le repliement vient du calcul, pas des données. Cela soutient le §0 du
  brainstorming.

**Conséquence piste B :** **déjà faite par Carson DAFx25** pour la variante « même
fréquence, sinus seuls, bins non harmoniques retirés de la cible ». **Non trouvés** : un
élève à fréquence supérieure et des excitations multi-sons. [D] La dilatation × M avec M
grand fournit un enseignant sans repliement pour **n'importe quelle** excitation, y
compris la musique, sans resynthèse de Fourier (§3.1). Cela change ce que la piste B
aurait à apporter.

### Q3 / piste F : ADAA dans les réseaux

- **Mikkonen et Werner, DAFx25 paper 72 (Soundtoys).**
  - ADAA d'ordre 1 et 2 sur les activations σ et tanh d'un GRU et d'un LSTM
    **pré-entraînés** (Blackstar HT-1, Mesa Mini Rectifier), sans réentraînement
    **[V, §4.1, conclusion]**.
  - Des filtres de synchronisation Ho alignent les signaux internes **[V, §3, Table 1]**.
  - Le retard ajouté oblige à tourner à f′ = (o+1)·f_SR. Le délai d'état passe à
    o + M_OS échantillons **[V, §3.2–3.3, éq. 65–66]**.
  - « this work focuses solely on antialiasing the activations and leaves explicit
    consideration of the Hadamard product out of scope » **[V, §3]**.
  - Gains de SNRA : environ 20 à 30 dB à 88,2 kHz, environ 20 et 40 dB à 176,4 kHz
    (ordres 1 et 2) **[V, §5.1]**.
  - Prix : écarts sur les harmoniques et effet passe-bas vers 20 kHz. Meilleur compromis
    trouvé : ordre 1, sur-échantillonné en plus d'un facteur 3 (4× la fréquence
    d'origine) **[V, §5.2, conclusion]**.
- **Esqueda et Murai (KORG), DAFx25 paper 61.**
  - Réseau de LRU réelles (SSM linéaires) intercalées avec des étages ponctuels
    « activation puis couche FC » **[V, §3]**.
  - Les seuils de mauvais conditionnement de l'ADAA-tanh sont introuvables dans un
    réseau : « we were unable to find suitable ill-conditioning thresholds » **[V, §4]**.
  - Solution : entraîner avec f(x) = x/√(1+x²). Sa forme ADAA d'ordre 1,
    (x_n + x_{n−1})/(√(1+x_n²) + √(1+x_{n−1}²)), n'est jamais mal conditionnée. On la
    substitue à l'inférence **[V, éq. 11–14]**.
  - Le demi-échantillon de retard est compensé dans la branche directe par une moyenne
    à deux points **[V, §4]**.
  - Mesures faites à 96 kHz. Aucune métrique chiffrée : spectrogrammes et écoute
    informelle **[V, §5.3, §6.2]**.
- **Köper et Holters, DAFx23 paper 53.** L'ADAA passe par les **données**, pas par le
  réseau.
  - Simulation ACME sur-échantillonnée (fs1 = 96 kHz, L = 4), entrée interpolée
    linéairement, sortie moyennée sur un intervalle, états sous-échantillonnés sans
    filtre **[V, §2–3]**.
  - Le STN reçoit en plus u(n−1).
  - Il faut les états internes du circuit, donc des données synthétiques **[V]**.
  - Passer de L = 4 à L = 8 n'apporte presque rien **[V, §5]**.
- **Gu et al., Pupu-Vocoder (arXiv:2512.20211v3, 2026).** Sur-échantillonnage plus ADAA
  sur l'activation SnakeBeta, **à l'entraînement**, dans un vocodeur. SnakeBeta est
  choisie pour que la forme ADAA n'ait pas de dénominateur **[V, §III]**. Ce n'est pas
  un modèle d'effet.
- **Vanhatalo, thèse §6.7.** Sur-échantillonner ×2 les activations d'un WaveNet (façon
  StyleGAN3) pendant l'entraînement réduit peu le repliement, et pas à chaque graine.
  ×16 ne fait pas mieux que ×2 **[V]**. Les filtres sont les couches up/down de
  PyTorch. Leur qualité n'est pas précisée, ce qui peut expliquer le résultat [NV].

**Conséquence piste F :** **soutenue et partiellement faite.** L'ADAA d'un réseau
pré-entraîné est faite pour les RNN (Mikkonen et Werner). Le problème du demi-échantillon
prévu par le brainstorming est **confirmé** : il est résolu en montant la fréquence
((o+1)·f) ou en compensant la branche directe (Esqueda). Pour les activations à portes
(tanh·σ) d'un WaveNet pré-entraîné : **non trouvé**. Le seul travail qui touche aux
portes les laisse hors périmètre. [D] Pour notre SSM (couches linéaires plus étages
ponctuels), Esqueda et Murai sont le précédent direct. Leur leçon : choisir
l'activation **à l'entraînement** pour que l'ADAA soit sûre.

### Q4 / pistes C et D : extension de bande et intermodulation ultrasonore

- **NU-Wave 2** (arXiv:2206.08545v2). Désormais **[V]** ; c'était [NV] le 22/09.
  - Diffusion. Sortie toujours à 48 kHz. Entrée de 6 à 48 kHz.
  - Conditionnement par un plongement one-hot qui marque les bandes venant de l'entrée,
    via la BSFT **[V, §3 et résumé]**.
  - Entraînement sur VCTK, de la parole seule **[V, §4.1]**.
- **AudioSR** (arXiv:2309.07314v1). Bande d'entrée de 2 à 16 kHz vers une bande de
  24 kHz à 48 kHz. Parole, musique et effets sonores. Utilisé en post-traitement de
  sorties génératives **[V, résumé, §1, §5]**.
- **SR de musique** (arXiv:2306.11282) : piano, cible à 16 kHz **[V, §4]**.
- **Aucun des trois ne va au-delà de 48 kHz, ni ne traite des sorties d'effets non
  linéaires.**
- **Intermodulation ultrasonore.**
  - Ashihara, Inter-noise 2000 **[V, résumé, §2]**. Son complexe à f0 = 2 kHz,
    harmoniques impairs jusqu'à 38 kHz. Les sujets détectent les composantes
    ultrasonores seulement sur un haut-parleur unique. Réparties sur six haut-parleurs,
    ce qui réduit l'intermodulation, elles deviennent indétectables. C'est le mécanisme
    de la piste D : une non-linéarité en aval rend audibles des ultrasons. Mais l'étage
    est un **haut-parleur**, pas un effet.
  - Stuart, Hollinshead, Capp, « Is High-Frequency Intermodulation Distortion a
    Significant Factor in High-Resolution Audio? », JAES 67(5), 2019 **[NV,
    métadonnées Crossref seulement]**.
  - Pour des **chaînes d'effets** (pédale puis ampli, étages numériques en cascade) :
    **non trouvé**. Seuls des forums en parlent, et ils ne sont pas cités.

**Conséquence piste C :** **non documentée.** L'extension de bande publiée s'arrête à
48 kHz et vise la plausibilité perceptive de la parole et de la musique. Elle ne
prolonge pas des harmoniques de distorsion au-delà de 24 kHz.

**Conséquence piste D :** **non documentée pour les effets.** Le mécanisme est
**soutenu** en reproduction sonore (Ashihara 2000). Aucune mesure publiée sur une chaîne
d'effets n'a été trouvée.

### Piste E (corpus synthétique à 192 kHz)

Des précédents génèrent des données de vérité sur-échantillonnées par simulation :
Vanhatalo (LTspice, 705,6 kHz) **[V]** ; Köper et Holters (ACME, 384 kHz) **[V]** ;
Carson (sinus dans le modèle, resynthèse). **Conséquence : soutenue** comme méthode.
Son usage pour tester A à D n'a pas été trouvé.

## 4. Non trouvé

Absence dans une recherche bornée, **pas** preuve de nouveauté.

- Interpolation à bande limitée des noyaux d'un CNN audio **pré-entraîné** pour
  sur-échantillonner. Non trouvé dans : les requêtes arXiv ci-dessus (3 requêtes dédiées,
  0 résultat), 2 WebSearch, le programme DAFx25 et le code de NAM-Oversampler.
- Distillation d'un élève à fréquence **supérieure**, ou avec des excitations
  **multi-sons** à bins traçables. Non trouvé dans : arXiv (3 requêtes, 0 résultat), le
  programme DAFx25 et les références de Carson DAFx25.
- ADAA sur les activations **à portes** (tanh·σ) d'un WaveNet ou TCN pré-entraîné. Non
  trouvé dans : arXiv, le programme DAFx25, Mikkonen et Werner (hors périmètre explicite)
  et Esqueda et Murai.
- Extension de bande au-dessus de 24 kHz, ou appliquée à des sorties d'effets non
  linéaires. Non trouvé dans : arXiv (3 requêtes), NU-Wave 2 et AudioSR.
- Mesure de l'intermodulation audible d'ultrasons dans une **chaîne d'effets audio**.
  Non trouvé dans : arXiv (2 requêtes), 2 WebSearch et Crossref.
- **Chaînage avant** (qui cite Carson DAFx25 ou Sato 2025) : **non fait**. Semantic
  Scholar a répondu 429 trois fois. OpenAlex n'est pas fiable ici. Des travaux de 2026
  ont pu échapper.

## 5. Correction à la revue du 22/09 (fichier non modifié)

`2026-09-22/revue_sr_independence.md` :
- §3.4 dit que la dilatation × M n'a « aucune mesure publiée ». Faux : Carson DAFx25
  §4.4 et Table 1 (colonnes ESR-S-OS2 et NMR-S-OS2) la mesurent à ×2 sur Vox et JCM
  (NAM) et sur les TCN Broadcast et JHM8.
- §3.4 dit que Carson DAFx25 « ne change pas la fréquence ». Vrai pour la méthode
  proposée, faux pour sa comparaison.
- §2.5 marquait NU-Wave 2 [NV]. C'est maintenant [V] (§3 ci-dessus).
- §4.5 marquait Vanhatalo et Köper [NV]. C'est maintenant [V] (thèse ch. 6 pour
  Vanhatalo ; DAFx23 paper 53 pour Köper).

## 6. Références

- Carson, Wright, Bilbao, « Anti-aliasing of neural distortion effects via model fine
  tuning », DAFx25 — [V, PDF lu] https://arxiv.org/abs/2505.11375 ·
  https://dafx.de/paper-archive/2025/DAFx25_paper_31.pdf · code :
  https://github.com/a-carson/dafx25_antialiasing_neural
- Sato, Smith, « Aliasing Reduction in Neural Amp Modeling by Smoothing Activations »,
  DAFx25 — [V, §3] https://arxiv.org/abs/2505.04082
- Mikkonen, Werner, « Antiderivative Antialiasing for Recurrent Neural Networks », DAFx25
  — [V] https://dafx25.dii.univpm.it/wp-content/uploads/2025/09/DAFx25_paper_72.pdf
- Esqueda, Murai, « Antialiased Black-Box Modeling of Audio Distortion Circuits Using
  Real Linear Recurrent Units », DAFx25 — [V]
  https://dafx25.dii.univpm.it/wp-content/uploads/2025/09/DAFx25_paper_61.pdf
- Köper, Holters, « Antialiased State Trajectory Neural Networks for Virtual Analog
  Modeling », DAFx23 — [V] https://www.dafx.de/paper-archive/2023/DAFx23_paper_53.pdf
- Vanhatalo, Legrand, Desainte-Catherine, Hanna, Pille, « Evaluation of Real-Time
  Aliasing Reduction Methods in Neural Networks for Nonlinear Audio Effects Modelling »,
  JAES 72(3):114–122, 2024 — [NV pour l'article ; contenu lu dans la thèse]
  https://doi.org/10.17743/jaes.2022.0122 · https://inria.hal.science/hal-04235385
- Vanhatalo, « Real-time simulation of nonlinear audio effects using artificial
  intelligence », thèse, 2024 — [V, ch. 6]
  https://inria.hal.science/tel-04921220
- Carson, Wright, Chowdhury, Välimäki, Bilbao, « Sample Rate Independent Recurrent
  Neural Networks for Audio Effects Processing », DAFx24 — [V le 22/09]
  https://arxiv.org/abs/2406.06293
- Carson, Välimäki, Wright, Bilbao, « Resampling Filter Design for Multirate Neural
  Audio Effect Processing » — [V le 22/09] https://arxiv.org/abs/2501.18470
- Gu, Zhang, Wang, Li, Wu, Juvela, « Aliasing-Free Neural Audio Synthesis », 2026 —
  [V, §III et §IV-B] https://arxiv.org/abs/2512.20211
- Paulus, Torcoli, « Sampling Frequency Independent Dialogue Separation » — [V]
  https://arxiv.org/abs/2206.02124
- Han, Lee, « NU-Wave 2: A General Neural Audio Upsampling Model for Various Sampling
  Rates », Interspeech 2022 — [V] https://arxiv.org/abs/2206.08545
- Liu, Chen, Tian, Wang, Plumbley, « AudioSR: Versatile Audio Super-resolution at
  Scale » — [V] https://arxiv.org/abs/2309.07314
- Zhang, Kolkman, Watanabe, « Phase Repair for Time-Domain Convolutional Neural Networks
  in Music Super-Resolution » — [V, §4] https://arxiv.org/abs/2306.11282
- Ashihara, « Audibility of complex tones above 20 kHz », Inter-noise 2000 — [V]
  http://www.conforg.fr/internoise2000/cdrom/data/articles/000477.pdf
- Stuart, Hollinshead, Capp, « Is High-Frequency Intermodulation Distortion a
  Significant Factor in High-Resolution Audio? », JAES 67(5):310–318, 2019 — [NV]
  https://doi.org/10.17743/jaes.2018.0060
- NAM-Oversampler (DLC86), commit 7d8df1b, et sous-module NeuralAmpModelerCore bea723b
  — [V, code] https://github.com/DLC86/NAM-Oversampler ·
  https://github.com/DLC86/NeuralAmpModelerCore/tree/nam-oversampler-time-scale
