# Vérification bibliographique : pistes A, C, D, F

2026-09-23. Vérification ciblée des pistes A, C, D et F de `brainstorm_sr_param.md` (même
dossier). Ce qui est déjà vérifié dans `../2026-09-22/revue_sr_independence.md` (« revue du
22/09 ») est cité, pas refait. Skill `literature-review`.

Convention : **[V]** = lu dans le texte primaire (PDF extrait par `pdftotext`, ou code
source lu), avec la section, l'équation, la table ou la ligne. **[NV]** = résumé de
moteur de recherche, citation de seconde main, ou **déduction** de ma part (dit comme tel).
Les PDF et les textes extraits sont dans `/fastdata/lavaulta/tmp/biblio_0923/`, hors du dépôt.

## 0. Cadrage

- **Question.** Pour chaque piste, la littérature contient-elle déjà (a) le mécanisme,
  (b) une mesure du transfert de fréquence qui le teste ?
- **Critères** (fixés en partie après les premières recherches, je le signale) : SSM,
  réseaux à temps continu, neural operators, modèles grey-box d'effets audio ; toutes
  années ; préprints compris ; hors audio accepté pour A, C et D.
- **Limite de couverture.** L'API Semantic Scholar a répondu 429 (limite de débit) à
  4 requêtes sur 5 ; le chaînage des citations n'a donc pas été fait. Le texte de la
  page MathWorks `d2d` n'a pas pu être récupéré (page vide de 430 octets).

## 1. En bref

1. **Piste A (maintien appris, correction « hold-matched »).** Aucune correction qui
   garde la porte de la fréquence d'entraînement n'a été trouvée. Le plus proche est
   Mamba-3 : un maintien *appris* à pas fixe, sans test de transfert.
2. **Piste C (ZOH contre bilinéaire).** Le « bilinéaire » de S4 n'applique le trapèze
   qu'à la matrice d'état. L'entrée n'y entre que par u_k. Il garde donc le même
   décalage d'un demi-échantillon que le ZOH (déduction, vérifiée numériquement au
   §2.3). Cela explique que S4D ne trouve aucune différence entre les deux règles
   (« no noticeable difference », dans la distribution seulement). Seule une règle
   symétrique sur l'entrée devrait annuler le décalage. Aucune n'a été testée en
   transfert. Le seul appui chiffré sur l'ordre du schéma est Wilczek (DAFx22, déjà lu
   le 22/09), avec des neural ODE, et l'effet n'y est pas monotone.
3. **FNO de Massi et al. (DAFx26).** Le papier est trouvé et lu. C'est un opérateur par
   trames de 20 ms, sans état. Le transfert y est mesuré contre la sortie du simulateur
   à 48 kHz rééchantillonnée : c'est de la cohérence, pas de la fidélité.
4. **Piste F (réglages sur les pôles).** C'est déjà fait en grey-box : réglage →
   résistance → matrices du circuit → discrétisation trapézoïdale avec T explicite
   (Esqueda 2021 ; Miklánek 2023). Dans les filtres différentiables, les réglages
   agissent sur des pôles ou coefficients discrets (tone stack 2024, Nercessian 2021). En
   boîte noire, ils agissent sur la matrice récurrente (HyperRNN) ou sur les activations
   après le SSM (nablafx, Simionato). **Non trouvé :** un SSM profond dont les réglages modulent Λ
   ou Δ. TIDES (2026) montre, hors audio et sans réglages, qu'une modulation qui porte
   sur Λ et non sur Δ extrapole à des Δ non vus.
5. **Piste D (entraînement multi-Δ).** Documenté hors audio : TAWM, TIDES, S5 §6.3,
   progressive resizing de S4ND. Les chiffres de transfert sont surtout en figure, et
   aucun ne sépare l'effet du multi-Δ. Pour les effets audio : non trouvé.

## 2. Q1 : discrétisations (pistes A et C)

### 2.1 Quelle règle chaque travail utilise

| Travail | Règle | Localisation |
|---|---|---|
| S4 (Gu, Goel, Ré, ICLR 2022) | bilinéaire (Tustin) | [V] §2.3, éq. (3) |
| S4D (Gu, Gupta, Goel, Ré, NeurIPS 2022) | les deux sont présentées ; listing 1 en bilinéaire | [V] §3.1, listing 1 |
| Code `state-spaces` (copie dans `third_party/nablafx/tests/.../ineffab1evista_s4drc/s4.py`) | `disc='zoh'` par défaut, commentaire « Change to 'bilinear' to match S4, but should make little difference either way » ; argument `rate` qui multiplie dt | [V] code, l. 994 et 1089 |
| S5 (Smith et al., ICLR 2023) | ZOH | [V] éq. (6) |
| Mamba (Gu, Dao 2023) | ZOH annoncé | [V] éq. (4) |
| Mamba-1/2, code publié | « exponential-Euler », B̄ = ΔB | [V] Mamba-3, §3.1, Table 1 et note 3 |
| Mamba-3 (Lahoti et al., 2026) | « exponential-trapezoidal », λ_t appris | [V] §3.1, Prop. 1 |
| Piano-SSM (Dallinger et al., DAFx25) | ZOH ; T_s = f_train/f_synth | [V] §3.1, éq. (2) |
| Simionato & Fasciani (arXiv 2405.04124) | **non précisée dans le texte** (S4D et S6) | [V] grep « discreti/zoh/bilinear » : aucune règle |
| Code compagnon de Simionato & Fasciani 2408.12549 (`third_party/optical-selective-ssm`, commit bd787ec) | ZOH : `dB = B * expm1(dA) / Lambda` | [V] `src/Code/S4D.py`, l. 115-117 |
| nablafx, bloc S4 (`third_party/nablafx`, commit 045db6e) | ZOH : `C * (exp(dtA) - 1) / A` | [V] `nablafx/processors/blocks.py`, `get_kernel`, l. 295-305 |
| SSM pour caméras événementielles (Zubić et al., CVPR 2024) | S5 en bilinéaire | [V] annexe, listing `discretize_bilinear` |
| Plaques (Bittner et al., DAFx26) | ZOH | [V] §2, l. 235 du texte extrait |

### 2.2 Le choix de discrétisation a-t-il été étudié en transfert de fréquence ?

- **S4D ne compare les règles que dans la distribution d'entraînement.** « Different
  discretizations … do not make a noticeable difference » [V, §5.1 (iii), Table 2 et
  Table 8 : sCIFAR, SC en AR, BIDMC] ; « the bilinear vs. ZOH discretizations make no
  difference » sur Path-X [V, annexe, Table 6]. La Table 11 donne bien un transfert
  zéro-shot 16 → 8 kHz, mais la règle de chaque ligne n'y est pas indiquée, les
  paramétrisations diffèrent (DPLR contre diagonale), et c'est seulement une descente
  [V]. Elle ne peut donc pas servir de comparaison ZOH contre bilinéaire.
- **S4D signale un effet de la règle sur le noyau.** « the bilinear transform actually
  perfectly smooths out the kernel used in DSS to match the S4 kernel » [V, §3.1 et
  Fig. 2d]. Les auteurs ajoutent que la règle du trapèze réduit l'erreur d'intégration
  [V, §3.1]. Ils ne testent pas cet effet sur un autre pas.
- **« Beyond ZOH » (Ibrahim et al., 2026, Vision Mamba).** Six règles comparées : ZOH,
  FOH, bilinéaire, polynomiale, maintien d'ordre supérieur, RK4. Uniquement à résolution
  fixe [V, résumé, §3 ; grep « resolution » : pas d'expérience de transfert].
- **Seule mesure trouvée de l'ordre du schéma sur le transfert.** Wilczek et al.
  (DAFx22), déjà lu [V] dans la revue du 22/09, §2.3, Tables 2 et 4. Les neural ODE sont
  entraînées à 44,1 kHz et intégrées par Euler avant (FE), IA, trapèze (TR) ou RK4. Le
  résultat n'est pas monotone en ordre (SDR) :

  | modèle | 44,1 kHz | 192 kHz |
  |---|---|---|
  | ODENet30-RK4 | 15,7 | 15,7 (stable) |
  | ODENet30-TR | 15,5 | 13,9 (−1,6 dB) |
  | ODENet30-FE | 11,6 | 11,0 |

  En Table 2, FE et IA *gagnent* à 192 kHz. Ce sont des neural ODE non linéaires, pas
  un SSM linéaire suivi de non-linéarités ponctuelles.
- **Discrétisation sans maintien.** L'invariance impulsionnelle de Saito et al. (revue
  du 22/09, §2.6) échantillonne directement la réponse continue, sans porte [V, 22/09].
- **Données de montée pour un SSM, hors audio.**
  - Caméras événementielles, S5 en bilinéaire : mAP 47,71 à 20 Hz, puis 46,44 / 45,08 /
    42,49 / 39,84 à 40 / 80 / 100 / 200 Hz sur Gen1 ; perte moyenne 4,25 mAP (3,27 sur
    1Mpx) [V, Table 2]. Contre 26,14 mAP pour RVT. Le rôle de la règle n'est pas isolé.
  - PDM (Boulanger & Wood, 2026) : encodeur SSM LegT, avec un classifieur entraîné
    sur la parole à 16 kHz ; je n'ai pas établi si l'encodeur lui-même est entraîné. Il
    est appliqué avec Δ rééchelonné à des flux PDM de 128 kHz à 2 MHz. Seul l'encodeur
    tourne à la fréquence PDM : sa sortie est décimée d'un facteur égal à l'OSR
    [V, légende Fig. 4]. « we maintain over 92%
    classification accuracy for PDM OSRs sampling rates above 512 kHz. This represents
    less than a 4% drop » par rapport à 93,53 % [V, §V et légende Fig. 10]. Le texte
    cite le bilinéaire comme exemple de règle ; je n'ai pas lu laquelle est utilisée [NV].
    La perte inclut le bruit de quantification du PDM. C'est une montée d'un facteur
    32 à 128 sans entraînement multi-fréquence.
- **Piège relevé.** Solozabal et al. (2025) : S4D-Batched-DFouT passe de 95,31 % à
  16 kHz à **0,07 %** à 8 kHz, alors que S4D-DFouT tient 93,85 % [V, Table 2]. C'est
  sous le hasard (environ 2,9 % pour 35 classes). La cause n'est pas expliquée dans les
  passages lus. Les deux variantes sont paramétrées en domaine discret, donc ce n'est
  pas la paramétrisation seule.

### 2.3 Maintien appris, retard de dt/2, correction « hold-matched »

- **Le plus proche d'un maintien appris : Mamba-3.** La récurrence s'écrit
  h_t = e^{Δ_t A_t} h_{t−1} + (1−λ_t) Δ_t e^{Δ_t A_t} B_{t−1} x_{t−1} + λ_t Δ_t B_t x_t.
  λ_t ∈ [0, 1] est appris et dépend des données [V, Prop. 1]. λ = 1/2 redonne le trapèze
  classique, λ = 1 l'Euler de Mamba-2 [V, Remarque 2]. L'erreur est en O(Δ³) si
  λ_t = 1/2 + O(Δ), mais « not enforcing this constraint is better for empirical
  performance » [V, Remarque 3]. Le modèle apprend donc la combinaison des deux
  extrémités, c'est-à-dire la position effective de l'échantillon dans l'intervalle.
  Mamba-3 est évalué en langage, à pas fixe : aucun test de transfert de fréquence
  [V, grep « resolution/sampling rate » sans résultat].
- **Convention de l'extrémité.** Mamba-3 dérive le ZOH en fixant l'entrée à l'extrémité
  *droite* de l'intervalle [τ_{t−1}, τ_t] [V, §3.1, « ZOH »]. Le noyau de nablafx donne
  la même chose : K[0] = C(e^{ΔA} − 1)/A · B = ∫₀^Δ h(s) ds [V, code]. Donc
  y_n = Σ_k u_{n−k} ∫_{kΔ}^{(k+1)Δ} h. L'échantillon u_m est tenu sur [(m−1)Δ, mΔ],
  c'est-à-dire centré en (m − ½)Δ.
- **Ce qui en découle (déduction).** Les deux lectures sont compatibles.
  - Le maintien à droite *avance* l'entrée tenue de Δ/2.
  - L'entraînement compense : pour que Δ·h((k+½)Δ) égale la cible Δ·g(kΔ), il faut
    h(t) ≈ g(t − Δ/2). Le h appris porte donc bien le *retard* de Δ/2 du brainstorm.
  - À un autre pas Δ′, le décalage net vaut (Δ − Δ′)/2 = τ/2. D'où une phase ∝ τ et un
    ESR ∝ τ², la loi observée dans le brainstorm.
  - La correction « hold-matched » garde la largeur de porte T48 : elle vaut quelle que
    soit la convention de signe.
- **Le « bilinéaire » de S4 a le même décalage (déduction, vérifiée numériquement).**
  - S4 (éq. 3), le listing de S4D et le code `s4.py` utilisent
    B̄ = (I − Δ/2·A)⁻¹ ΔB, avec l'entrée u_k seule [V]. Mamba-3, Table 1, classe ce
    « Trapezoidal » de S4 avec β = « — » : il n'y a pas de terme en u_{k−1} [V].
  - Au premier ordre, B̄ ≈ Δ e^{AΔ/2} B, donc K[k] ≈ Δ·h((k+½)Δ), comme en ZOH. Un
    vrai Tustin sur l'entrée, B̄(u_k + u_{k−1})/2, donne K[k] ≈ Δ·h(kΔ), sans
    décalage.
  - Contrôle numérique (numpy, un pôle, Re λ = −300 s⁻¹, Δ = 1/48000 ; recherche du
    décalage s qui minimise l'écart entre K[k]/Δ et h((k+s)Δ)) :

    | pôle | ZOH | bilinéaire de S4 | Tustin sur l'entrée |
    |---|---|---|---|
    | 200 Hz | s = +0,500 | s = +0,495 | s = −0,005 |
    | 1 kHz | s = +0,500 | s = +0,410 | s = −0,090 |

    À 3 kHz, le gauchissement du bilinéaire domine et aucun décalage ne recale le noyau.
  - Conséquence : la nullité ZOH contre bilinéaire dans S4D est attendue, puisque les
    deux règles partagent le décalage. « Bilinéaire comme dans le S4 original » garderait
    le décalage de τ/2. Seules des règles symétriques sur l'entrée devraient l'annuler :
    le trapèze complet (Mamba-3 avec λ = ½) ou un maintien du premier ordre
    triangulaire.
- **Correction qui garde la porte de la fréquence d'entraînement : non trouvée**
  (requêtes au §6). En contrôle, re-discrétiser un modèle à une autre période
  (d2c puis c2d, `d2d` de MATLAB) revient à rééchelonner Δ avec une porte à la nouvelle
  période [NV : connaissance générale ; page MathWorks non lue].
- **Voisin hors audio (non lu en entier).** Un modèle constitutif de plasticité (arXiv
  2609.07294) place la taille de l'incrément dans la discrétisation ZOH d'un S5 et
  annonce une robustesse à la résolution du chemin [V, résumé seulement].

### 2.4 Massi, Mezza, Bernardini : FNO pour la modélisation VA indépendante de la fréquence (DAFx26)

Papier 27 de DAFx26 (Cambridge, MA, 1-4 septembre 2026), pp. 217-222.

- **Mécanisme** [V, §2-3].
  - Le signal est découpé en trames de durée fixe T = 20 ms, avec une fenêtre racine
    de Hann et 75 % de recouvrement, puis reconstruit par recouvrement-addition [V, §3.1,
    §3.3, §4].
  - Chaque couche fait zℓ₊₁ = σ(F⁻¹(↑ϱ(W F₁:Ñ(zℓ) + R̃ F₁:κ̃(zℓ)))). Les poids spectraux
    R̃ agissent sur κ̃ = 64 modes de Fourier fixes, et la branche W est limitée à la bande
    d'entrée [V, éq. (8)].
  - Comme κ ne dépend pas du nombre de points de la FFT, le même opérateur s'évalue sur
    une grille plus ou moins dense [V, §2].
  - Le sur-échantillonnage interne ϱ se fait par zero-padding spectral ; il vaut 2 à
    l'entraînement et peut être changé à l'inférence [V, §3.2, §4].
  - Le FNO n'a pas d'état d'une trame à l'autre, et le traitement est tamponné par
    trame [V, §3.1, §3.3].
- **Dispositif** [V, §4-5].
  - Étage d'entrée du Big Muff Pi (un BJT 2N5089), simulé sous Simscape ; environ 130 s,
    découpées 90/10.
  - Entraînement à 48 kHz. FNO de 17 785 paramètres (L = 3, 8 canaux, SiLU).
  - Référence : GRU-64 de 12 929 paramètres, rendu indépendant de la fréquence par la
    méthode LIDL de Carson et al.
  - Tests à 24, 88,2 et 96 kHz, avec ϱ₀ = 1 à l'inférence.
- **Résultats** [V, Table 1] :

  | métrique | modèle | 24 kHz | 48 kHz | 88,2 kHz | 96 kHz |
  |---|---|---|---|---|---|
  | MAE (×10⁻³) | LIDL-RNN | N/A | 2,173 | 2,319 | 2,333 |
  | MAE (×10⁻³) | FNO | 2,894 | 2,897 | 2,898 | 2,898 |
  | SC (×10⁻²) | LIDL-RNN | N/A | 3,852 | 3,920 | 3,932 |
  | SC (×10⁻²) | FNO | 2,932 | 3,237 | 3,408 | 3,431 |

  - En montée, la MAE favorise le RNN à toutes les fréquences et la SC favorise le FNO
    [V, §5.2].
  - Le FNO s'applique aussi en descente (24 kHz), ce que LIDL ne permet pas.
  - À 24 kHz, la SC (2,932) est plus basse qu'à 48 kHz. Je ne l'interprète pas.
  - À 24 kHz, ϱ₀ = 2 à l'inférence réduit le repliement : SNRA plus élevé en haut de la
    plage. Ce résultat n'est donné qu'en figure (Fig. 4 et 5) [V, §5.3].
- **Réserves.**
  - La vérité terrain à 24, 88,2 et 96 kHz est le signal d'évaluation *rééchantillonné*
    [V, §5]. Pour le sweep à 24 kHz, c'est la sortie Simscape rééchantillonnée [V, §5.3].
    Le papier mesure donc la cohérence avec 48 kHz, pas la fidélité au-dessus de 24 kHz.
    C'est la même limite que la nôtre (revue du 22/09, §1.6).
  - Un seul circuit, un seul entraînement : aucun écart-type ni graine dans le texte [V].
    Pas d'ESR.
  - Architecture par trames, sans état : on ne peut pas la comparer directement au
    rééchelonnement de Δ d'un SSM échantillon par échantillon.
- **Ce qui concerne A et C.** Le FNO échappe à la question du maintien : il n'a pas de
  récurrence discrétisée. Le papier ne discute ni ZOH, ni bilinéaire, ni retard de dt/2
  [V, lecture intégrale].

**Conséquence pour la piste A :** non documentée. Aucune correction « hold-matched »
trouvée. Le maintien appris existe à pas fixe (Mamba-3), sans transfert. Le mécanisme
du brainstorm est cohérent avec l'analyse du noyau (§2.3, déduction), mais la loi en τ²
reste à mesurer.

**Conséquence pour la piste C :** non documentée en transfert, et à reformuler.
Passer au « bilinéaire comme dans le S4 original » ne devrait pas supprimer le décalage
(§2.3, déduction). Seule une règle symétrique sur l'entrée le devrait : le trapèze
complet avec le terme u_{k−1}, ou le maintien triangulaire. Aucune n'a été testée en
transfert de fréquence. Wilczek montre que l'ordre du schéma n'agit pas de façon
monotone sur le transfert des neural ODE. La prédiction « dt⁴ » n'a aucun appui publié.

## 3. Q2 : les réglages modulent-ils les pôles ou Δ ? (piste F)

Classement par ce que le réglage touche.

### 3.1 Valeurs de composants continues, avec T explicite

- **Esqueda, Kuznetsov, Parker (DAFx20in21).**
  - Réglage → petit réseau tanh de largeur 1 (forme de Holmes et van Walstijn) →
    résistance du potentiomètre → matrices A, B, C du tone stack Marshall en maille
    [V, §3, éq. (12)-(14)].
  - Discrétisation par la règle du trapèze, avec ρ = T_s/(2RC) pour le RC [V, §2].
    Les valeurs de composants sont apprises par des facteurs d'échelle [V, §3].
  - Mesures à 96 kHz ; aucun test à une autre fréquence [V, §2.2.1].
- **Miklánek, Wright, Välimäki, Schimmel (DAFx23).**
  - Réglages → « tapers » non linéaires appris (même forme) → résistances variables R_V
    → méthode DK [V, §3.1, éq. (14)-(15)].
  - Gx = 2C/T, où T est la période d'échantillonnage : discrétisation trapézoïdale
    [V, §3.1].
  - Composants appris à ±20 % [V, §3.1]. Enregistrement à 44,1 kHz ; aucun test à une
    autre fréquence [V, grep].
- **Déduction.** Dans ces deux modèles, le réglage agit sur une constante de temps
  continue, et T n'entre qu'à la discrétisation. Ils sont donc indépendants de la
  fréquence par construction, pour la partie linéaire (au gauchissement du trapèze
  près). Aucun des deux ne le mesure.

### 3.2 Pôles ou coefficients discrets, liés à la fréquence d'entraînement

- **Sinjanakhom, Damskägg et al. (DAFx24, tone stack).**
  - Un réseau dense sur les réglages prédit y ; y est transformé en pôles complexes
    discrets p₀ = f(y₀ + j y₁), stabilisés (|p| < 1), dans un filtre d'état en forme
    couplée [V, §3, éq. (1)-(3)].
  - Les pôles vivent dans le plan z, donc sont liés à la fréquence d'entraînement
    (déduction).
- **Nercessian, Sarroff, Werner (IEEE 2021, arXiv 2103.08709).** Hyperconditionnement :
  une transformation affine des réglages donne, au choix [V, §3.2-3.3] :
  - les coefficients de biquads ;
  - des paires pôle/zéro ;
  - des paramètres d'égaliseur (f, g, Q) passés par les formules RBJ.

  La variante en Hz serait indépendante de la fréquence (déduction).
- **Yeh, Hsiao, Yang, HyperRNN (DAFx24).**
  - FiLM-RNN met à l'échelle F_hc = W_h h_{t−1}, donc le terme de rétroaction [V, §4.1].
  - StaticHyper-RNN génère W_h, W_x et b à partir des réglages [V, §4.2].
  - DynamicHyper fait varier ces poids dans le temps [V, §4.3].
  - Sur le LA-2A, StaticHyper-GRU bat FiLM-GRU en L1 (0,008 contre 0,011) et en STFT
    (0,371 contre 0,597). FiLM-GRU gagne en transitoires (15,825 contre 20,437) et en
    facteur de crête (2,006 contre 2,386) [V, Table 2].
  - Les deux variantes agissent sur la récurrence : cette table ne compare pas les pôles
    et les activations.
  - **Déduction :** moduler W_h revient à moduler les pôles de la récurrence linéarisée,
    dans le domaine discret.
- **DDSP Guitar Amp (Yeh et al., 2024).**
  - Un contrôleur MLP transforme les réglages en paramètres d'un tone stack
    low-shelf/peak/high-shelf [V, §2.2].
  - Le réglage de gain agit avant et après la non-linéarité [V, §2.1].
- **nablafx, grey-box (Comunità, Steinmetz, Reiss, 2025, arXiv 2502.14405).**
  - Un modèle grey-box est une suite de processeurs pilotés chacun par un contrôleur ;
    des contrôleurs conditionnels le rendent paramétrique [V, §3.2].
  - L'égaliseur paramétrique expose gain, fréquence de coupure et Q par bande [V, code
    `nablafx/processors/ddsp.py`, `ParametricEQ`].
  - Le papier classe Nercessian comme de l'hyperconditionnement [V, §2].

### 3.3 Activations après le SSM (la pratique actuelle, y compris la nôtre)

- **nablafx, boîte noire.** `S4CondBlock.forward` enchaîne linéaire, activation, `s4`,
  puis FiLM/TFiLM/TVFiLM sur la sortie du SSM [V, code `blocks.py`, l. 438-466].
- **Simionato & Fasciani (2405.04124).** FiLM appliqué juste avant la couche de sortie.
  « Placing the conditioning block after the recurrent layers rather than before has
  been found beneficial » [V, §3]. **Nuance :** « avant » y désigne l'entrée de la
  couche récurrente, pas ses pôles. Ce n'est donc pas un contre-argument direct à F,
  mais c'est la seule comparaison de placement trouvée.
- **Compresseur optique, S6 (Simionato & Fasciani, JAES 2025, 2408.12549).** Le
  conditionnement vient après le bloc S6. Les réglages de temps (attaque et relâchement,
  donc des constantes de temps) passent par un Temporal FiLM sur la sortie, pas par Δ
  [V, §3 et Fig. 3].

### 3.4 Indice hors audio : où mettre la dépendance

- **TIDES (Soydan et al., 2026).**
  - La dépendance à l'entrée porte sur Re(Λ), pas sur Δ, qui garde son sens physique
    (ZOH) [V, résumé, §3].
  - Sur la tâche « Fading Flash », entraînement pour Δ ∈ [0,5 ; 1,5] et test de 0,1 à
    2,0 [V, annexe C.4]. TIDES et S5 gardent une décroissance apprise constante sur tout
    l'intervalle de test.
  - Le Mamba-like, où Δ est une fonction apprise de l'entrée, « drift[s] outside the
    training range » [V, §4, Fig. 5]. Chiffres en figure seulement, une seule graine
    [V, annexe C.3].
  - La dépendance vient de l'entrée, pas de réglages. **Déduction pour F :** une
    modulation par les réglages devrait porter sur Λ, ou sur un facteur de Δ séparé du
    1/fs, pour garder le rééchelonnement exact.
- **Pôles modulés par un LFO en grey-box** (phaser 2306.01332 ; filtres tout-pôles
  2404.07970) : relevés, non lus [NV].

**Conséquence pour la piste F :** déjà faite en grey-box (Esqueda 2021, Miklánek 2023)
pour des circuits connus : réglage → R → pôles continus → discrétisation avec T. Non
trouvée en boîte noire : aucun SSM profond dont les réglages modulent Λ ou Δ. Le plus
proche est la génération de W_h par HyperRNN, dans le domaine discret. TIDES soutient
le choix de Λ plutôt que Δ. La comparaison « FiLM sur les pôles » contre « FiLM sur les
activations » à paramètres égaux n'a pas été trouvée.

## 4. Q3 : entraînement multi-fréquence ou multi-Δ (piste D)

- **S4ND (Nguyen et al., NeurIPS 2022), progressive resizing.**
  - CIFAR-10, bas puis base (80 − 20 époques) : 92,32 % contre 93,40 % en base seule,
    avec 21,8 % de temps en moins [V, Table 5].
  - Celeb-A, bas puis moyen (16 − 4 époques), testé en base : 88,57 % [V, Table 5].
  - **Piège :** le bon comparateur est l'entraînement moyen testé en base, 90,14 %
    [V, Table 4], pas l'entraînement bas testé en base (84,73 %). Les budgets d'époques
    de la Table 4 n'ont pas été lus [NV].
  - Présenté par les auteurs comme une méthode d'accélération : aucun gain de transfert
    n'est isolé.
- **TAWM (Nhu, Son, Lin 2025).**
  - Modèle du monde conditionné par Δt et entraîné sur Δt tiré log-uniformément
    (Meta-World : 0,001 à 0,05 s) [V, §3-4].
  - Il bat les modèles entraînés à Δt fixe sur une plage de Δt d'inférence
    [V, texte §5]. Chiffres en figure seulement, 3 graines [V, légendes].
  - Ce n'est pas un SSM : Δt est une entrée du réseau.
- **TIDES.** Entraînement sur une plage de Δ, test hors plage (§3.4). Chiffres en
  figure, une graine [V].
- **S5 §6.3** : Δt fourni pas à pas pour des observations irrégulières (pendule). Déjà
  [V] dans la revue du 22/09, §2.5.
- **NU-Wave 2** : fréquence d'entrée tirée pendant l'entraînement, pour de la
  super-résolution de parole [NV, résumé de recherche ; voir aussi la revue du 22/09].
- **Pas du multi-Δ, mais de la montée depuis une seule fréquence :** PDM (§2.2), caméras
  événementielles (§2.2), Piano-SSM (revue du 22/09).
- **Relevé, non lu :** « Efficient and interpretable raw audio classification with
  diagonal state space models » (Machine Learning, Springer 2025), qui mentionne
  l'utilisation à plusieurs fréquences [NV, accès refusé].

**Conséquence pour la piste D :** non documentée pour les effets audio. Le principe
existe hors audio (TAWM, TIDES, S5), mais aucune source ne donne en table le gain de
transfert dû au multi-Δ par rapport à un entraînement à une seule fréquence. Le point
du brainstorm tient : au-dessus de 24 kHz il n'y a pas de cible. Aucune source ne
contredit cela.

## 5. Non trouvé

Après les requêtes du §6, **non trouvé** (ce qui n'établit pas que ça n'existe pas) :

1. Une correction de maintien qui garde la porte de la fréquence d'entraînement quand
   Δ change (« hold-matched »), en SSM, en neural ODE ou en contrôle.
2. Une comparaison ZOH / FOH / bilinéaire *en transfert de fréquence* pour un SSM,
   quel que soit le domaine.
3. Une analyse d'un « retard de dt/2 » appris par un SSM et de son effet au changement
   de fréquence.
4. Un SSM profond d'effet audio dont les réglages modulent Λ, Re λ ou Δ.
5. Un entraînement multi-fréquence ou multi-Δ pour un modèle d'effet ou d'ampli.
6. Un SSM d'effet audio testé à une autre fréquence : toujours rien après la revue du
   22/09. Le seul nouveau travail audio VA sur la fréquence est le FNO de Massi, qui
   n'est pas un SSM.

## 6. Requêtes consignées (2026-09-23)

**WebSearch** (pour trouver seulement) :

| # | requête | résultat utile |
|---|---|---|
| 1 | Massi Mezza Bernardini Fourier Neural Operators sample-rate-independent virtual analog modeling | rien de direct |
| 2 | "Fourier Neural Operators" "Sample-Rate-Independent" DAFx 2026 | rien de direct ; site de DAFx26 |
| 3 | DAFx26 accepted papers proceedings 2026 | programme sur dafx26.mit.edu |
| 4 | Miklánek DAFx 2023 hybrid neural tone stack DK method differentiable | 9 liens ; tone stack DAFx24, nablafx |
| 5 | tone stack neural state-space model 2024 potentiometer conditioning arXiv | DAFx24 paper 58 |
| 6 | "Neural Grey-Box Guitar Amplifier Modelling With Limited Data" pdf DAFx23 | PDF DAFx23 n° 52 |
| 7 | Esqueda Kuznetsov Parker "Differentiable white-box virtual analog modeling" | PDF DAFx20in21 n° 39 |
| 8 | state space model trained at multiple sampling rates … Δ training | TAWM, TIDES, caméras événementielles |
| 9 | Mamba-3 exponential-trapezoidal discretization arXiv | 2603.15569, 2604.20606 |
| 10 | state space model discretization choice ZOH bilinear effect on resolution transfer zero-shot | 2604.20606, 2508.20441 |
| 11 | first-order hold discretization structured state space model S4 FOH generalized hold | rien de nouveau |
| 12 | neural state space model trained on multiple sampling rates … unseen sampling rate audio | 2608.28472 |
| 13 | "zero-order hold" learned continuous-time model … half-sample delay correction neural ODE | rien de pertinent |
| 14 | SSM control parameters modulate eigenvalues or timescale Δ conditioning knob audio effect | 2408.12549 |

**API arXiv** (`export.arxiv.org`, nombre de résultats) :

| requête | nombre |
|---|---|
| `abs:operator AND abs:analog AND abs:audio` | 19, aucun pertinent |
| `abs:operator AND abs:"sample rate"` | 20 ; 2608.28472 relevé |
| `abs:"zero-order hold" AND abs:"state space" AND abs:resolution` | 1 (2609.07294) |
| `abs:"zero-order hold" AND abs:"sampling rate" AND abs:neural` | 0 |
| `abs:discretization AND abs:"state space model" AND abs:"sampling rate"` | 0 |
| `abs:"first-order hold" AND abs:"state space" AND abs:neural` | 0 |
| `abs:bilinear AND abs:"zero-order hold" AND abs:SSM` | 1 (2604.20606) |
| `abs:"state space" AND abs:"audio effect"` | 2 |
| `abs:"state space" AND abs:"virtual analog"` | 1 (2405.04124) |
| `abs:hypernetwork AND abs:"audio effect"` | 0 |
| `au:Massi AND au:Bernardini` | 0 : papier FNO non trouvé sur arXiv par cette requête |

**Semantic Scholar** : « state space model audio effect conditioning control
parameters » (4811 résultats ; 15 premiers titres lus). Les autres requêtes ont reçu
une réponse 429.

**Archives** : programme de DAFx26 (`dafx26.mit.edu/program/`), analysé et parcouru
pour operator, sample rate, state space, SSM, Massi, Bernardini, tone stack, discreti.

**Tri** : 24 PDF téléchargés et lus au moins par grep ciblé, plus 2 dépôts de code ; 21 sources citées ci-dessus
avec [V]. Écartées après lecture : 2604.20606 (pas de transfert) ; 2609.07294 (hors
sujet, résumé seulement) ; 2508.20441 (gardée pour un seul chiffre).

## 7. Sources

Métadonnées arXiv vérifiées par `id_list` le 2026-09-23.

**Déjà dans la revue du 22/09 (non relues ici, sauf mention)**
- Gu, Goel, Ré, S4, ICLR 2022 — https://arxiv.org/abs/2111.00396 (relu : §2.3, éq. 3)
- Smith, Warrington, Linderman, S5, ICLR 2023 — https://arxiv.org/abs/2208.04933 (relu : éq. 6)
- Nguyen et al., S4ND, NeurIPS 2022 — https://arxiv.org/abs/2210.06583 (relu : Tables 4-5)
- Dallinger et al., Piano-SSM, DAFx25 — https://dafx25.dii.univpm.it/wp-content/uploads/2025/07/DAFx25_paper_57.pdf (relu : §3.1, éq. 2)
- Simionato, Fasciani, arXiv 2405.04124 / EURASIP JASMP 2025 — https://arxiv.org/abs/2405.04124 (relu : §3)
- Wilczek et al., DAFx22 — https://www.dafx.de/paper-archive/2022/papers/DAFx20in22_paper_12.pdf (non relu ; chiffres [V] le 22/09)
- Saito et al., EUSIPCO 2021 — https://arxiv.org/abs/2105.04079 (non relu)

**Nouvelles sources, lues [V]**
- Massi, Mezza, Bernardini, *Fourier Neural Operators for Sample-Rate-Independent Virtual Analog Modeling*, DAFx26, pp. 217-222 — https://dafx26.mit.edu/assets/papers/DAFx26_paper_27.pdf
- Gu, Gupta, Goel, Ré, *On the Parameterization and Initialization of Diagonal State Space Models* (S4D) — https://arxiv.org/abs/2206.11893
- Gu, Dao, *Mamba* — https://arxiv.org/abs/2312.00752
- Lahoti et al., *Mamba-3: Improved Sequence Modeling using State Space Principles*, 2026 — https://arxiv.org/abs/2603.15569
- Ibrahim, Liu, Wang, *Beyond ZOH: Advanced Discretization Strategies for Vision Mamba*, 2026 — https://arxiv.org/abs/2604.20606
- Solozabal et al., *Uncovering the Spectral Bias in Diagonal State Space Models*, 2025 — https://arxiv.org/abs/2508.20441
- Zubić, Gehrig, Scaramuzza, *State Space Models for Event Cameras*, CVPR 2024 — https://arxiv.org/abs/2402.15584
- Boulanger, Wood, *Multirate State Space Models for End-to-End Processing of Pulse Density Modulated Speech Signals*, 2026 (soumis à l'IEEE) — https://arxiv.org/abs/2608.28472
- Bittner et al., *Diagonal Complex-Valued State Space Models for … Metal Plate Reverbs*, DAFx26 — https://dafx26.mit.edu/assets/papers/DAFx26_paper_03.pdf
- Esqueda, Kuznetsov, Parker, *Differentiable White-Box Virtual Analog Modeling*, DAFx20in21 — https://dafx.de/paper-archive/2021/proceedings/papers/DAFx20in21_paper_39.pdf
- Miklánek, Wright, Välimäki, Schimmel, *Neural Grey-Box Guitar Amplifier Modelling with Limited Data*, DAFx23 — https://www.dafx.de/paper-archive/2023/DAFx23_paper_52.pdf
- Sinjanakhom, Damskägg et al., *Guitar Tone Stack Modeling with a Neural State-Space Filter*, DAFx24 — https://www.dafx.de/paper-archive/2024/papers/DAFx24_paper_58.pdf
- Nercessian, Sarroff, Werner, *Lightweight and interpretable neural modeling of an audio distortion effect using hyperconditioned differentiable biquads*, IEEE 2021 — https://arxiv.org/abs/2103.08709
- Yeh, Hsiao, Yang, *Hyper Recurrent Neural Network*, DAFx24 — https://arxiv.org/abs/2408.04829
- Yeh et al., *DDSP Guitar Amp: Interpretable Guitar Amplifier Modeling*, 2024 — https://arxiv.org/abs/2408.11405
- Comunità, Steinmetz, Reiss, *Differentiable Black-box and Gray-box Modeling of Nonlinear Audio Effects*, 2025 — https://arxiv.org/abs/2502.14405
- Simionato, Fasciani, *Modeling Time-Variant Responses of Optical Compressors with Selective State Space Models*, JAES 2025 — https://arxiv.org/abs/2408.12549
- Soydan, Bessa, Mohr, Barreira, *TIDES: Implicit Time-Awareness in Selective State Space Models*, 2026 — https://arxiv.org/abs/2605.09742
- Nhu, Son, Lin, *Time-Aware World Model for Adaptive Prediction and Control*, 2025 — https://arxiv.org/abs/2506.08441

**Code lu localement [V]**
- `third_party/nablafx` (commit 045db6e) : `nablafx/processors/blocks.py`, `nablafx/processors/ddsp.py`, `tests/unit/test_processors/ineffab1evista_s4drc/s4.py` (copie du S4 autonome de `state-spaces`)
- `third_party/optical-selective-ssm` (commit bd787ec) : `src/Code/S4D.py`

**Relevées, non lues [NV]**
- Comunità, Steinmetz, Reiss, *NablAFx* (article du framework) — https://arxiv.org/abs/2502.11668
- Constitutive State-Space model — https://arxiv.org/abs/2609.07294 (résumé seulement)
- Phaser grey-box — https://arxiv.org/abs/2306.01332 ; filtres tout-pôles différentiables — https://arxiv.org/abs/2404.07970
- Bittner, Schnöll, Wess et al., *Efficient and interpretable raw audio classification with diagonal state space models*, Machine Learning 114 (2025), DOI vérifié par Crossref — https://link.springer.com/article/10.1007/s10994-025-06807-z
- MathWorks `d2d` — https://www.mathworks.com/help/control/ref/dynamicsystem.d2d.html
