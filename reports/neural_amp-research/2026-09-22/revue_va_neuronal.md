<!-- Copie telle quelle du scratchpad de la session du 2026-09-22 (draft.md) : revue VA neuronal, dont conditionnement par les reglages, donnees et evaluation. -->

# Draft — revue VA neuronal (mes findings vérifiés)

## Ce que l'équipe a DÉJÀ (paper/bibliography.bib + icassp2027/refs.bib + reports/neural_amp-research/2026-09-05/review.bib)
Wright DAFx19/AppliedSci20, Damskägg ICASSP19+SMC19, Comunità Frontiers25+NablAFx+ToneTwist,
Comunità ICASSP23 TFiLM, TFiLM Birnbaum, WaveNet, S4/DSS/S4D/SaShiMi/Mamba/LRU,
Yin2024 SSM DRC, Simionato&Fasciani EURASIP25 + JAES25, Conditioning Methods SMC24,
Esqueda&Murai DAFx25 LRU antialiasé, SolidStateBusComp DAFx25, Unsupervised Estimation DAFx25,
Bittner DAFx26 plate SSM, auraloss, ParallelWaveGAN, sato2025 aliasing, sato2026 pruning iOS,
KLANN 2024, Wright perceptual loss 2020, Steinmetz micro-TCN 2021, Parker DAFx19 STN,
Kuznetsov DAFx20 IIR diff., DDSP Engel, Neural ODE VA 2022, all-pole diff. 2024,
tone stack neural state-space 2024, ADAA for RNN 2025, Cassidy 2023 perceptual eval,
EGFxSet, SRIndependent RNN 2024, phaser gray-box 2023, Beintema 2021, Tallec, Aicher.

=> NE PAS réexpliquer. Chercher AILLEURS.

## VÉRIFIÉ PAR LECTURE (moi)

### Mikkonen, Wright, Välimäki — "Sampling the user controls in neural modeling of audio devices"
EURASIP J. Audio Speech Music Process. 2024:26. DOI 10.1186/s13636-024-00347-5
https://link.springer.com/article/10.1186/s13636-024-00347-5
PDF libre: https://www.pure.ed.ac.uk/ws/portalfiles/portal/507801085/Wright2024EURASIPSampling.pdf
Code: https://github.com/01tot10/neural-parameter-sampling
- Cibles: ProCo RAT (distorsion) + Pultec EQ, SIMULÉES EN SPICE (pas de matériel réel).
- 125 jeux de données par cible; δ ∈ {3,5,9,17,continu}, n ∈ {1,2,4,8,16}, 5 tirages k par (δ,n).
  => la variance mesurée est celle du TIRAGE DE DONNÉES, PAS DE LA GRAINE. Un modèle par jeu.
  Agrégation par MÉDIANE. 430 000 pas d'optimisation, 3-5 h GPU par run.
- Test: contrôles tirés continûment dans U{0,1} = réglages NON VUS mais DANS la plage → interpolation.
  Audio de test: guitare/basse inédites MAIS "from the same distributions used during training" → PAS d'OOD.
- Résultat: δ=5 suffit. δ ≥ 8 continu légèrement DÉGRADÉ ("continuous sampling is a suboptimal choice").
- Écoute: MUSHRA-like ITU-R BS.1534, webMUSHRA, 15 auditeurs expérimentés, cabines Aalto, HD650.
  Ancre basse: tanh × 25 gain + high-shelf −18 dB @ Nyquist, coin 5,5 kHz. Segments −23 LUFS (pyloudnorm).
  Pultec EQ: tous les modèles δ≥5 perceptuellement indiscernables de la référence.
  ProCo RAT: saturation entre "good" et "excellent" — jamais "excellent".
- CITATION CLÉ: "the ESR is not a conclusive perceptual metric, and it should not be understood as a
  direct indicator of the model performance." Le modèle δ=9 du RAT a un ESR monotone meilleur mais
  est jugé NETTEMENT PIRE à l'écoute; c'est la perte STFT qui le détecte, pas l'ESR.
- Verrou déclaré par les auteurs: "Further work is also needed to establish an explanation of why the
  densest possible sampling of the parameter space is not always the best choice for the considered task."

### Juvela, Damskägg, Peussa, Mäkinen, Sherson, Mimilakis, Gotsopoulos — "End-to-End Amp Modeling: From Data to Controllable Guitar Amplifier Models"
ICASSP 2023. arXiv:2403.08559  https://arxiv.org/abs/2403.08559  (Neural DSP)
- Matchless DC-30, 5 contrôles continus (volume, bass, treble, tone cut, master).
- ROBOT de capture motorisé; problème combinatoire posé explicitement: 10^n enregistrements pour n boutons,
  "over a million combinations" à 6 boutons → résolu par un parcours optimisé (TSP-like) minimisant le
  déplacement des potentiomètres.
- Données: 4,5 h d'audio apparié, segments de 1 s à 48 kHz, 15 000 train + 1 000 val.
- Modèle: LSTM 1 couche 32 cellules, conditionnement par CONCATÉNATION des contrôles normalisés [0,1].
  1 M itérations, perte ESR, Adam.
- Référence blanche: LTspice + modèle de tube Koren; "each of test samples took several hours to render",
  "far from real time feasibility".
- Écoute: DMOS (1-5), webMUSHRA modifié, 5 cas, 30 auditeurs experts, casque, à distance.
  IR de baffle commune, loudness LUFS appariée.
  Test t apparié + Bonferroni: AUCUNE différence significative LSTM-32 vs SPICE.
  Les autres paires diffèrent significativement.
- Limite: les positions de boutons de test sont "adjusted to fit the musical context" — pas de balayage
  systématique des réglages non vus.

### Comunità, Steinmetz, Reiss — "Differentiable black-box and gray-box modeling of nonlinear audio effects"
Frontiers in Signal Processing 5:1580395 (2025). arXiv:2502.14405
https://www.frontiersin.org/journals/signal-processing/articles/10.3389/frsip.2025.1580395/full
(l'équipe l'a, mais les CHIFFRES de la Table 8 sont le seul vrai résultat publié de corrélation métrique/écoute)
- MUSHRA, 27 participants après filtrage sévère (écarter ceux qui notent la référence < 80 plus d'une fois
  sur 7 essais); 12 effets (3 par type), 68 questions possibles, 7 tirées par participant.
  => 350 notes valides, ~35 par architecture, ~8,75 par type d'effet. ÉCHANTILLON PAR CELLULE TRÈS PETIT.
- Table 8 — Spearman ρ entre notes subjectives et métriques (négatif = métrique basse ↔ note haute):
  Overall:      L1 −0,332 | MR-STFT −0,427 | MSE −0,314 | ESR −0,311 | MAPE −0,339
                FAD_VGGish −0,430 | FAD_PANN −0,303 | FAD_CLAP −0,406 | FAD_AFx-Rep −0,417
  Comp./Limit.: ESR −0,149 (p > 0,05, NON SIGNIFICATIF) ; MAPE −0,152 (NON SIGNIFICATIF) ; L1 −0,188
  Overdrive:    MAPE −0,361 (meilleure) ; MR-STFT −0,277 ; ESR −0,264
  Distortion:   MR-STFT −0,536 ; FAD_CLAP −0,613 ; ESR −0,389
  Fuzz:         MR-STFT −0,488 ; FAD_VGGish −0,511 ; ESR −0,470
  => MÊME LA MEILLEURE MÉTRIQUE (|ρ| ≈ 0,43) n'explique qu'environ 18 % de la variance de rang.
     Sur la compression, l'ESR n'est pas corrélé du tout de façon significative.
- Conclusions: S4 le plus adapté; LSTM "unreliable across devices"; DDSP gray-box "not yet on par";
  TFiLM améliore médiane ET variance.
- Verrous déclarés (future work, verbatim): identifier des métriques perceptuellement pertinentes;
  amener le gray-box au niveau du black-box; étudier le conditionnement paramétrique sur une large gamme
  d'effets avec implémentations efficaces; architectures universelles; optimisation d'hyperparamètres +
  pruning/distillation.
- Sur la généralisation (motivant ToneTwist): les travaux antérieurs entraînent et évaluent sur les mêmes
  sources, "not allowing to reach a reliable conclusion about the generalization capabilities of a trained
  model"; normalisation sans vérifier la plage dynamique, "not ensuring trained models to be accurate
  across signal levels".
- Instabilité LSTM: "resulting in frequent gradient explosion failures despite adjustments".

## NOUVEAU / NON DANS LEURS BIBS (vérifié par titre+venue, lecture variable)

### Conditionnement
- Yeh, Hsiao, Yang, "Hyper Recurrent Neural Network: Condition Mechanisms for Black-box Audio Effect
  Modeling", DAFx-24 (Guildford), pp. 97-104. arXiv:2408.04829
  https://dafx.de/paper-archive/2024/papers/DAFx24_paper_16.pdf
  3 mécanismes de conditionnement pour RNN (StaticHyper-RNN, DynamicHyper-RNN) vs concaténation.
- Kallinen, Juvela, Sherson, "Deep Regularized RNNs for Virtual Analog", DAFx-26 (Cambridge).
  arXiv:2509.15622  https://arxiv.org/abs/2509.15622
  VERROU: quand le conditionnement VARIE DANS LE TEMPS, les RNN produisent des artefacts de bruit;
  régulariser la dynamique les réduit mais au prix de la précision. Proposent LSTM profond conditionné +
  perte gammatone (GFB). NB: v1 avait des erreurs dans les dérivations Eq.(17) et (21), corrigées en v2.
- Grötschla, Lanzendörfer, Jiao, Wattenhofer, "Parametric Neural Amp Modeling with Active Learning"
  (PANAMA), ISMIR 2025 Late-Breaking Demo. arXiv:2507.02109 (+ 2509.26564)
  Choix actif des réglages à capturer par optimisation à base de gradient.
- Chen, Cheng, Yeh, Wu, Ho, Jang, Yang, "Towards zero-shot amplifier modeling: One-to-many amplifier
  modeling via tone embedding control", ISMIR 2024. arXiv:2407.10646
  + démo hypernetwork arXiv:2410.04702. Données Positive Grid BIAS FX2: 80 000 timbres, 5 300 h.
  Test zéro-shot sur 2 amplis non vus (High Gain EL34 V2, Dumble ODS 50).

### Boîte blanche / hybride WDF
- Giampiccolo, Polimeno, Macrì, Lenoci, Massi, Bernardini, "Explicit Wave Digital Model of the Fulltone
  OCD Pedal Based on Canonical Piecewise-Linear Functions", DAFx-26.  <-- MÊME PÉDALE QUE LEUR DIAGNOSIS
- Massi, Mezza, Giampiccolo, Bernardini, "Training Neural Models of Nonlinear Multi-Port Elements Within
  Wave Digital Structures Through Discrete-Time Simulation", DAFx-25. (convergence WDF × apprentissage)
- Massi, Mezza, Bernardini, "Fourier Neural Operators for Sample-Rate-Independent Virtual Analog
  Modeling", DAFx-26.
- Zea & Rivera, "Towards Efficient Emulation of Nonlinear Analog Circuits for Audio Using Constraint
  Stabilization and Convex Quadratic Programming", DAFx-25 ; et "Residual-Driven Adaptive Multi-Rate
  Quadratic Programming Framework for Nonlinear Analog Audio Circuit Emulation", DAFx-26.
- Andò, Bozzo, Fontana, "Non-Iterative Numerical Simulation in Virtual Analog: A Framework Incorporating
  Current Trends", DAFx-25.
- Pines, "Real-Time Virtual Analog Modelling of Diode-Based VCAs", DAFx-25.
- Pakarinen & Yeh, "A review of digital techniques for modeling vacuum-tube guitar amplifiers",
  Computer Music Journal 33(2):85-100, 2009. (revue historique boîte blanche)

### Aliasing
- Vanhatalo, Legrand, Desainte-Catherine, Hanna, Pille, "Evaluation of Real-Time Aliasing Reduction
  Methods in Neural Networks for Nonlinear Audio Effects Modelling", JAES 72(3):114-122, mars 2024.
  https://aes.org/publications/elibrary-page/?id=22384  DOI 10.17743/jaes.2022.0122
  Méthodes comparées: données suréchantillonnées, passe-bas entre couches, pertes spectrales, pruning.
  SEULE méthode temps réel viable trouvée: SPARSITÉ FORCÉE par pruning — efficace sur TCN, moins sur
  LSTM, et au prix de la précision.
- Carson, Wright, Bilbao, "Anti-Aliasing of Neural Distortion Effects via Model Fine Tuning", DAFx-25.
  arXiv:2505.11375
- Mikkonen & Werner, "Antiderivative Antialiasing for Recurrent Neural Networks", DAFx-25. (dans leur bib)
- Gabrielli & Squartini, "PolyADAA: Improving Aliasing Reduction in Memoryless Nonlinearities Using
  Lagrange Interpolation and Polynomial Approximation", DAFx-26.
- Gabrielli & Squartini, "Simplifying Antiderivative Antialiasing with Lookup Table Integration", DAFx-25.

### Outils / reproductibilité
- Yeh, Hsiao, Yang, "PyNeuralFx: A Python Package for Neural Audio Effect Modeling", ISMIR 2024 LBD.
  arXiv:2408.06053  https://github.com/ytsrt66589/pyneuralfx
  Motivation explicite: "different works often employ varied training strategies, loss functions, and
  evaluation metrics, making it hard to compare models". Concurrent/antérieur à NablAFx.
- 1st DAFx Challenge (DAFx-26, Cambridge): https://dafx26.mit.edu/challenge/
  https://github.com/LOGUNIVPM/1st-DAFx-Challenge
  MAIS: porte sur l'ESTIMATION DE PARAMÈTRES D'UN MODÈLE MODAL DE PLAQUE (reverb à plaque),
  PAS sur la modélisation d'amplis. 22 soumissions, 13 équipes. => il n'existe TOUJOURS PAS de
  benchmark partagé pour le VA neuronal d'amplis.

### Données
- Wright, Carson, Juvela, "Open-Amp: Synthetic Data Framework for Audio Effect Foundation Models",
  ICASSP 2025. arXiv:2411.14972
  Crowdsource des émulations neuronales d'utilisateurs (GuitarML, Neural Amp Modeler) → "hundreds of
  devices"; rendu audio en ligne pendant l'entraînement. NB: vérité terrain = un MODÈLE, pas l'appareil.
- Dal Rí, Stefani, Turchet, Conci, "MorphDrive: Latent Conditioning for Cross-Circuit Effect Modeling and
  a Parametric Audio Dataset of Analog Overdrive Pedals", DAFx-25. (NOUVEAU dataset paramétrique)
- Gu, Zhang, Juvela, Wu, "Solid State Bus-Comp", DAFx-25 (déjà dans leur bib)
- Mikkonen et al., "Magnetic Tape Recorder Dataset", Zenodo 8026272.
- SignalTrain (Hawley et al. 2019), EGDB.

### Quantité de données
- Wright, Damskägg, Välimäki, Applied Sciences 10(3):766, 2020: 3 minutes d'audio suffisent.
  Ablation: sous-ensemble de 5 min (2,5 guitare + 2,5 basse) pour l'entraînement, +1 min de validation.
- Carson et al. DAFx-25 fine-tuning: 3 min 40 s à 44,1 kHz (chirps + bruit + jeu), signal GuitarML.
- Contraste: zéro-shot multi-amplis → 5 300 h / 80 000 timbres.

### Mémoire longue / état
- Mikkonen, Wright, Moliner, Välimäki, "Neural Modeling of Magnetic Tape Recorders", DAFx-23
  (Copenhague), pp. 196-203. arXiv:2305.16862
  https://www.dafx.de/paper-archive/2023/DAFx23_paper_51.pdf
  HYSTÉRÉSIS modélisée par GRU + couche linéaire, motivée par la nature "stateful" de la récurrence;
  délai fluctuant et bruit par modèles de DIFFUSION (U-net) séparés. Données CHOWTape (jouet) + appareil réel.
  3 schémas d'entraînement (2 supervisés, 1 adverse).
- Simionato & Fasciani, "Towards Neural Emulation of Voltage-Controlled Oscillators", DAFx-25.

### LA RÉFÉRENCE LA PLUS PROCHE DE LEUR TRAVAIL SUR LA DISPERSION — VÉRIFIÉE PAR LECTURE
Bourdin, Legrand, Roche, "Empirical Results for Adjusting Truncated Backpropagation Through Time
while Training Neural Audio Effects", DAFx-25 (Ancona). arXiv:2512.07393
https://arxiv.org/abs/2512.07393  code: https://github.com/ybourdin/sptmod
démos+écoute: https://ybourdin.github.io/sptmod/
- Grille TBPTT: N (sous-séquences) ∈ {1,2,3} × B ∈ {8,16,32,64,128} × L ∈ {4096,8192,16384,32768}
  (hors (128,32768), mémoire). 570 runs par jeu de données.
- 10 instances par combinaison = 10 DÉCOUPAGES train/val/test ("by shuffling items before splitting"),
  PAS 10 graines. AUCUNE ablation graine-seule à tout le reste fixé. Ils reconnaissent la graine comme
  source de variance ("variance from sources like initial model weights and batches constitution and
  order") sans l'isoler.
- Ils rapportent une MAD (médiane des écarts absolus à la médiane) sur les 10 instances, Tables 3 et 4.
  "using TBPTT (N>1) typically improves MAD values" ; "for high L ... small MAD values are achieved
  with N=1" ; "Training variance, measured by MAD values, is generally greater on the Threshold-Ratio
  dataset than the snapshot one".
- CITATION MÉTHODOLOGIQUE CLÉ (la plus proche de la préoccupation du projet, verbatim):
  "training a model once does not reliably estimate risk or allow meaningful model comparisons"
- State Prediction Network (SPN): prédit les états initiaux LSTM à partir du DRY seul
  ("receives the first L_lookback samples" de l'entrée), évitant un préchauffage long.
  => C'est exactement l'idée I03 de leur propre revue, déjà publiée.
- Cible: compresseur API-2500+ (mode feed-forward), Arturia AudioFuse, 44,1 kHz. 3 jeux de données
  (snapshot 16 items ; Threshold-Ratio 16 items ; full-modeling 160 items par Latin Hypercube Sampling).
- Écoute: MUSHRA via webMUSHRA en ligne, 3 extraits × 4 réglages × 2 longueurs = 24 items,
  29 participants, 14 retenus après critères → 168 évaluations. Durée médiane 26 min.
  "the revised TBPTT setup did not significantly impact the model's perceptual quality."
- Limite déclarée: le jeu full-modeling exclu de la grille (coût) ; "implementing the model in C++ will
  allow for a precise estimation of its computational efficiency and real-time viability".

### Autres DAFx-25 / DAFx-26 non dans leurs bibs
- Clarke & Chowdhury, "Inference-Time Structured Pruning for Real-Time Neural Network Audio Effects", DAFx-25.
- Rossi, Iacca, Turchet, "Automatic Classification of Chains of Guitar Effects Through Evolutionary
  Neural Architecture Search", DAFx-25.
- Ringhofer, Gnoss, Schiele, "Balancing Error and Latency of Black-Box Models for Audio Effects Using
  Hardware-Aware Neural Architecture Search", DAFx-24.
- Hoopes, Chalmers, Zappi, "Neural Audio Processing on Android Phones", DAFx-24.
- Giampiccolo et al., "Wave Digital Model of the MXR Phase 90 Based on a Time-Varying Resistor
  Approximation of JFET Elements", DAFx-24.
- Massi, Manino, Bernardini, "Wave Digital Modeling of Circuits with Multiple One-Port Nonlinearities
  Based on Lipschitz-Bounded Neural Networks", DAFx-24.
- Köper & Holters, "Antialiased state trajectory neural networks for virtual analog modeling", DAFx-23.
- He, Williams, Fazenda, "Evaluating the Performance of Objective Audio Quality Metrics in Response to
  Common Audio Degradations", DAFx-25.
- Pontelli & Roman, "CLEAN2FX: Label-Conditioned Modeling for Clean-to-Effect Guitar Audio
  Transformations", DAFx-26.
- Atkinson, "Slimmable NAM: Neural Amp Models with adjustable runtime computational cost",
  NeurIPS 2025 workshop AI for Music. arXiv:2511.07470 (2 pages)

### ÉVALUATION — la critique la plus tranchante des métriques de forme d'onde (2026)
Thompson & Heilemann, "Evaluating Dynamic Range Compressor Models Using Control-Voltage
Measurements: an Approach and Dataset", DAFx-26 + soumis à IEEE/ACM TASLP. arXiv:2606.18573
https://arxiv.org/abs/2606.18573
Dataset: SSL Bus Compressor Control Voltage Dataset, Figshare DOI 10.60593/ur.d.31892026
HuggingFace: bthomp23/SSL_bus_compressor_control_voltage_dataset
- Idée: mesurer directement la TENSION DE COMMANDE de réduction de gain du matériel et l'utiliser comme
  vérité terrain, au lieu de métriques de forme d'onde sur la sortie audio.
- Argument: isoler le signal de réduction de gain à partir des seules paires entrée/sortie est un problème
  inverse MAL CONDITIONNÉ; les métriques de forme d'onde sont contaminées par des effets secondaires de
  la chaîne analogique et de capture, "even when those effects are inaudible".
- Résultat clé: "the waveform-domain metrics assigned similar errors to models that were clearly
  separated by the direct metric" — les modèles entraînés sur pertes proxy n'atteignent pas la parité
  avec un modèle entraîné directement sur le signal de commande.

### DAFx-26 — temps réel mesuré en conditions réelles (nouveau)
- Balasubramaniam & Timoney, "Real-Time Neural Audio on Apple Silicon: Benchmarking Inference
  Frameworks Under Realistic DAW Contention", DAFx-26.
- Huang, Srinivasan, van Troyer, Zappi, "Benchmarking Integrated GPU Acceleration of Real-Time Neural
  Audio Inference on Snapdragon", DAFx-26.
- Giampiccolo, Torres, de Bari, Limier, Bernardini, "A Comparative Study of Kolmogorov-Arnold Networks
  and Multi-Layer Perceptrons for Virtual Analog Modeling in Wave Digital Filters", DAFx-26.
- Chowdhury & Rau, "Performance-Oriented Wave Digital Circuit Emulation", DAFx-26.
- McClellan, "Stability Analysis of Time-Varying Virtual Analog Filters", DAFx-26.
- Oyama, "Quantifying Nonlinear Behavior in Digital Moog Ladder Filters: Cross-Implementation Comparison
  and Common-Core Ablation", DAFx-26.

### Autres
- Wright, Välimäki, Juvela, "Adversarial guitar amplifier modelling with unpaired data", ICASSP 2023.
- Carson et al., "Interpolation Filter Design for Sample Rate Independent Audio Effect RNNs",
  arXiv:2409.15884.
- Steinmetz & Reiss, "Steerable discovery of neural audio effects", arXiv:2112.02926.
- Wright & Välimäki, "Perceptual Loss Function for Neural Modelling of Audio Systems", ICASSP 2020,
  arXiv:1911.08922 — note: ils choisissent la BASSE plutôt que la guitare pour le test d'écoute car
  "preliminary listening tests indicated that they resulted in differences between the RNN models and
  the reference that were much more audible". (dans leur review.bib)

### HYSTÉRÉSIS DANS LE TRANSFO DE SORTIE D'UN AMPLI À LAMPES — meilleure référence trouvée
Massi, Mezza, Giampiccolo, Bernardini, "Deep learning-based wave digital modeling of rate-dependent
hysteretic nonlinearities for virtual analog applications", EURASIP J. Audio Speech Music Process.
2023, article 12. DOI 10.1186/s13636-023-00277-8
https://asmp-eurasipjournals.springeropen.com/articles/10.1186/s13636-023-00277-8
- Un RNN encapsulé dans un BLOC WAVE DIGITAL UN-PORT caractérisant localement les relations de
  diffusion d'ondes d'une réluctance hystérétique. Hybride WDF × apprentissage.
- Appliqué à l'ÉTAGE DE SORTIE d'un ampli guitare à lampes, transformateur non linéaire modélisé
  en multiphysique.
- CITATION: "the discrete-time simulation of circuits exhibiting rate-dependent hysteretic phenomena
  remains an open challenge."

### REVUE DE RÉFÉRENCE DU DOMAINE — verrous déclarés en 2022 (source canonique)
Vanhatalo, Legrand, Desainte-Catherine, Hanna, Brusco, Pille, Bayle,
"A Review of Neural Network-Based Emulation of Guitar Amplifiers",
Applied Sciences 12(12):5894, 9 juin 2022. DOI 10.3390/app12125894
Audio + comparaison d'architectures: https://www.math.u-bordeaux.fr/~plegra100p/NNA_AMPLI_EMU.php
Verrous déclarés (verbatim, section 4.2 et Conclusions):
- Temps réel: "they remain computationally heavy and have not been demonstrated to be able to achieve
  real-time speeds for sampling rates over 44.1 kHz" ; 44,1 kHz est "the lower bound for acceptable
  audio quality" et "is insufficient for anti-aliasing techniques based on upsampling the signals".
- Aliasing: "formal study of the aliasing introduced by neural networks in this field is lacking".
  Damskägg et al. avaient observé de l'aliasing en sortie de leur WaveNet malgré un entraînement sur
  données non aliasées, mais "this aliasing could not clearly be heard in the predictions".
- Évaluation: "there is a need for perceptually relevant, objective metrics to facilitate and automate
  evaluation. Additionally, differentiable objective metrics are also required".
  Et dans leur propre comparaison: "This highlights the divergence between the objective values and
  actual perceptual quality." (tous les modèles sauf Shallow-TCN jugés acceptables à l'écoute
  informelle alors que leurs métriques temporelles varient beaucoup).
- Paramétrique: "the exploration into parametric models, that have only been alluded to in the current
  state-of-the-art without explicit demonstration".
- Interprétabilité.
Financement: ANRT CIFRE 2020/1209 + Orosys (contexte industriel français).

### CASSIDY & DE SENA — le protocole d'écoute le plus rigoureux du domaine (VÉRIFIÉ PAR LECTURE)
Cassidy & De Sena, "Perceptual Evaluation and Genre-specific Training of Deep Neural Network Models
of a High-Gain Guitar Amplifier", DAFx-23 (Copenhague).
https://dafx.de/paper-archive/2023/DAFx23_paper_40.pdf
- Peavey 6505 (haut gain). LSTM-RNN vs WaveNet CNN, 3 jeux d'entraînement (général / rock / metal).
- 21 participants (tests 1-3), 16 (tests 4-5), tous étudiants Tonmeister (Surrey), formés à l'écoute
  critique. Salle traitée, ATH-M40X. Friedman + Wilcoxon post hoc, Bonferroni.
- MUSHRA-like : la RÉFÉRENCE CACHÉE n'obtient que 79 % en moyenne — c'est CE QUI A MOTIVÉ l'ABX.
  (Un score de référence cachée à 79 signale que le test ne discrimine plus.)
- ABX (référence étiquetée + référence cachée + 1 modèle, binomiale cumulée) :
  * à 95 % de confiance, TOUS les modèles sont distinguables de la référence ;
  * à 99 %, le LSTM-RNN rock-spécifique ne dépasse pas le seuil critique — identifié correctement
    57 % du temps seulement ;
  * par extrait : sur l'extrait aigu à haut gain (H), AUCUN modèle n'est distinguable ;
    sur l'extrait BAS GAIN (F), seul le LSTM rock-spécifique est indistinguable.
  => RÉSULTAT CONTRE-INTUITIF ET IMPORTANT : ce sont les extraits proches de "l'edge of breakup"
     (bas gain) qui révèlent le plus de différences, pas les extraits saturés.
- Préférence en contexte de mix (batterie + basse) : le VRAI ampli n'est préféré à AUCUN modèle.
  Un modèle est même jugé PLUS RÉALISTE que l'ampli réel (p=0,0383 ajusté).
- PIÈGE MÉTHODOLOGIQUE MAJEUR : l'ESR est la fonction de perte d'entraînement mais AUCUNE VALEUR
  D'ESR N'EST RAPPORTÉE dans tout l'article (vérifié par grep sur le texte complet). Le papier
  d'écoute le plus rigoureux du domaine est donc inutilisable pour calibrer un seuil ESR.
- Données d'entraînement : fichier de Wright et al. (HT-1), issu des bases Fraunhofer IDMT,
  5 min 40 s, moitié basse.

### SEUIL D'AUDIBILITÉ DE L'ESR : N'EXISTE PAS (triple confirmation)
- Vérifié indépendamment par moi (recherche dédiée) ET par un agent de recherche ayant lu les textes
  intégraux de Damskägg SMC-19, Wright DAFx-19, Damskägg ICASSP-19, Wright Applied Sci. 2020.
- AUCUN de ces quatre textes n'énonce de seuil numérique.
- Origine probable du folklore « ESR < 1 % » : Wright et al., Applied Sciences 10(3):766, 2020, §4.3
  écrit "for the two larger WaveNet models a loss of under 1% is achieved", et le §4.4 enchaîne
  immédiatement sur "To investigate the impact of the ESR on the perceptual quality of the models,
  listening tests were conducted." La juxtaposition a été lue comme un seuil. Ce n'en est pas un.
- CONTRE-EXEMPLE DANS LE MÊME ARTICLE : sur le HT5M, RNN-32 a un ESR PIRE (4,2 %) que WaveNet1
  (2,8 %) mais un score MUSHRA MEILLEUR (80 contre 75). Inversion de rang ESR/perception.
- Wright 2020 : MUSHRA, 14 sujets dont 5 exclus (9 retenus). Damskägg ICASSP-19 : MUSHRA, 14 sujets,
  10 retenus. Wright & Välimäki ICASSP 2020 : MUSHRA, 14 sujets.
- PIÈGE DE COMPARABILITÉ : les ESR publiés ne sont PAS comparables entre articles, car la
  pré-accentuation diffère — Damskägg SMC-19 entraîne avec H(z)=1−0,95z⁻¹ mais rapporte l'ESR de test
  SANS pré-accentuation ; Wright DAFx-19 utilise 1−0,85z⁻¹ ; Wright 2020 §4 un A-weighting passe-bas
  + terme DC ; Mikkonen 2024 aucune. Un « seuil » universel est donc structurellement impossible.
- Wright & Välimäki, "Perceptual Loss Function for Neural Modelling of Audio Systems", ICASSP 2020,
  arXiv:1911.08922 (ID VÉRIFIÉ PAR MOI : c'est bien ce papier, Wright & Välimäki, eess.AS).
  Le filtre A-weighting donne le meilleur score MUSHRA (79 et 86 vs 74/77 sans pré-accentuation) ET
  une erreur spectrale ~5 dB plus basse sur 1-2 kHz, zone de sensibilité auditive maximale.

### MÉTRIQUES PERCEPTUELLES EMPRUNTÉES — recherche largement INFRUCTUEUSE
PEAQ, PEMO-Q, ViSQOL, CDPAM : AUCUN papier de modélisation VA neuronale d'ampli/pédale ne les utilise.
Seules FAD (NablAFx, 4 embeddings) et NMR (lignée antialiasing Carson/Välimäki) ont fait le pont.
Le seuil NMR de −10 dB vient de Lehtonen, Pekonen, Välimäki, JASA 132(4):2721-2733, 2012
(audibilité de l'aliasing dans des dents de scie) — c'est un seuil d'AUDIBILITÉ D'ALIASING,
transposé par commodité, PAS un seuil de fidélité de modèle.

### LA RÉFÉRENCE DÉCISIVE SUR LES RÉGLAGES NON VUS (VÉRIFIÉE, TABLEAUX LUS)
Miklánek, Wright, Välimäki, Schimmel, "Neural Grey-Box Guitar Amplifier Modelling With Limited Data",
DAFx-23 (Copenhague), pp. 151-158.  https://www.dafx.de/paper-archive/2023/DAFx23_paper_52.pdf
Données: Marshall JVM 410H Dataset, Zenodo DOI 10.5281/zenodo.7970723, CC BY 4.0
Code/écoute: https://stepanmk.github.io/grey-box-amp/  https://github.com/stepanmk/grey-box-amp
- Méthode: RNN (LSTM) + tone stack DÉRIVÉ PAR LA DK-METHOD (analyse nodale de Kirchhoff discrétisée),
  implémenté par échantillonnage fréquentiel (frequency sampling) pour un entraînement efficace.
  => hybride boîte blanche (le tone stack, circuit LINÉAIRE paramétrique) × boîte noire (le reste).
- Montage: Marshall JVM 410H, canal OD1 (fort gain). SEULS bass/middle/treble sont rendus pilotables;
  volume canal à 10, gain/master/resonance/presence à 5. Jeu de 6 min (4 train / 1 val / 1 test),
  sons issus des bases IDMT. Chaque bouton varié de 0 à 10 par pas de 2 (les deux autres à 5),
  plus 3 signaux tous-boutons-égaux (0, 5, 10) => jusqu'à 21 permutations.
  9 signaux SUPPLÉMENTAIRES capturés sur le sous-ensemble de test pour des réglages NON VUS.
- TABLE 4 (ESR / STFT, Test puis Test NON VU, durée d'entraînement) :
    RNN1  (boîte noire, 4 min, 1 perm.) : 0,020 / 0,793  ->  1,748 / 2,573   <-- EFFONDREMENT
    RNN3  (12 min)                      : 0,025 / 0,933  ->  1,446 / 2,511
    RNN15 (60 min)                      : 0,017 / 0,743  ->  0,040 / 0,953
    RNN21 (84 min)                      : 0,015 / 0,693  ->  0,039 / 0,893
    TS1   (gris, 4 min, 1 perm.)        : 0,011 / 0,760  ->  0,048 / 0,854
    TS3   (gris, 12 min)                : 0,020 / 0,764  ->  0,023 / 0,715   <-- MEILLEUR non vu
    TS15  (60 min)                      : 0,029 / 0,780  ->  0,038 / 0,805
    TS21  (84 min)                      : 0,028 / 0,924  ->  0,039 / 0,955
  => Le RNN boîte noire entraîné sur UN SEUL réglage a un ESR de 1,748 sur réglages non vus :
     l'énergie d'erreur dépasse celle du signal. Ce n'est pas une dégradation, c'est un effondrement.
  => TS1 (4 min, 1 réglage) bat RNN21 (84 min, 21 réglages) sur l'erreur STFT non vue.
- ÉCOUTE MUSHRA, 14 participants, 1 exclu (référence cachée < 90 sur plus de 15 % des essais) => 13.
  Deux réglages non vus testés :
    Scénario 1 — cb=0,1 cm=0,3 ct=0,7 (ENTRE les pas d'entraînement = interpolation, cas « facile ») :
      REF 98,8±0,7 | TS3 94,7±1,8 | RNN21 94,0±1,9 | TS1 87,0±3,0 | RNN1 47,2±4,2 | RNN3 29,5±3,8 | ANC 1,4±0,8
    Scénario 2 — cb=1 cm=0 ct=0 (COMBINAISON EXTRÊME, jamais vue conjointement) :
      REF 99,4±0,6 | TS1 92,1±2,1 | TS3 87,1±2,4 | RNN21 46,8±3,9 | RNN3 21,3±3,1 | RNN1 17,8±3,3 | ANC 2,5±1,9
  => LE RÉSULTAT CENTRAL : RNN21 passe de 94,0 (« Excellent ») à 46,8 (« Fair ») entre interpolation
     et combinaison extrême, alors que TS1 (4 min, UN réglage) reste à 92,1 (« Excellent »).
     La généralisation aux réglages non vus n'est donc PAS un problème de quantité de données :
     c'est un problème de STRUCTURE. 84 minutes de boîte noire ne remplacent pas un modèle de circuit
     du tone stack obtenu en 4 minutes.
  => Noter aussi RNN3 (12 min) NOTÉ PIRE que RNN1 (4 min) au scénario 1 : 29,5 contre 47,2.
- COÛT (Table 3, implémentation C++, RTF à 44,1 kHz ; RTF > 1 = plus rapide que le temps réel) :
    RNN de référence 10 417 paramètres, RTF 21,28 ; TS proposé (domaine fréquentiel) 7 208 paramètres,
    RTF 19,61. La version « recursive » du TS a le même nombre de paramètres mais un pas
    d'entraînement de 14,17 s contre 0,45 s pour la version par échantillonnage fréquentiel
    (et 26,59 s pour le tone stack différentiable antérieur) => gain d'ENTRAÎNEMENT d'un facteur ~59.
- ANOMALIE DÉCLARÉE, NON EXPLIQUÉE : « for the models TS15 and TS21, the additional data seemed to
  impact the model training negatively. An abrupt increase in the validation loss occurs after a few
  epochs, and this behaviour was observed consistently across different training runs. »
  Hypothèse avancée par les auteurs : les tapers de potentiomètre entraînables seraient sensibles aux
  erreurs de positionnement manuel des boutons lors de la capture.
  => MÊME PHÉNOMÈNE que Mikkonen et al. 2024 (« continuous sampling is a suboptimal choice ») :
     DEUX équipes observent indépendamment que DAVANTAGE de diversité de réglages DÉGRADE le modèle,
     et AUCUNE ne l'explique. C'est un verrou ouvert et peu visible.
- Limites déclarées : « our study was limited to a single amplifier » ; les amplis à très fort gain
  pourraient exiger des RNN plus gros.

### CORRECTION IMPORTANTE À MON PROPRE CONSTAT SUR LES GRAINES
Il EXISTE un travail VA qui rapporte une variance inter-graines systématique, et l'équipe l'a déjà :
Sato & Smith, "Aliasing Reduction in Neural Amp Modeling by Smoothing Activations", DAFx-25,
arXiv:2505.04082 (= sato2025aliasing dans paper/bibliography.bib). VÉRIFIÉ PAR LECTURE :
- « we conducted experiments using 100 deterministic training seeds for each function » ;
  « each model configuration was tested with 100 unique deterministic seeds ».
- Table 1 comporte les colonnes ASR, ASR std, ESR, ESR std : moyenne ET écart-type sur les graines.
- 124 configurations d'activation × 100 graines = 12 400 modèles, puis 2 fonctions × 100 α × 100 graines
  = 20 000 modèles. A100, 1-2 min par modèle, <= 10 époques.
- Mise en garde explicite des auteurs (verbatim) : « these minimum values represent the best outcomes
  from random initialization rather than consistently achievable performance » et « we consider the
  average performance metrics to be more reliable indicators of practical utility ».
- PORTÉE LIMITÉE : UN SEUL appareil modélisé (l'ampli à lampes boutique fourni en exemple par NAM,
  fichier d'entrée standard 3 min 10 s), UNE SEULE architecture (WaveNet/TCN, 18 couches, canal 16),
  et l'objet de l'étude est la comparaison de FONCTIONS D'ACTIVATION, pas la dispersion elle-même.
  Aucune comparaison de la dispersion inter-graines à l'écart modèle/appareil réel.
=> Formulation correcte : un seul travail VA rapporte une variance inter-graines ; il le fait comme
   précaution méthodologique dans une étude d'activation, pas comme objet d'étude. La question posée
   par le projet (la dispersion inter-graines est-elle du même ordre que l'erreur de modélisation ?)
   n'est traitée nulle part.

### DONNÉES — points saillants non dans leurs bibs
- ToneTwist AFx : « the largest collection (40 devices at the time of writing) », 4 catégories
  (analog / analog parametric / digital / digital parametric), WAV 48 kHz 32-bit float, code MIT,
  audio sur Zenodo (un DOI par appareil). AUCUNE durée totale agrégée n'est publiée par les auteurs.
  Leur propre aveu : « there is no clear state of the art yet that identifies the best-performing approach ».
  Leurs critiques des jeux antérieurs (verbatim) : IDMT-SMT-Audio-Effects « limited to single notes and
  chords from two electric guitars and two electric basses » ; EGFxSet « only features single notes from
  one electric guitar [...] unsuitable due to high background noise and other artifacts ».
- EGFxSet : Pedroza, Meza, Roman, ISMIR 2022 LATE-BREAKING DEMO (pas un article complet),
  Zenodo 10.5281/zenodo.7044411, CC BY 4.0, 5,8 Go. Stratocaster, 5 configurations de micros,
  12 pédales MATÉRIELLES réelles. => la référence exacte dans review.bib est à vérifier (statut LBD).
- IDMT-SMT-Audio-Effects : Stein, Abeßer, Dittmar, Schuller, AES 128th Convention, Londres, mai 2010
  (PAS DAFx). Zenodo 10.5281/zenodo.7544032. 55 044 WAV, 44,1 kHz/16 bits MONO, ~30 h,
  2 guitares + 2 basses, 10 effets NUMÉRIQUES. Licence CC BY-NC-ND 4.0 (dérivés non redistribuables).
- IDMT-SMT-Guitar : Kehling et al., DAFx 2014, Erlangen. Zenodo 7544110. Signaux PROPRES uniquement,
  donc inutilisable tel quel en VA apparié ; utile comme source DI. CC BY-NC-ND 4.0.
- GuitarSet : Xi, Bittner, Pauwels, Ye, Bello, ISMIR 2018. Zenodo 10.5281/zenodo.3371780, CC BY 4.0.
  Pas de traitement ampli/effet.
- GUITAR-FX-DIST : Comunità, Stowell, Reiss, JAES 69(7/8):594-604, 2021 (arXiv:2012.03216).
  ~305 h, 14 effets overdrive/distortion/fuzz, 4 sous-jeux discret/continu × mono/poly.
- SignalTrain LA2A : Hawley, Colburn, Mimilakis, AES 147th Convention 2019, arXiv:1905.11928.
  Zenodo 3824876 (v1.1), 20,7 Go, CC BY 4.0. Teletronix LA-2A, Peak Reduction balayé.
- Solid State Bus-Comp : 175 morceaux × 220 combinaisons de paramètres = 2 528 h. Compresseur
  SSL 500 G-Bus réel, capture automatisée par ReaScript. Le plus dense/large en balayage de réglages.
- Guitar-TECHS : Pedroza, Abreu, Corey, Roman, ICASSP 2025, arXiv:2501.03720, Zenodo 14963133.
  3 joueurs, > 5 h, micros égocentrique + exocentrique, matériel varié — MAIS conçu pour la
  transcription, pas de paires sec/traité VA.
- GOAT : Loth, Sarmento, Sarkar, Guo, Barthet, Sandler, ISMIR 2025, arXiv:2509.22655, Zenodo 17706552.
- Hinrichs, Gerkens, Ostermann et al., EvoMUSART 2022 + EURASIP JASMP 2022(28) : grille 2 paramètres
  au pas de 0,05, 67 200 échantillons. Disponibilité publique NON VÉRIFIÉE.
- NAM / ToneHunt / TONE3000 : héberge des FICHIERS DE MODÈLES (.nam), PAS un corpus audio.
  Chaque contributeur ré-amplifie en privé un DI standard et ne publie que le modèle entraîné.
  => AUCUN corpus audio brut partagé et traçable. Licences hétérogènes. Open-Amp hérite de ce problème.
- « DiffVintage » : n'existe pas (recherche explicite infructueuse).

### RIGUEUR MÉTHODOLOGIQUE — constats négatifs confirmés indépendamment
- Aucun « reproducibility challenge » VA, aucune réplication indépendante d'un résultat VA publié.
- Aucun leaderboard / benchmark partagé multi-équipes. NablAFx est le plus proche mais mono-équipe,
  et son GitHub ne mentionne ni « benchmark » ni « leaderboard ».
- NablAFx / ToneTwist : AUCUNE mention de « seed », chaque configuration entraînée UNE SEULE FOIS
  (« Models are trained for a maximum of 15k steps »). L'écart-type publié est calculé ENTRE APPAREILS.
- Aucune recommandation de taille d'échantillon spécifique au VA (graines, sujets, minutes).
- Nombre d'auditeurs : la pratique VA effective est de 10 à 15 retenus (Damskägg ICASSP19 : 14 recrutés,
  10 retenus ; Wright 2020 : 14 recrutés, 9 retenus ; Miklánek DAFx23 : 14 recrutés, 13 retenus ;
  Mikkonen 2024 : 15, aucun exclu ; Comunità 2025 : 27 ; Cassidy 2023 : 21 puis 16 ;
  Juvela ICASSP23 : 30 ; Bourdin DAFx25 : 29 recrutés, 14 retenus ; Carson DAFx25 : 16 recrutés,
  12 retenus). ITU-R BS.1116-1 (norme soeur de BS.1534) indique « data from 20 subjects is often
  sufficient for drawing appropriate conclusions » => la pratique du champ est souvent SOUS ce repère.
- Littérature ML générale transférable sur les graines : Reimers & Gurevych, EMNLP 2017
  (arXiv:1707.09861, 50 000 LSTM) ; Picard, arXiv:2109.08203 (2021) ; Bouthillier et al., MLSys 2021
  (arXiv:2103.03098) ; Henderson et al., AAAI 2018 ; Colas, Sigaud, Oudeyer, arXiv:1806.08295.
- MIR/validité : Sturm, IEEE Trans. Multimedia 16(6):1636-1644, 2014 (« horse ») ;
  Sturm & Flexer, ISMIR 2023 / arXiv:2301.01578 (quatre types de validité) ;
  Flexer, J. New Music Research 35(2):113-120, 2006 ; Urbano, Schedl, Serra, JIIS 41:345-369, 2013 ;
  McFee et al., IEEE SPM 36(1):128-137, 2019 (reproductibilité du pipeline MIR).
  Bogdanov = infrastructures/datasets, PAS critique de validité : mauvaise piste, écartée.
- Voisinage : Le Roux et al., « SDR – Half-baked or Well Done? », ICASSP 2019 — une métrique standard
  peut biaiser des classements entiers. Zieliński, Rumsey, Bech, JAES 56(6):427-451, 2008 — biais des
  tests d'écoute modernes. Schoeffler et al., J. Open Research Software 6(1):8, 2018 — webMUSHRA.

## RECHERCHES INFRUCTUEUSES (à signaler comme telles)
1. SAG D'ALIMENTATION: aucun article de modélisation neuronale VA ne le prend pour objet. Deux requêtes
   distinctes → seuls des BREVETS analogiques des années 1990 (US5434536, US5636284, US5734725) décrivent
   explicitement le mécanisme (redressement + filtre RC + décalage de polarisation).
   Le concept est traité implicitement sous "compression/long-term dynamics" (KLANN, SSM, compresseurs).
2. DÉRIVE THERMIQUE: idem, rien en VA neuronal. La littérature sur la mémoire longue thermique existe en
   RF/power amplifiers (self-heating, trapping, cellules RC thermiques), pas en audio VA.
3. VARIANCE INTER-GRAINES: aucun article de VA neuronal ne rapporte d'écart-type sur plusieurs graines
   d'initialisation. Simionato & Fasciani utilisent Friedman/Wilcoxon ENTRE ARCHITECTURES;
   Mikkonen et al. répètent sur des TIRAGES DE DONNÉES (k=1..5) et agrègent par médiane.
   Comunità et al. parlent de "variance" au sens de la dispersion des notes MUSHRA entre effets/appareils.
   => la question posée par le projet (dispersion due à la seule graine) n'est PAS traitée dans la littérature.
4. BENCHMARK / LEADERBOARD VA amplis: n'existe pas. Le 1er DAFx Challenge porte sur les plaques.
5. OOD guitare/micro/niveau: seul ToneTwist AFx sépare explicitement instruments train/test; le niveau
   d'entrée n'est traité que par UN exemple du test MUSHRA de Comunità (guitare à très bas niveau).
   Aucune étude dédiée à la robustesse au niveau d'entrée.
