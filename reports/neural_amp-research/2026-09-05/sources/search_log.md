# Journal de recherche — 5 septembre 2026

## Stratégie

Revue exploratoire sur modélisation d'effets audio, mémoire dynamique, entraînement,
mesures et contrôle de validité. Période 2016-2026; anglais et sources primaires.
Recherches web de secours : `paper-search` absent du PATH et emplacement de dépôt
indiqué par le skill absent. Aucun téléchargement d'audio, checkpoint ou données scellées.
Aucune dépendance du projet d'entraînement n'a été modifiée.

Sources interrogées par le moteur : arXiv, DAFx, PMLR, DBLP; chaînes de citations et
notices institutionnelles Aalto/Edinburgh; ISMIR pour données. Les recherches ne sont
pas des exports API exhaustifs. Les nombres bruts par requête sont non normalisés,
donc non revendiqués comme un décompte PRISMA.

## Requêtes principales exécutées

### Ronde 1 — périmètre et état récent

- neural audio effect modeling state space models S4 TFiLM amplifier 2025 2026
- site.dafx.de neural guitar amplifier modeling recurrent neural network Wright 2019 2020
- site.arxiv.org audio effects modeling dataset ToneTwist NABLA-Fx
- site.dafx.de "neural" "state" "2024" modeling
- site.arxiv.org audio effects modeling state space selective S4 S6
- site.dafx.de differentiable audio modeling loss oversampling recurrent
- "audio effects" "state-space" modeling S4
- "audio effects" "selective" "state" modeling
- "guitar amplifier" "neural" "2025" modeling
- "audio effects" "differentiable" "2024" modeling

### Ronde 2 — mémoire, modèles et métriques

- site.arxiv.org "audio effects" "state-space" "Comunità"
- "KLANN" "Koopman" arxiv
- site.dafx.de "Antiderivative Antialiasing for Recurrent Neural Networks"
- site.arxiv.org "audio effects" "loss" modeling perceptual
- site.arxiv.org "KLANN"
- site.arxiv.org "State-Space" "Audio Effect"
- site.arxiv.org "Black-Box Modeling" "Audio" "2024"
- site.arxiv.org "audio effects" "loss functions"
- "KLANN" audio site:arxiv.org/abs
- "Simionato" "Fasciani" "state" audio modeling 2024
- "Yin" "2024" "audio effects" S4
- "audio" "loss functions" "Wright" "2020"
- "KLANN: Linearising" authors
- "Efficient Neural Networks for Real-time Modeling of Analog Dynamic Range Compression" arxiv
- "Real-Time Guitar Amplifier Emulation with Deep Learning" Wright
- "Real-Time Modeling of Audio Distortion Circuits with Deep Learning"
- KLANN Linearising Long-Term Dynamics Koopman Networks Yu 2024
- site.dafx.de "Modelling of Nonlinear State-Space Systems"
- site.arxiv.org "DDSP: Differentiable Digital Signal Processing"
- site.arxiv.org "Differentiable" "Gray-box" "Virtual Analog"
- "Differentiable" "Gray-Box" "Virtual Analog Modeling"
- "Deep Learning for Tube Amplifier Emulation"
- "Modelling of Nonlinear State-Space Systems Using a Deep Neural Network"
- "Neural Ordinary Differential Equations for Black-box Modeling of Audio Effects"
- Parker Esqueda Bergner 2019 state trajectory network dafx
- neural ordinary differential equations audio effects modeling Esqueda 2021
- differentiable grey box virtual analog modeling audio Wiener Hammerstein
- EGFxset guitar effects dataset 2023
- "Neural Ordinary Differential Equations" "Audio Effects"
- site.arxiv.org "differentiable" "virtual analog" grey box
- site.arxiv.org "Differentiable All-pole Filters"
- site.dafx.de "Differentiable IIR Filters"

### Ronde 3 — transfert structurel et vérification

- "Temporal FiLM" arxiv
- "Unbiasing Truncated Backpropagation Through Time" arxiv
- "Adaptive Truncation" "Backpropagation" Aicher
- "Nonlinear state-space identification" "encoder" Beintema 2021
- site.arxiv.org "Temporal FiLM" neural audio
- site.arxiv.org "Resurrecting Recurrent Neural Networks for Long Sequences"
- site.arxiv.org "Mamba: Linear-Time Sequence Modeling with Selective State Spaces"
- site.arxiv.org "WaveNet: A Generative Model for Raw Audio"
- Temporal FiLM Capturing Long-Range Sequence Dependencies Feature-Wise Modulations 2019 arxiv
- site.arxiv.org 1905.07473
- site.dafx.de "oversampling" "neural" "2023"
- site.dafx.de "training signals" "neural" amplifier
- site.dafx.de "Sample Rate Independent Recurrent Neural Networks" authors
- site.dafx.de "Perceptual Evaluation and Genre-specific Training" authors
- site.ismir.net "EGFxSet" Pedroza
- site.arxiv.org virtual analog gray box modeling differentiable signal processing
- "Antiderivative Antialiasing for Recurrent Neural Networks" authors
- "Adaptively Truncating Backpropagation Through Time to Control Gradient Bias"
- site.dblp.org "Differentiable Black-box and Gray-box Modeling of Nonlinear Audio Effects"
- site.dblp.org "Modeling Time-Variant Responses of Optical Compressors"
- site.dblp.org "Temporal FiLM"

### Ronde 4 — actualisation 2026

- site.arxiv.org "guitar amplifier" "2026"
- site.arxiv.org "audio effects modeling" "2026"
- "2607.10086"

## Screening et exclusions

33 publications distinctes retenues; liste finale dans `corpus.json`.
Raisons d'exclusion appliquées : autre tâche sans mécanisme pertinent; doublon;
source secondaire générée; notice commerciale; sujet sans lien avec audio ou identification.

Exemples exclus :
- Selective Structured State-Spaces for Long-Form Video Understanding : vision, hors périmètre.
- Audio Mamba: Selective State Spaces for Self-Supervised Audio Representations : reconnaissance, pas émulation.
- Spiking Structured State Space Model for Monaural Speech Enhancement : débruitage, intérêt indirect insuffisant ici.
- Text2FX / Diff-MST : contrôle/style transfer, hors question de fidélité dry/wet.
- Horizontal Attacks against ECC et Horizontal SCA... : homonyme Klann, hors domaine.
- DENT-DDSP : reconnaissance vocale bruitée, hors périmètre retenu.
- A Generative Model for Raw Audio Using Transformer Architectures : non nécessaire au transfert choisi.
- Revues automatiques, ResearchGate et agrégateurs : servent au repérage seulement; source primaire recherchée.
- Prépublication 2502.14405 et article Frontiers 2025 : un seul item conservé.
- Doublons arXiv / DBLP / dépôts institutionnels : fusionnés par titre et auteurs.

Limites : pas de double screening, pas de compte exhaustif des résultats bruts,
pas de métrique de citations utilisée comme score de qualité, pas de demande de texte aux auteurs.
Cette transparence interdit de présenter la note comme une revue systématique exhaustive.
