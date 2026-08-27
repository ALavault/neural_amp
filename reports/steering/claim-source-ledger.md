# Registre claims → sources — steering R1

| ID | Claim vérifié | Source primaire | Date | Accès / portée |
|---|---|---|---|---|
| C1 | M4 est NO-GO ; scores, recovery et limites | [FINAL_AUDIT](../FINAL_AUDIT.md), [M4_CORE](../M4_CORE.md), [M4_RECOVERY](../M4_RECOVERY.md) | 2026-08-27 | Artefacts locaux, commit `b38d16f` |
| C2 | M4 optimise MSE brute + 0,0005 MR-STFT ; 200 updates | [`configs/training/m4_smoke.yaml`](../../configs/training/m4_smoke.yaml), [`run_m4_smoke.py`](../../scripts/run_m4_smoke.py) | 2026-08-27 | Code et config locaux |
| C3 | Énergie train Big Muff 0,001548 contre Fulltone 0,012501 | WAV préparés dans `datasets/raw/internal_m4/` | 2026-08-27 | Calcul RMS local ; données non redistribuées |
| C4 | RF résidu 31 ; RF A2 6 347 | [`residual.py`](../../src/fssr_nam/models/residual.py), [`architecture.json`](../../experiments/summaries/m2_a2_architecture/architecture.json) | 2026-08-27 | Code et synthèse locaux |
| C5 | Big Muff publié : 5:42 train, loss ESR préaccentuée + DC, 20 h, ESR LSTM-64 4,1 % | [Wright, Damskägg & Välimäki, “Real-Time Black-Box Modelling With Recurrent Neural Networks”](https://dafx.de/paper-archive/2019/DAFx2019_paper_43.pdf) | 2019 | PDF DAFx primaire |
| C6 | Le code Wright expose `ESRPre`, DC et préaccentuation passe-haut | [Automated-GuitarAmpModelling](https://github.com/Alec-Wright/Automated-GuitarAmpModelling/blob/e3146386b0fd0b562bc393231be3a5938cf9feac/dist_model_recnet.py) | consulté 2026-08-27 | Dépôt auteur figé au commit `e314638` |
| C7 | Même Big Muff/réglage : meilleur ESR 0,1076 ; gray-box 0,5869–0,7042 | [Appendice de Comunità et al.](https://github.com/mcomunita/nnlinafx-supp-material/blob/bbcb628afd9418777df254e901f252d4b17e37f4/Differentiable_Black_box_and_Gray_box_Modeling_of_Nonlinear_Audio_Effects___Appendix___Arxiv.pdf) | 2025 | PDF au commit `bbcb628` |
| C8 | Étude 16 dispositifs : S4/TFiLM les plus robustes ; gray-box en retrait ; LR par bloc important ; 15k updates non-LSTM | [Comunità, Steinmetz & Reiss, Frontiers](https://www.frontiersin.org/journals/signal-processing/articles/10.3389/frsip.2025.1580395/full) | 2025 | Article revu par les pairs |
| C9 | TFiLM aide les dépendances longues de fuzz sans étendre le backbone court | [Comunità et al., ICASSP](https://mcomunita.github.io/files/comunita2023gcntfilm-paper.pdf) | 2023 | Manuscrit auteur de l'article IEEE |
| C10 | L'identification block-oriented étagée réduit ambiguïtés et minima locaux | [Eichas, Möller & Zölzer, DAFx](https://dafx.de/paper-archive/2015/DAFx-15_submission_21.pdf) | 2015 | PDF DAFx primaire |
| C11 | Le Big Muff comporte deux étages de clipping à diodes en cascade | [Wright et al., DAFx-19](https://dafx.de/paper-archive/2019/DAFx2019_paper_43.pdf) | 2019 | Description du dispositif mesuré |
| C12 | Un modèle DDSP multi-étages rapporte une fidélité comparable avec <10 % des opérations | [Yeh et al., “DDSP Guitar Amp”](https://arxiv.org/abs/2408.11405) | 2024 | Préprint ; autre amplificateur, pas Big Muff |
| C13 | Un faible ASR peut provenir d'une sortie quasi silencieuse | [Sato & Smith, “Aliasing Reduction…”](https://dafx.de/paper-archive/2025/DAFx25_paper_50.pdf) | 2025 | PDF DAFx primaire |
| C14 | A2 Full figé utilise 23 couches, LeakyReLU, MSE implicite + MR-STFT 0,0005 et sélection ESR | [configuration NAM au commit figé](https://github.com/sdatkinson/neural-amp-modeler/blob/f26112906de06ec6b796ad6d1982e29eed83144e/nam/train/_resources/config_model_packed.json) | commit consulté 2026-08-27 | Code officiel figé ; MSE par défaut vérifiée localement dans `lightning_module.py` |

## Contradictions et réserves

- C5 et C7 donnent 0,041 et 0,1076 sur Big Muff avec des architectures et
  protocoles différents. Ce n'est pas une contradiction : le premier entraîne un
  LSTM conditionné sur cinq réglages pendant 20 h ; le second compare de nombreux
  modèles avec un protocole commun. Le steering retient donc une borne de compétence
  tolérante de 0,15, pas la réplication exacte d'un score unique.
- Les résultats Fulltone de C7 emploient drive 10, tandis que M4 emploie `O050`.
  Ils documentent l'apprenabilité du dispositif, pas un seuil comparable.
- C12 reste une indication de faisabilité pour la structure en cascade ; sa preuve
  ne se transfère pas automatiquement aux pédales de ce dépôt.
