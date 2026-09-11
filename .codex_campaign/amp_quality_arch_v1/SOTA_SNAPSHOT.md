# Snapshot état de l'art — 2026-08-28

## Portée

Ce snapshot sélectionne des comparateurs d'architecture qui peuvent être
réentraînés sur les mêmes paires 48 kHz et évalués par le même code. Il ne
transcrit pas les scores publiés comme s'ils étaient comparables aux scores de
la campagne. La comparaison finale portera uniquement sur les reproductions
locales.

Le terme « état de l'art » est donc limité au plus fort des comparateurs ouverts
et reproductibles ci-dessous. Il ne signifie pas que toutes les solutions
propriétaires ou tous les dispositifs analogiques existants ont été couverts.

## Sources primaires consultées

1. **NAM A2 Full.** Implémentation et format officiels NAM, trainer `v0.13.0`
   au commit `f26112906de06ec6b796ad6d1982e29eed83144e` et NAM Core `v0.5.4`
   au commit `1f42f88535884450104b8711d7595019afa0495b` :
   <https://github.com/sdatkinson/neural-amp-modeler> et
   <https://github.com/sdatkinson/NeuralAmpModelerCore>. A2 est la baseline
   industrielle prioritaire et possède déjà un chemin C++ épinglé dans le dépôt.
2. **LSTM Wright.** Wright, Damskägg, Juvela et Välimäki, *Real-Time Guitar
   Amplifier Emulation with Deep Learning* :
   <https://doi.org/10.3390/app10030766>. Code officiel épinglé au commit
   `e3146386b0fd0b562bc393231be3a5938cf9feac` :
   <https://github.com/Alec-Wright/Automated-GuitarAmpModelling>. Cette famille
   reste un contrôle récurrent fort pour les distorsions.
3. **TCN-TFiLM et S4-TFiLM.** Comunità et al., *Differentiable black-box and
   gray-box modeling of nonlinear audio effects* :
   <https://doi.org/10.3389/frsip.2025.1580395>. Le panel multi-dispositif
   rapporte un avantage global de S4-TFiLM et un bénéfice robuste de TFiLM sur
   TCN, GCN et S4. Code NablAFx épinglé au commit
   `045db6e7d6087151c7e3a264844bd8c4eafc885c` :
   <https://github.com/mcomunita/nablafx>.
4. **S6 sélectif.** Simionato et Fasciani, *Comparative Study of State-based
   Neural Networks for Virtual Analog Audio Effects Modeling* :
   <https://arxiv.org/abs/2405.04124>, et *Modeling Time-Variant Responses of
   Optical Compressors with Selective State Space Models* :
   <https://arxiv.org/abs/2408.12549>. Le code primaire est identifié au commit
   distant `15098cc7941d6518d9672cba4d767ba15f986c96` :
   <https://github.com/RiccardoVib/A-Comparative-Study-State-Based>. La variante
   compresseur la plus directement pertinente est épinglée au commit distant
   `bd787ecc60024cf364efd4cef07f692af05277e6` :
   <https://github.com/RiccardoVib/Optical-DRC-with-Selective-SSMs>.
5. **Antialiasing.** Sato et Smith, *Aliasing Reduction in Neural Amp Modeling
   by Smoothing Activations* :
   <https://www.dafx.de/paper-archive/2025/DAFx25_paper_50.pdf>. Cette source
   justifie les gardes anti-silence, mais son résultat ne remplace pas la gate
   synthétique `FSSR-QUALITY-AA-v2` déjà acquise.

## Comparateurs retenus après audit exécutable

| ID gelable | Famille | Variante de départ | Rôle | Reproductibilité locale |
|---|---|---|---|---|
| `nam_a2_full` | WaveNet A2 | 23 couches, 8 canaux | baseline prioritaire | trainer et C++ déjà épinglés |
| `wright_lstm64` | LSTM | 1×64 + tête linéaire + skip | contrôle récurrent distorsion | modèle et loader locaux existants |
| `nablafx_tcn_tfilm` | TCN + TFiLM | RF 133333/118097, 16 canaux, variantes S/L départagées sur validation | convolution longue + TFiLM | adapter local en parité source |
| `nablafx_s4_tfilm` | S4 + TFiLM | variantes S/L départagées sur validation | meilleur panel publié multi-effets | code/configs NablAFx présents |

Pour TCN et S4, la variante S/L ne peut être choisie qu'avec la validation
`INTERNAL_DEV`. Une variante ne peut pas être ajoutée après observation d'un
test. Le « meilleur comparateur » final est celui qui minimise l'ESR agrégée de
validation selon la règle préenregistrée, jamais celui qui paraît le plus faible
sur un dispositif donné.

## Exclusions documentées

- Le GCN-TFiLM n'est pas un sixième comparateur : le panel primaire conclut que
  son surcoût face à TCN-TFiLM est difficile à justifier lorsque TFiLM est actif.
  Il peut néanmoins inspirer un hybride candidat si un test mécaniste le motive.
- Le micro-TCN original est couvert par le comparateur TCN-TFiLM plus récent.
- Les GAN DAFx-2024 modifient surtout l'objectif d'entraînement autour d'un
  générateur WaveNet; ils ne constituent pas ici une architecture de runtime
  distincte. Une loss adversariale peut être étudiée uniquement dans le budget
  de loss préenregistré.
- Les modèles gray-box NablAFx historiques restent des ablations mécanistes, pas
  des comparateurs SOTA principaux, car leur performance médiane publiée est
  inférieure aux familles black-box sur le panel large.
- Le S6 sélectif a été audité puis retiré avant gel, sans score audio. Au commit
  primaire épinglé, `Models.py` référence une classe `S6` non importée et
  `Mamba.py` n'est pas syntaxiquement/exécutivement cohérent; l'environnement
  requis impose en outre TensorFlow 2.15. Une adaptation PyTorch réparant ces
  ambiguïtés ne serait plus une reproduction loyale du comparateur publié.
- Les systèmes propriétaires sans code et sans procédure de réentraînement sur
  les mêmes données sont hors comparaison quantitative.

## Résolutions du preflight

- La dépendance optionnelle `rational` n'est pas installée : les sous-ensembles
  tanh nécessaires sont adaptés localement et testés contre les sources
  épinglées; auraloss 0.4.0 est épinglé séparément pour la loss exacte.
- Les chemins différables FFT S4 et leur récurrence causale sont numériquement
  équivalents; les adapters retardés passent reset, causalité et blocs arbitraires.
- Les anticipations TFiLM sont déclarées et alignées; elles excluent les
  comparateurs NablAFx de la gate ≤48, sans les exclure de la comparaison de
  fidélité.
- Aucun de ces articles ne démontre une supériorité sur NAM A2 avec les quatre
  splits de ce dépôt. Ce point est précisément la question confirmatoire.
