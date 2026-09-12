# Fiche technique de la démo

Mesures produit sur le jeu de développement (Fulltone, Big Muff, ToneTwist
CC-BY-NC). Aucune revendication scientifique ; voir `.codex_campaign/` pour
la voie scientifique et ses verdicts.

Repères : sur ce même Big Muff ToneTwist, la littérature rapporte 0,1076
d'ESR pour S4-TFiLM `large` et 0,59 à 0,70 pour des gray-box à une seule
non-linéarité. Le Big Muff est le cas difficile (deux étages de clipping
en cascade) ; 400 époques au lieu de 100 n'y gagnent que 10 % d'ESR, donc
le plateau vient du modèle, pas du budget. L'alignement prédiction/cible
a été vérifié (décalage de pic nul).

Le coût CPU est mesuré avec `-O3` sans `-ffast-math`, afin de préserver la
propagation des NaN et la parité IEEE ; il est donc plus élevé que les
mesures historiques de la voie scientifique compilées en `-Ofast`.

## Fidélité (jeu de test de développement, par appareil)

Ce jeu de test n'est pas scellé : le budget d'époques a été choisi après
l'avoir lu. Sur le Big Muff, l'ESR de validation (0,109) est 1,7 fois
meilleur que celui de test (0,188). Le DI de test est mesurablement plus
exigeant que celui de validation : RMS 0,110 contre 0,0708 (+3,8 dB, donc
un écrêtage plus profond), facteur de crête 5,00 contre 7,78 (jeu plus
soutenu) et flux spectral 43 % plus élevé.

| Appareil | Modèle | ESR | MAE | corrélation | MR-STFT | log-mel |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| fulltone_full_drive_2 | A2 lite | 0.002983 | 0.005349 | 0.9986 | 0.499 | 0.1184 |
| fulltone_full_drive_2 | A2 full | 0.00102 | 0.003257 | 0.9995 | 0.4097 | 0.06382 |
| electro_harmonix_big_muff | A2 lite | 0.2941 | 0.004463 | 0.8514 | 0.7554 | 0.2594 |
| electro_harmonix_big_muff | A2 full | 0.1882 | 0.003421 | 0.9060 | 0.6682 | 0.2196 |

## Coût CPU natif (NeuralAmpModelerCore, Release -O3 sans fast-math)

Le facteur temps réel p95 indique combien de fois plus vite que le temps
réel le modèle calcule (plus grand = mieux) ; la charge CPU est son inverse
sur un cœur. Référence historique en `-Ofast` : 2 862 ns/échantillon pour
A2 Full au bloc 64. Le médian est la mesure fiable ici : le p95 capte
toute autre charge de la machine au moment du banc, et cette machine est
partagée avec des entraînements. À lire avec les mesures sur machine
dédiée quand elles existeront.

| Modèle | Bloc | ns/éch. médian | p95 | x temps réel p95 | charge CPU p95 |
| --- | ---: | ---: | ---: | ---: | ---: |
| fulltone_full_drive_2 A2 lite | 32 | 893.6 | 905.8 | 23.00x | 4.3 % |
| fulltone_full_drive_2 A2 lite | 64 | 836.6 | 947.2 | 22.00x | 4.5 % |
| fulltone_full_drive_2 A2 lite | 128 | 813.7 | 898.3 | 23.19x | 4.3 % |
| fulltone_full_drive_2 A2 lite | 256 | 817.1 | 933.1 | 22.33x | 4.5 % |
| fulltone_full_drive_2 A2 full | 32 | 5288.2 | 5722.3 | 3.64x | 27.5 % |
| fulltone_full_drive_2 A2 full | 64 | 4849.9 | 5313.6 | 3.92x | 25.5 % |
| fulltone_full_drive_2 A2 full | 128 | 4512.1 | 5037.3 | 4.14x | 24.2 % |
| fulltone_full_drive_2 A2 full | 256 | 4256.3 | 4531.3 | 4.60x | 21.8 % |
| electro_harmonix_big_muff A2 lite | 32 | 896.3 | 1080.0 | 19.29x | 5.2 % |
| electro_harmonix_big_muff A2 lite | 64 | 834.7 | 1009.5 | 20.64x | 4.8 % |
| electro_harmonix_big_muff A2 lite | 128 | 815.4 | 979.8 | 21.26x | 4.7 % |
| electro_harmonix_big_muff A2 lite | 256 | 793.2 | 983.9 | 21.17x | 4.7 % |
| electro_harmonix_big_muff A2 full | 32 | 5291.9 | 5995.3 | 3.47x | 28.8 % |
| electro_harmonix_big_muff A2 full | 64 | 4839.1 | 5250.1 | 3.97x | 25.2 % |
| electro_harmonix_big_muff A2 full | 128 | 4424.2 | 4630.2 | 4.50x | 22.2 % |
| electro_harmonix_big_muff A2 full | 256 | 4252.4 | 4841.7 | 4.30x | 23.2 % |

## Parité moteur (Python d'entraînement vs moteur natif C++)

| Modèle | Python vs natif (bloc 64) | blocs réguliers vs irréguliers | reset exact |
| --- | ---: | ---: | --- |
| fulltone_full_drive_2 A2 lite | 1.79e-07 | 0.00e+00 | True |
| fulltone_full_drive_2 A2 full | 2.24e-07 | 0.00e+00 | True |
| electro_harmonix_big_muff A2 lite | 1.68e-06 | 0.00e+00 | True |
| electro_harmonix_big_muff A2 full | 2.33e-06 | 0.00e+00 | True |

Le rendu hors ligne du plugin JUCE (`scripts/product_plugin_parity.py`)
est bit à bit identique au runner natif sur les quatre modèles, malgré
les conversions float/double et le flush des dénormaux.

## Robustesse (moteur natif, blocs irréguliers)

| Modèle | sonde | fini | crête |
| --- | --- | --- | ---: |
| fulltone_full_drive_2 A2 lite | silence | True | 0.0005422 |
| fulltone_full_drive_2 A2 lite | dc | True | 0.3867 |
| fulltone_full_drive_2 A2 lite | hot | True | 2.088 |
| fulltone_full_drive_2 A2 full | silence | True | 0.0001178 |
| fulltone_full_drive_2 A2 full | dc | True | 0.4188 |
| fulltone_full_drive_2 A2 full | hot | True | 2.265 |
| electro_harmonix_big_muff A2 lite | silence | True | 0.0009789 |
| electro_harmonix_big_muff A2 lite | dc | True | 0.2865 |
| electro_harmonix_big_muff A2 lite | hot | True | 0.4343 |
| electro_harmonix_big_muff A2 full | silence | True | 0.0002657 |
| electro_harmonix_big_muff A2 full | dc | True | 0.1374 |
| electro_harmonix_big_muff A2 full | hot | True | 0.1353 |
