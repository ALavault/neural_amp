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
meilleur que celui de test (0,188) ; le fichier de test est un autre jeu
de guitare, plus dense, donc plus difficile.

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
A2 Full au bloc 64.

| Modèle | Bloc | ns/éch. médian | p95 | x temps réel p95 | charge CPU p95 |
| --- | ---: | ---: | ---: | ---: | ---: |
| fulltone_full_drive_2 A2 lite | 32 | 893.8 | 981.8 | 21.22x | 4.7 % |
| fulltone_full_drive_2 A2 lite | 64 | 831.8 | 938.2 | 22.21x | 4.5 % |
| fulltone_full_drive_2 A2 lite | 128 | 802.4 | 898.7 | 23.18x | 4.3 % |
| fulltone_full_drive_2 A2 lite | 256 | 823.4 | 944.9 | 22.05x | 4.5 % |
| fulltone_full_drive_2 A2 full | 32 | 5292.0 | 5976.5 | 3.49x | 28.7 % |
| fulltone_full_drive_2 A2 full | 64 | 4957.0 | 6992.7 | 2.98x | 33.6 % |
| fulltone_full_drive_2 A2 full | 128 | 4490.2 | 12237.4 | 1.70x | 58.7 % |
| fulltone_full_drive_2 A2 full | 256 | 4981.8 | 27653.9 | 0.75x | 132.7 % |
| electro_harmonix_big_muff A2 lite | 32 | 1228.6 | 1349.8 | 15.43x | 6.5 % |
| electro_harmonix_big_muff A2 lite | 64 | 1052.5 | 1166.4 | 17.86x | 5.6 % |
| electro_harmonix_big_muff A2 lite | 128 | 912.1 | 1148.2 | 18.14x | 5.5 % |
| electro_harmonix_big_muff A2 lite | 256 | 861.9 | 1365.8 | 15.25x | 6.6 % |
| electro_harmonix_big_muff A2 full | 32 | 7233.3 | 7825.4 | 2.66x | 37.6 % |
| electro_harmonix_big_muff A2 full | 64 | 4903.4 | 6822.4 | 3.05x | 32.7 % |
| electro_harmonix_big_muff A2 full | 128 | 4424.3 | 4654.4 | 4.48x | 22.3 % |
| electro_harmonix_big_muff A2 full | 256 | 4200.9 | 4311.1 | 4.83x | 20.7 % |

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
