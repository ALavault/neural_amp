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

## Fidélité (jeu de test scellé par appareil)

| Appareil | Modèle | ESR | MAE | corrélation | MR-STFT | log-mel |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| fulltone_full_drive_2 | A2 lite | 0.002983 | 0.005349 | 0.9986 | 0.499 | 0.1184 |
| fulltone_full_drive_2 | A2 full | 0.00102 | 0.003257 | 0.9995 | 0.4097 | 0.06382 |
| electro_harmonix_big_muff | A2 lite | 0.2941 | 0.004463 | 0.8514 | 0.7554 | 0.2594 |
| electro_harmonix_big_muff | A2 full | 0.1882 | 0.003421 | 0.9060 | 0.6682 | 0.2196 |

## Coût CPU natif (NeuralAmpModelerCore, Release -O3 sans fast-math)

| Modèle | Bloc | ns/échantillon médian | p95 | facteur temps réel p95 |
| --- | ---: | ---: | ---: | ---: |
| fulltone_full_drive_2 A2 lite | 32 | 902.2 | 974.1 | 21.39x |
| fulltone_full_drive_2 A2 lite | 64 | 829.7 | 942.3 | 22.11x |
| fulltone_full_drive_2 A2 lite | 128 | 817.0 | 886.9 | 23.49x |
| fulltone_full_drive_2 A2 lite | 256 | 812.1 | 897.6 | 23.21x |
| fulltone_full_drive_2 A2 full | 32 | 5267.1 | 5611.4 | 3.71x |
| fulltone_full_drive_2 A2 full | 64 | 4828.5 | 5185.1 | 4.02x |
| fulltone_full_drive_2 A2 full | 128 | 4399.1 | 4690.4 | 4.44x |
| fulltone_full_drive_2 A2 full | 256 | 4251.1 | 4405.3 | 4.73x |
| electro_harmonix_big_muff A2 lite | 32 | 906.7 | 971.6 | 21.44x |
| electro_harmonix_big_muff A2 lite | 64 | 835.7 | 957.3 | 21.76x |
| electro_harmonix_big_muff A2 lite | 128 | 815.1 | 980.0 | 21.26x |
| electro_harmonix_big_muff A2 lite | 256 | 802.3 | 888.9 | 23.44x |
| electro_harmonix_big_muff A2 full | 32 | 5293.7 | 5770.3 | 3.61x |
| electro_harmonix_big_muff A2 full | 64 | 4824.3 | 5132.3 | 4.06x |
| electro_harmonix_big_muff A2 full | 128 | 4422.3 | 4610.0 | 4.52x |
| electro_harmonix_big_muff A2 full | 256 | 4242.4 | 4820.2 | 4.32x |

## Robustesse

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
