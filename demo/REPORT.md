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

Le facteur temps réel p95 indique combien de fois plus vite que le temps
réel le modèle calcule (plus grand = mieux) ; la charge CPU est son inverse
sur un cœur. Référence historique en `-Ofast` : 2 862 ns/échantillon pour
A2 Full au bloc 64.

| Modèle | Bloc | ns/éch. médian | p95 | x temps réel p95 | charge CPU p95 |
| --- | ---: | ---: | ---: | ---: | ---: |
| fulltone_full_drive_2 A2 lite | 32 | 896.1 | 1013.8 | 20.55x | 4.9 % |
| fulltone_full_drive_2 A2 lite | 64 | 832.7 | 957.2 | 21.76x | 4.6 % |
| fulltone_full_drive_2 A2 lite | 128 | 819.6 | 960.2 | 21.70x | 4.6 % |
| fulltone_full_drive_2 A2 lite | 256 | 796.8 | 995.8 | 20.92x | 4.8 % |
| fulltone_full_drive_2 A2 full | 32 | 5270.3 | 5599.8 | 3.72x | 26.9 % |
| fulltone_full_drive_2 A2 full | 64 | 4852.2 | 5276.3 | 3.95x | 25.3 % |
| fulltone_full_drive_2 A2 full | 128 | 4443.4 | 4703.8 | 4.43x | 22.6 % |
| fulltone_full_drive_2 A2 full | 256 | 4254.7 | 15919.5 | 1.31x | 76.4 % |
| electro_harmonix_big_muff A2 lite | 32 | 893.2 | 1078.8 | 19.31x | 5.2 % |
| electro_harmonix_big_muff A2 lite | 64 | 860.1 | 1065.5 | 19.55x | 5.1 % |
| electro_harmonix_big_muff A2 lite | 128 | 813.9 | 1037.5 | 20.08x | 5.0 % |
| electro_harmonix_big_muff A2 lite | 256 | 792.9 | 1000.9 | 20.82x | 4.8 % |
| electro_harmonix_big_muff A2 full | 32 | 5277.6 | 5726.4 | 3.64x | 27.5 % |
| electro_harmonix_big_muff A2 full | 64 | 4853.5 | 5244.7 | 3.97x | 25.2 % |
| electro_harmonix_big_muff A2 full | 128 | 4416.0 | 4618.2 | 4.51x | 22.2 % |
| electro_harmonix_big_muff A2 full | 256 | 4249.4 | 4863.4 | 4.28x | 23.3 % |

## Parité moteur (Python d'entraînement vs moteur natif C++)

| Modèle | Python vs natif (bloc 64) | blocs réguliers vs irréguliers | reset exact |
| --- | ---: | ---: | --- |
| fulltone_full_drive_2 A2 lite | 1.79e-07 | 0.00e+00 | True |
| fulltone_full_drive_2 A2 full | 2.24e-07 | 0.00e+00 | True |
| electro_harmonix_big_muff A2 lite | 1.68e-06 | 0.00e+00 | True |
| electro_harmonix_big_muff A2 full | 2.33e-06 | 0.00e+00 | True |

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
