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
| fulltone_full_drive_2 A2 lite | 32 | 909.5 | 1071.8 | 19.44x | 5.1 % |
| fulltone_full_drive_2 A2 lite | 64 | 834.2 | 917.0 | 22.72x | 4.4 % |
| fulltone_full_drive_2 A2 lite | 128 | 801.4 | 908.8 | 22.92x | 4.4 % |
| fulltone_full_drive_2 A2 lite | 256 | 796.8 | 967.9 | 21.52x | 4.6 % |
| fulltone_full_drive_2 A2 full | 32 | 5275.0 | 5635.0 | 3.70x | 27.0 % |
| fulltone_full_drive_2 A2 full | 64 | 4847.0 | 5206.3 | 4.00x | 25.0 % |
| fulltone_full_drive_2 A2 full | 128 | 4418.9 | 4646.9 | 4.48x | 22.3 % |
| fulltone_full_drive_2 A2 full | 256 | 4199.8 | 4308.5 | 4.84x | 20.7 % |
| electro_harmonix_big_muff A2 lite | 32 | 891.5 | 970.5 | 21.47x | 4.7 % |
| electro_harmonix_big_muff A2 lite | 64 | 838.4 | 937.2 | 22.23x | 4.5 % |
| electro_harmonix_big_muff A2 lite | 128 | 815.2 | 891.4 | 23.37x | 4.3 % |
| electro_harmonix_big_muff A2 lite | 256 | 804.4 | 901.3 | 23.12x | 4.3 % |
| electro_harmonix_big_muff A2 full | 32 | 5293.4 | 5821.5 | 3.58x | 27.9 % |
| electro_harmonix_big_muff A2 full | 64 | 4851.8 | 5198.7 | 4.01x | 25.0 % |
| electro_harmonix_big_muff A2 full | 128 | 4418.3 | 5514.0 | 3.78x | 26.5 % |
| electro_harmonix_big_muff A2 full | 256 | 4199.6 | 4330.9 | 4.81x | 20.8 % |

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

## Aliasing (sondes sinus cohérentes, grille R2 gelée)

Sondes à bin exact : N = 65 536, 6 trames, la dernière analysée, sans
fenêtre ni bourrage. L'alias est l'énergie hors bins harmoniques sous
Nyquist, rapportée à l'énergie harmonique. Rendu par le moteur natif au
bloc 64 (65 536 / 64 : les bords de bloc tombent sur les trames).

Plancher de mesure (la sonde float32 elle-même) : -154.8 dB.
Au-delà, c'est le modèle, pas l'arithmétique. Contexte d'amplitude : le
DI de test a un RMS de 0,07 à 0,11, donc 0,48 correspond à un jeu fort.
La périodicité inter-trames est exacte sur les 36 sondes (erreur -inf dB) :
le WaveNet est déterministe et son champ réceptif (6 347) tient dans une
trame, donc les trames successives sont bit à bit identiques.

**Ce que le chiffre mesure vraiment.** L'énergie hors bins harmoniques ne
contient pas que du repliement : elle contient toute erreur non harmonique
du modèle. À 9 kHz, seules deux harmoniques tiennent sous Nyquist, donc la
mesure y est surtout un plancher de bruit du modèle rapporté à une énergie
harmonique très réduite. À lire comme une borne supérieure du repliement,
pas comme une mesure isolée.

Les colonnes principales retirent la composante continue. La métrique gelée
compte le bin 0 comme alias, et A2 apprend un décalage continu : sur le
Fulltone il domine tout le reste (jusqu'à 30 dB d'écart, colonne « gelé »).
C'est un constat sur les modèles, pas un défaut de la métrique.

| Modèle | Fréquence | 0,10 | 0,25 | 0,48 | gelé à 0,48 |
| --- | ---: | ---: | ---: | ---: | ---: |
| fulltone_full_drive_2 A2 lite | 1249 Hz | -54.0 dB | -52.2 dB | -49.8 dB | -26.1 dB |
| fulltone_full_drive_2 A2 lite | 5999 Hz | -19.6 dB | -20.3 dB | -23.4 dB | -4.3 dB |
| fulltone_full_drive_2 A2 lite | 8999 Hz | -12.0 dB | -13.8 dB | -16.8 dB | -1.5 dB |
| fulltone_full_drive_2 A2 full | 1249 Hz | -57.5 dB | -53.5 dB | -50.6 dB | -20.0 dB |
| fulltone_full_drive_2 A2 full | 5999 Hz | -21.4 dB | -22.4 dB | -24.0 dB | 5.1 dB |
| fulltone_full_drive_2 A2 full | 8999 Hz | -9.7 dB | -12.0 dB | -13.8 dB | 5.7 dB |
| electro_harmonix_big_muff A2 lite | 1249 Hz | -23.2 dB | -19.7 dB | -18.8 dB | -16.4 dB |
| electro_harmonix_big_muff A2 lite | 5999 Hz | -19.0 dB | -18.2 dB | -22.1 dB | -14.7 dB |
| electro_harmonix_big_muff A2 lite | 8999 Hz | -9.8 dB | -12.8 dB | -12.1 dB | -5.6 dB |
| electro_harmonix_big_muff A2 full | 1249 Hz | -34.5 dB | -31.8 dB | -33.2 dB | -32.6 dB |
| electro_harmonix_big_muff A2 full | 5999 Hz | -23.5 dB | -21.6 dB | -21.4 dB | -21.3 dB |
| electro_harmonix_big_muff A2 full | 8999 Hz | -17.1 dB | -15.6 dB | -14.7 dB | -14.6 dB |
