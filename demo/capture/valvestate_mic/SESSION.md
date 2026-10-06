# Séance de capture au micro : Marshall Valvestate VS100R

Voie produit. Généré par `scripts/product_valvestate_mic_session.py` (graine 20261006) ; ne pas éditer à la main, relancer le script.

23 prises de 5.1 min sur 20 réglages, environ 3 h avec les changements de réglage et l'installation.

## Chaîne

- Source : `datasets/raw/internal/valvestate_vs100r_mic/reamp_valvestate_mic_192k.wav`, 308.683 s, 192 kHz, 24 bits, sha256 `8ee5f7e171fab628…`. Sections et positions : `demo/capture/valvestate_mic/layout.json`. Les 212 premières secondes sont la source de septembre, identique ; suivent la polarité, les niveaux et les salves.
- Interface : Antelope Zen Go Synergy Core, **192 kHz**, 24 bits, taille de tampon fixe pour toute la séance. **Tous les effets Synergy Core et le monitoring direct coupés** sur les entrées enregistrées.
- **Entrée 1 = bouclage**, mode ligne, 0 dB, comme en septembre : c'est l'entrée des modèles. Noter d'où il est pris (sortie de la Zen Go avant la boîte de réamplification, ou sortie de la boîte).
- **Entrée 2 = micro**, mode micro (alimentation fantôme si micro statique). C'est la cible.
- Sortie moniteur (jack TRS) → boîte de réamplification → entrée de l'ampli, comme en septembre. Noter le modèle et les réglages de la boîte (non notés en septembre).
- **Reaper** : projet à la fréquence de la source (sinon Reaper la rééchantillonne) ; WAV 24 bits ; piste source à 0 dB, sans effet, envoi au master décoché, sortie matérielle 1 ; deux pistes mono armées sur les entrées 1 et 2, monitoring d'entrée coupé. Lancer lecture et enregistrement ensemble, arrêter après la fin du fichier.

## Ampli, micro et pièce

- Canal **OD2**. Reverb à **0**. Power Dimension **éteint**. Retour de boucle vide.
- **Volume OD2 et master fixes pour toute la séance**, à un niveau de jeu habituel. Le haut-parleur et l'ampli de puissance font partie de la cible : un autre volume serait un autre appareil.
- **Micro** : position fixe pour toute la séance. Mesurer la distance à la grille (cm), la position sur le cône (centre, mi-rayon, bord) et l'angle ; marquer l'emplacement au ruban, photographier. Retard acoustique attendu : distance / 343 m/s (10 cm ≈ 0,29 ms ≈ 56 échantillons à 192 kHz).
- **Pièce** : fermée, mêmes conditions toute la séance ; noter les bruits notables (chauffage, circulation) et à quelle prise. Chaque prise contient déjà 5 s de silence pour le bruit de fond.
- Si le micro ou l'ampli bouge pendant la séance : le noter, remettre en place sur les marques, et refaire une prise centre avant de continuer.

## Test des deux entrées, avant de régler les niveaux (5 min)

Les trois prises A, B et C de `demo/capture/valvestate/INTER_INPUT_TEST.md` (un signal partagé sur les deux entrées en mode ligne), jamais faites depuis septembre. Elles tranchent le décalage d'un échantillon des modèles de septembre. Il change le niveau de sortie : le faire avant le réglage des niveaux ci-dessous, puis remettre l'entrée 2 en mode micro.

## Niveaux, réglés une fois

- **Entrée de l'ampli** : mêmes réglages de sortie et de boîte qu'en septembre si on les retrouve (0 dBFS de la source vers 0 dBu à l'entrée de l'ampli, méthode dans `demo/capture/valvestate/SESSION.md`) ; sinon les régler ainsi et les noter.
- **Gain du micro**, avant la première prise : ampli au réglage probablement le plus fort du plan (gain 10, contour 10, celui de la prise 2), jouer la section « niveau +6 dB » de la source (255 à 265 s) : crêtes de l'entrée 2 vers -10 dBFS, jamais d'écrêtage. **Ne plus toucher à aucun niveau ni volume** ensuite.

Valeurs fixes, à noter une fois :

| | valeur |
|---|---|
| Volume OD2 | |
| Master | |
| Micro : modèle, distance (cm), position sur le cône, angle | |
| Haut-parleur et baffle (d'origine ?) | |
| Pièce | |
| Boîte de réamplification : modèle, réglages | |
| Bouclage : pris avant ou après la boîte | |
| Volume moniteur de la Zen Go (dB) | |
| Gains et modes des deux entrées | |
| Taille de tampon | |

## Prises, dans cet ordre

Boutons de 0 à 10. Revenir au 5 des prises « centre » **depuis un côté différent** à chaque fois. Une photo du panneau par prise. Nommer chaque fichier comme dans la dernière colonne (suffixe de voie : `_loop`, `_mic`). Noter toute prise refaite.

| # | fait | gain | contour | grave | médium | aigu | rôle | en sept. | fichier |
|---|---|---|---|---|---|---|---|---|---|
| 1 | ☐ | 5 | 5 | 5 | 5 | 5 | center (train) | oui | `G050_C050_B050_M050_T050.vs100r_mic_01.wav` |
| 2 | ☐ | 10 | 10 | 5 | 5 | 5 | unseen pair (test) |  | `G100_C100_B050_M050_T050.vs100r_mic_02.wav` |
| 3 | ☐ | 7,5 | 2,5 | 7,5 | 2,5 | 5 | combination (train) | oui | `G075_C025_B075_M025_T050.vs100r_mic_03.wav` |
| 4 | ☐ | 10 | 2,5 | 5 | 5 | 5 | unseen pair (test) |  | `G100_C025_B050_M050_T050.vs100r_mic_04.wav` |
| 5 | ☐ | 5 | 5 | 0 | 5 | 5 | tone (train) | oui | `G050_C050_B000_M050_T050.vs100r_mic_05.wav` |
| 6 | ☐ | 7,5 | 5 | 5 | 5 | 5 | gain (train) | oui | `G075_C050_B050_M050_T050.vs100r_mic_06.wav` |
| 7 | ☐ | 5 | 5 | 10 | 5 | 5 | tone (train) | oui | `G050_C050_B100_M050_T050.vs100r_mic_07.wav` |
| 8 | ☐ | 5 | 5 | 5 | 0 | 5 | tone (train) | oui | `G050_C050_B050_M000_T050.vs100r_mic_08.wav` |
| 9 | ☐ | 2,5 | 5 | 5 | 5 | 5 | gain (train) | oui | `G025_C050_B050_M050_T050.vs100r_mic_09.wav` |
| 10 | ☐ | 2,5 | 7,5 | 2,5 | 7,5 | 7,5 | sept take16 (test) | oui | `G025_C075_B025_M075_T075.vs100r_mic_10.wav` |
| 11 | ☐ | 5 | 5 | 5 | 5 | 10 | tone (train) | oui | `G050_C050_B050_M050_T100.vs100r_mic_11.wav` |
| 12 | ☐ | 5 | 5 | 5 | 5 | 5 | center (train) | oui | `G050_C050_B050_M050_T050.vs100r_mic_12.wav` |
| 13 | ☐ | 2,5 | 2,5 | 5 | 5 | 5 | gain 2.5 (train) |  | `G025_C025_B050_M050_T050.vs100r_mic_13.wav` |
| 14 | ☐ | 2,5 | 5 | 7,5 | 2,5 | 7,5 | gain 2.5 (train) |  | `G025_C050_B075_M025_T075.vs100r_mic_14.wav` |
| 15 | ☐ | 5 | 0 | 5 | 5 | 5 | contour (train) | oui | `G050_C000_B050_M050_T050.vs100r_mic_15.wav` |
| 16 | ☐ | 5 | 5 | 5 | 10 | 5 | tone (train) | oui | `G050_C050_B050_M100_T050.vs100r_mic_16.wav` |
| 17 | ☐ | 5 | 10 | 5 | 5 | 5 | contour (train) | oui | `G050_C100_B050_M050_T050.vs100r_mic_17.wav` |
| 18 | ☐ | 2,5 | 10 | 5 | 5 | 5 | unseen pair (test) |  | `G025_C100_B050_M050_T050.vs100r_mic_18.wav` |
| 19 | ☐ | 5 | 5 | 5 | 5 | 0 | tone (train) | oui | `G050_C050_B050_M050_T000.vs100r_mic_19.wav` |
| 20 | ☐ | 2,5 | 7,5 | 2,5 | 7,5 | 7,5 | sept take16 (test) | oui | `G025_C075_B025_M075_T075.vs100r_mic_20.wav` |
| 21 | ☐ | 10 | 5 | 5 | 5 | 5 | gain (train) | oui | `G100_C050_B050_M050_T050.vs100r_mic_21.wav` |
| 22 | ☐ | 7,5 | 7,5 | 5 | 5 | 5 | unseen pair (test) |  | `G075_C075_B050_M050_T050.vs100r_mic_22.wav` |
| 23 | ☐ | 5 | 5 | 5 | 5 | 5 | center (train) | oui | `G050_C050_B050_M050_T050.vs100r_mic_23.wav` |

Rôles : *train* sert à l'entraînement ; *test* n'y entre jamais. *center* trois fois pour la dérive. *sept take16* : le réglage de la prise 16 de septembre, deux fois, pour savoir si cette prise était représentative. *unseen pair* : couples gain-contour jamais enregistrés. « en sept. » : le même réglage existe dans la capture de septembre (départ de boucle).

## Après la séance

Déposer les fichiers dans `datasets/amp_capture_mic/` (non commité), avec les photos et ce tableau rempli. Les scripts de traitement de septembre lisent l'ancien layout : à rediriger avant de traiter ces prises.
