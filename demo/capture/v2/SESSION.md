# Séance de capture v2 : ampli logiciel

Voie produit. Généré par `scripts/product_capture_v2.py` (graine 20261010) ; ne pas éditer à la main, relancer le script. Appareil **logiciel** seulement (research-memory D-0013) : Overloud TH-U 2 ou Neural DSP Archetype: Nolly. Un dossier par logiciel (`thu2/`, `nolly/`).

## Avant la première prise

- **Source** : `datasets/raw/internal/capture_v2/reamp_v2_192k.wav`, 451.631 s, 192 kHz, 24 bits, sha256 `9ebc26a3875ab1fc…`. Sections : `layout.json`.
- **Reaper** : projet à **192 kHz** (sinon Reaper rééchantillonne la source et l'entrée n'est plus identique au bit, D-0012). Item source : volume d'item 0 dB, sans fondu, sans changement de vitesse ni de hauteur. Piste source : volume 0 dB, pan centré, aucun effet hors le plugin ; master à 0 dB, sans effet. Auto-bypass de Reaper sur silence **désactivé** (pour le plugin et pendant les rendus) : il couperait le silence et les queues.
- **Rendu** : hors ligne pleine vitesse, WAV **32 bits flottants** (un ampli logiciel peut dépasser 0 dBFS : 24 bits écrêterait), **mono**, 192 kHz, ni dither ni normalisation, bornes = tout le projet (le fichier finit par 6 s de silence pour la queue). Méthode simple : une piste par prise, chacune avec son instance du plugin, puis « Render stems (selected tracks) » en une passe.
- **Place** : ~347 Mo par prise, ~14,6 Go pour les 42 prises.

### Contrôle 0 : bouclage logiciel (condition d'arrêt)

Rendre la piste source **sans le plugin** (`loopback_software.wav`), avec exactement les réglages de rendu ci-dessus. Test du nul : sur un nouveau projet, importer le WAV source d'origine (nouvel item, réglages par défaut) et ce rendu, polarité inversée sur l'un des deux : le master doit rester à -inf. **Sinon, ne rien rendre d'autre** (rééchantillonnage, gain d'item, loi de panoramique, dither, normalisation ou fader à corriger). M'envoyer aussi ce rendu : je le compare au fichier d'origine échantillon par échantillon.

### État fixe du plugin, toute la séance

- **Coupés** : noise gate (il couperait les paliers faibles, le silence et le bruit rose), toutes les pédales, EQ graphique, doubler, delay, reverb, transposition, accordeur.
- **Baffle (cab)** : coupé, sauf le bloc de contrôle *cab on* en fin de séance (même baffle, mêmes micros, même placement pour ses trois prises).
- **Fixes et notés** : gain d'entrée, gain de sortie, volume ou master de l'ampli, mode de qualité ou de suréchantillonnage.
- **Modèle d'ampli** : un seul pour toute la séance ; pas une émulation Blackstar ni 1176 (volet scientifique scellé).
- **Boutons** : valeur exacte, vérifiée sur l'affichage numérique du plugin ; la méthode dépend du plugin (chez Nolly X, le double clic rétablit la valeur par défaut : réglage fin par glissement avec la touche indiquée dans le manuel). Si l'affichage est en %, 10 = 100 %. Pour chaque prise, enregistrer le preset sous le nom du fichier et une capture d'écran.

### Pré-contrôles sur l'ampli choisi (centre, tous boutons à 5)

- Le plugin se charge et rend à **192 kHz**. S'il refuse : s'arrêter et demander une variante (48 kHz par décimation de la source).
- **Déterminisme** : rendre le centre deux fois, dans deux instances séparées ; test du nul entre les deux : -inf attendu. Sinon, vérifier les réglages, puis s'arrêter et m'envoyer les deux rendus : un résidu peut venir d'un réglage, d'un état initial ou d'un décalage, pas seulement du plugin.
- **Stéréo** : rendre le centre une fois en stéréo (`center_stereo.wav`) ; les deux canaux doivent être identiques, sinon un étage stéréo est resté actif. L'envoyer avec `loopback_software.wav` : je vérifie les deux.
- **Latence** : noter la PDC affichée par Reaper pour le plugin.

Valeurs fixes, à noter une fois par logiciel :

| | valeur |
|---|---|
| Logiciel, version, format (VST3/AU) | |
| Système, version de Reaper | |
| Modèle d'ampli, canal | |
| Commande de l'ampli pour « gain » (G) | |
| Commande de l'ampli pour « grave » (B) | |
| Commande de l'ampli pour « médium » (M) | |
| Commande de l'ampli pour « aigu » (T) | |
| Commande de l'ampli pour « présence » (P) | |
| Autres commandes de l'ampli (valeur fixe) | |
| Gain d'entrée, gain de sortie | |
| Mode de qualité ou de suréchantillonnage | |
| PDC affichée (échantillons) | |
| Déterminisme : résidu du nul (dBFS) | |
| Bloc cab on : baffle, micros, placement | |

## Prises, dans cet ordre

Boutons de 0 à 10. Nommer chaque rendu comme dans la dernière colonne, dans le dossier du logiciel.

| # | fait | gain | grave | médium | aigu | présence | baffle | rôle | fichier |
|---|---|---|---|---|---|---|---|---|---|
| 1 | ☐ | 5 | 5 | 5 | 5 | 5 | off | center (train) | `G050_B050_M050_T050_P050.capv2_01.wav` |
| 2 | ☐ | 5 | 10 | 5 | 5 | 5 | off | one knob (train) | `G050_B100_M050_T050_P050.capv2_02.wav` |
| 3 | ☐ | 5 | 5 | 0 | 5 | 5 | off | one knob (train) | `G050_B050_M000_T050_P050.capv2_03.wav` |
| 4 | ☐ | 5 | 5 | 5 | 10 | 5 | off | one knob (train) | `G050_B050_M050_T100_P050.capv2_04.wav` |
| 5 | ☐ | 5 | 5 | 5 | 5 | 2,5 | off | one knob (train) | `G050_B050_M050_T050_P025.capv2_05.wav` |
| 6 | ☐ | 7,5 | 5 | 5 | 5 | 5 | off | one knob (train) | `G075_B050_M050_T050_P050.capv2_06.wav` |
| 7 | ☐ | 10 | 5 | 5 | 5 | 5 | off | one knob (train) | `G100_B050_M050_T050_P050.capv2_07.wav` |
| 8 | ☐ | 5 | 7,5 | 7,5 | 2,5 | 2,5 | off | combination test (test) | `G050_B075_M075_T025_P025.capv2_08.wav` |
| 9 | ☐ | 5 | 7,5 | 0 | 7,5 | 10 | off | combination train (train) | `G050_B075_M000_T075_P100.capv2_09.wav` |
| 10 | ☐ | 5 | 0 | 5 | 5 | 5 | off | one knob (train) | `G050_B000_M050_T050_P050.capv2_10.wav` |
| 11 | ☐ | 5 | 5 | 5 | 2,5 | 5 | off | one knob (train) | `G050_B050_M050_T025_P050.capv2_11.wav` |
| 12 | ☐ | 7,5 | 2,5 | 2,5 | 7,5 | 2,5 | off | combination test (test) | `G075_B025_M025_T075_P025.capv2_12.wav` |
| 13 | ☐ | 5 | 0 | 7,5 | 10 | 10 | off | combination train (train) | `G050_B000_M075_T100_P100.capv2_13.wav` |
| 14 | ☐ | 5 | 10 | 0 | 10 | 0 | off | combination test (test) | `G050_B100_M000_T100_P000.capv2_14.wav` |
| 15 | ☐ | 10 | 2,5 | 7,5 | 5 | 2,5 | off | combination train (train) | `G100_B025_M075_T050_P025.capv2_15.wav` |
| 16 | ☐ | 2,5 | 5 | 5 | 5 | 5 | off | one knob (train) | `G025_B050_M050_T050_P050.capv2_16.wav` |
| 17 | ☐ | 2,5 | 5 | 7,5 | 2,5 | 0 | off | combination train (train) | `G025_B050_M075_T025_P000.capv2_17.wav` |
| 18 | ☐ | 2,5 | 5 | 0 | 0 | 10 | off | combination test (test) | `G025_B050_M000_T000_P100.capv2_18.wav` |
| 19 | ☐ | 10 | 10 | 0 | 2,5 | 2,5 | off | combination train (train) | `G100_B100_M000_T025_P025.capv2_19.wav` |
| 20 | ☐ | 5 | 5 | 5 | 5 | 5 | off | center (train) | `G050_B050_M050_T050_P050.capv2_20.wav` |
| 21 | ☐ | 5 | 7,5 | 5 | 5 | 5 | off | one knob (train) | `G050_B075_M050_T050_P050.capv2_21.wav` |
| 22 | ☐ | 5 | 2,5 | 5 | 5 | 5 | off | one knob (train) | `G050_B025_M050_T050_P050.capv2_22.wav` |
| 23 | ☐ | 5 | 5 | 5 | 5 | 10 | off | one knob (train) | `G050_B050_M050_T050_P100.capv2_23.wav` |
| 24 | ☐ | 5 | 2,5 | 5 | 2,5 | 10 | off | combination train (train) | `G050_B025_M050_T025_P100.capv2_24.wav` |
| 25 | ☐ | 0 | 2,5 | 2,5 | 10 | 5 | off | combination train (train) | `G000_B025_M025_T100_P050.capv2_25.wav` |
| 26 | ☐ | 5 | 5 | 2,5 | 5 | 5 | off | one knob (train) | `G050_B050_M025_T050_P050.capv2_26.wav` |
| 27 | ☐ | 2,5 | 7,5 | 5 | 0 | 7,5 | off | combination train (train) | `G025_B075_M050_T000_P075.capv2_27.wav` |
| 28 | ☐ | 5 | 5 | 5 | 5 | 7,5 | off | one knob (train) | `G050_B050_M050_T050_P075.capv2_28.wav` |
| 29 | ☐ | 2,5 | 7,5 | 2,5 | 7,5 | 7,5 | off | combination test (test) | `G025_B075_M025_T075_P075.capv2_29.wav` |
| 30 | ☐ | 0 | 5 | 5 | 5 | 5 | off | one knob (train) | `G000_B050_M050_T050_P050.capv2_30.wav` |
| 31 | ☐ | 5 | 5 | 5 | 7,5 | 5 | off | one knob (train) | `G050_B050_M050_T075_P050.capv2_31.wav` |
| 32 | ☐ | 7,5 | 2,5 | 7,5 | 2,5 | 5 | off | combination test (test) | `G075_B025_M075_T025_P050.capv2_32.wav` |
| 33 | ☐ | 5 | 5 | 5 | 0 | 5 | off | one knob (train) | `G050_B050_M050_T000_P050.capv2_33.wav` |
| 34 | ☐ | 5 | 5 | 7,5 | 5 | 5 | off | one knob (train) | `G050_B050_M075_T050_P050.capv2_34.wav` |
| 35 | ☐ | 5 | 5 | 10 | 5 | 5 | off | one knob (train) | `G050_B050_M100_T050_P050.capv2_35.wav` |
| 36 | ☐ | 5 | 5 | 5 | 5 | 0 | off | one knob (train) | `G050_B050_M050_T050_P000.capv2_36.wav` |
| 37 | ☐ | 10 | 5 | 10 | 10 | 0 | off | combination test (test) | `G100_B050_M100_T100_P000.capv2_37.wav` |
| 38 | ☐ | 5 | 0 | 10 | 0 | 10 | off | combination test (test) | `G050_B000_M100_T000_P100.capv2_38.wav` |
| 39 | ☐ | 5 | 5 | 5 | 5 | 5 | off | center (train) | `G050_B050_M050_T050_P050.capv2_39.wav` |
| 40 | ☐ | 7,5 | 2,5 | 7,5 | 2,5 | 5 | on | cab on (control) | `G075_B025_M075_T025_P050_X100.capv2_40.wav` |
| 41 | ☐ | 5 | 0 | 10 | 0 | 10 | on | cab on (control) | `G050_B000_M100_T000_P100_X100.capv2_41.wav` |
| 42 | ☐ | 5 | 5 | 5 | 5 | 5 | on | cab on (control) | `G050_B050_M050_T050_P050_X100.capv2_42.wav` |

Rôles : *one knob* et *combination train* servent à l'entraînement ; *combination test* n'y entre jamais ; *center* trois fois (déterminisme) ; *cab on* (suffixe `X100`) est un contrôle : la même chaîne avec le baffle, en principe un filtre linéaire de plus (les sondes le vérifieront).
