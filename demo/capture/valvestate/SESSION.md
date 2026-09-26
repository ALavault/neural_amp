# Séance de capture : Marshall Valvestate VS100R

Voie produit. Généré par `scripts/product_valvestate_session.py` (graine 20260926) ; ne pas éditer à la main, relancer le script.

## Avant la première prise

- Source : `datasets/raw/internal/valvestate_vs100r/reamp_valvestate_96k.wav`, 210.508 s, 96 kHz, 24 bits, sha256 `90719ea2bd8dcb39…`. Sections et positions : `layout.json`.
- Enregistrer à **96 kHz**, 24 bits ou float, trois voies : départ de boucle (FX Send), sortie ligne émulée, retour en boucle du signal envoyé.
- Interface → boîte de réamplification → entrée de l'ampli. **Jamais** la sortie haut-parleur vers l'interface.
- Reverb Clean et OD à **0**. Power Dimension **éteint** sauf pour le dernier bloc. Retour de boucle vide.
- **Test de 2 minutes** : note tenue ; bouger le master, puis Power Dimension ; noter laquelle des deux sorties change.
  - FX Send dépend du master : oui / non. De Power Dimension : oui / non.
  - Sortie émulée dépend du master : oui / non. De Power Dimension : oui / non.
- Régler le niveau de réamplification sur la prise 1 (centre) : aucune voie ne doit écrêter, crêtes vers -6 dBFS. **Ne plus y toucher** ensuite.

Valeurs fixes, à noter une fois :

| | valeur |
|---|---|
| Volume OD2 | |
| Volume Clean | |
| Master (s'il existe) | |
| Niveau de départ de boucle (FX level) | |
| Niveau de sortie de l'interface / boîte de réamplification | |
| Gains d'entrée de l'interface (3 voies) | |

## Prises, dans cet ordre

Boutons de 0 à 10. Revenir au 5 des prises « centre » **depuis un côté différent** à chaque fois. Une photo du panneau par prise. Nommer chaque fichier comme dans la dernière colonne (suffixe de voie à ajouter : `_send`, `_line`, `_loop`).

| # | fait | canal | gain | contour | grave | médium | aigu | tone shift | power dim. | rôle | fichier |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | ☐ | OD2 | 5 | 5 | 5 | 5 | 5 |  | off | od2 center (train) | `G050_C050_B050_M050_T050.vs100r_01.wav` |
| 2 | ☐ | OD2 | 7,5 | 2,5 | 7,5 | 2,5 | 5 |  | off | od2 combination test (test) | `G075_C025_B075_M025_T050.vs100r_02.wav` |
| 3 | ☐ | OD2 | 5 | 10 | 0 | 10 | 0 |  | off | od2 combination test (test) | `G050_C100_B000_M100_T000.vs100r_03.wav` |
| 4 | ☐ | OD2 | 5 | 5 | 10 | 5 | 5 |  | off | od2 one knob (train) | `G050_C050_B100_M050_T050.vs100r_04.wav` |
| 5 | ☐ | OD2 | 2,5 | 5 | 5 | 5 | 7,5 |  | off | od2 combination train (train) | `G025_C050_B050_M050_T075.vs100r_05.wav` |
| 6 | ☐ | OD2 | 0 | 2,5 | 5 | 5 | 10 |  | off | od2 combination train (train) | `G000_C025_B050_M050_T100.vs100r_06.wav` |
| 7 | ☐ | OD2 | 0 | 5 | 5 | 5 | 5 |  | off | od2 one knob (train) | `G000_C050_B050_M050_T050.vs100r_07.wav` |
| 8 | ☐ | OD2 | 10 | 0 | 5 | 7,5 | 7,5 |  | off | od2 combination train (train) | `G100_C000_B050_M075_T075.vs100r_08.wav` |
| 9 | ☐ | OD2 | 10 | 5 | 5 | 5 | 5 |  | off | od2 one knob (train) | `G100_C050_B050_M050_T050.vs100r_09.wav` |
| 10 | ☐ | OD2 | 0 | 5 | 2,5 | 10 | 7,5 |  | off | od2 combination train (train) | `G000_C050_B025_M100_T075.vs100r_10.wav` |
| 11 | ☐ | OD2 | 5 | 5 | 7,5 | 5 | 5 |  | off | od2 one knob (train) | `G050_C050_B075_M050_T050.vs100r_11.wav` |
| 12 | ☐ | OD2 | 5 | 5 | 5 | 5 | 0 |  | off | od2 one knob (train) | `G050_C050_B050_M050_T000.vs100r_12.wav` |
| 13 | ☐ | OD2 | 5 | 5 | 0 | 5 | 5 |  | off | od2 one knob (train) | `G050_C050_B000_M050_T050.vs100r_13.wav` |
| 14 | ☐ | OD2 | 5 | 7,5 | 7,5 | 2,5 | 2,5 |  | off | od2 combination test (test) | `G050_C075_B075_M025_T025.vs100r_14.wav` |
| 15 | ☐ | OD2 | 5 | 5 | 5 | 0 | 5 |  | off | od2 one knob (train) | `G050_C050_B050_M000_T050.vs100r_15.wav` |
| 16 | ☐ | OD2 | 2,5 | 7,5 | 2,5 | 7,5 | 7,5 |  | off | od2 combination test (test) | `G025_C075_B025_M075_T075.vs100r_16.wav` |
| 17 | ☐ | OD2 | 5 | 0 | 10 | 0 | 10 |  | off | od2 combination test (test) | `G050_C000_B100_M000_T100.vs100r_17.wav` |
| 18 | ☐ | OD2 | 5 | 2,5 | 5 | 5 | 5 |  | off | od2 one knob (train) | `G050_C025_B050_M050_T050.vs100r_18.wav` |
| 19 | ☐ | OD2 | 5 | 5 | 5 | 7,5 | 5 |  | off | od2 one knob (train) | `G050_C050_B050_M075_T050.vs100r_19.wav` |
| 20 | ☐ | OD2 | 5 | 5 | 5 | 5 | 5 |  | off | od2 center (train) | `G050_C050_B050_M050_T050.vs100r_20.wav` |
| 21 | ☐ | OD2 | 5 | 5 | 5 | 5 | 7,5 |  | off | od2 one knob (train) | `G050_C050_B050_M050_T075.vs100r_21.wav` |
| 22 | ☐ | OD2 | 5 | 7,5 | 5 | 5 | 5 |  | off | od2 one knob (train) | `G050_C075_B050_M050_T050.vs100r_22.wav` |
| 23 | ☐ | OD2 | 5 | 5 | 5 | 5 | 10 |  | off | od2 one knob (train) | `G050_C050_B050_M050_T100.vs100r_23.wav` |
| 24 | ☐ | OD2 | 5 | 5 | 5 | 5 | 2,5 |  | off | od2 one knob (train) | `G050_C050_B050_M050_T025.vs100r_24.wav` |
| 25 | ☐ | OD2 | 5 | 5 | 2,5 | 5 | 5 |  | off | od2 one knob (train) | `G050_C050_B025_M050_T050.vs100r_25.wav` |
| 26 | ☐ | OD2 | 2,5 | 5 | 5 | 5 | 5 |  | off | od2 one knob (train) | `G025_C050_B050_M050_T050.vs100r_26.wav` |
| 27 | ☐ | OD2 | 0 | 0 | 5 | 5 | 10 |  | off | od2 combination train (train) | `G000_C000_B050_M050_T100.vs100r_27.wav` |
| 28 | ☐ | OD2 | 5 | 5 | 5 | 2,5 | 5 |  | off | od2 one knob (train) | `G050_C050_B050_M025_T050.vs100r_28.wav` |
| 29 | ☐ | OD2 | 7,5 | 2,5 | 5 | 10 | 0 |  | off | od2 combination train (train) | `G075_C025_B050_M100_T000.vs100r_29.wav` |
| 30 | ☐ | OD2 | 10 | 5 | 10 | 10 | 0 |  | off | od2 combination test (test) | `G100_C050_B100_M100_T000.vs100r_30.wav` |
| 31 | ☐ | OD2 | 5 | 0 | 5 | 5 | 5 |  | off | od2 one knob (train) | `G050_C000_B050_M050_T050.vs100r_31.wav` |
| 32 | ☐ | OD2 | 2,5 | 2,5 | 0 | 2,5 | 5 |  | off | od2 combination train (train) | `G025_C025_B000_M025_T050.vs100r_32.wav` |
| 33 | ☐ | OD2 | 7,5 | 2,5 | 2,5 | 7,5 | 2,5 |  | off | od2 combination test (test) | `G075_C025_B025_M075_T025.vs100r_33.wav` |
| 34 | ☐ | OD2 | 5 | 5 | 5 | 10 | 5 |  | off | od2 one knob (train) | `G050_C050_B050_M100_T050.vs100r_34.wav` |
| 35 | ☐ | OD2 | 7,5 | 10 | 2,5 | 0 | 2,5 |  | off | od2 combination train (train) | `G075_C100_B025_M000_T025.vs100r_35.wav` |
| 36 | ☐ | OD2 | 2,5 | 5 | 0 | 0 | 10 |  | off | od2 combination test (test) | `G025_C050_B000_M000_T100.vs100r_36.wav` |
| 37 | ☐ | OD2 | 5 | 10 | 5 | 5 | 5 |  | off | od2 one knob (train) | `G050_C100_B050_M050_T050.vs100r_37.wav` |
| 38 | ☐ | OD2 | 7,5 | 5 | 5 | 5 | 5 |  | off | od2 one knob (train) | `G075_C050_B050_M050_T050.vs100r_38.wav` |
| 39 | ☐ | OD2 | 5 | 5 | 5 | 5 | 5 |  | off | od2 center (train) | `G050_C050_B050_M050_T050.vs100r_39.wav` |
| 40 | ☐ | Clean |  |  | 10 | 5 | 0 | off | off | clean (control) | `B100_M050_T000_S000.vs100r_40.wav` |
| 41 | ☐ | Clean |  |  | 5 | 5 | 5 | on | off | clean (control) | `B050_M050_T050_S100.vs100r_41.wav` |
| 42 | ☐ | Clean |  |  | 0 | 5 | 10 | off | off | clean (control) | `B000_M050_T100_S000.vs100r_42.wav` |
| 43 | ☐ | Clean |  |  | 7,5 | 2,5 | 7,5 | off | off | clean (control) | `B075_M025_T075_S000.vs100r_43.wav` |
| 44 | ☐ | Clean |  |  | 5 | 5 | 5 | off | off | clean (control) | `B050_M050_T050_S000.vs100r_44.wav` |
| 45 | ☐ | OD2 | 5 | 5 | 5 | 5 | 5 |  | on | power dimension (control) | `G050_C050_B050_M050_T050_P100.vs100r_45.wav` |
| 46 | ☐ | OD2 | 5 | 0 | 10 | 0 | 10 |  | on | power dimension (control) | `G050_C000_B100_M000_T100_P100.vs100r_46.wav` |
| 47 | ☐ | OD2 | 7,5 | 2,5 | 7,5 | 2,5 | 5 |  | on | power dimension (control) | `G075_C025_B075_M025_T050_P100.vs100r_47.wav` |

Rôles : *one knob* et *combination train* servent à l'entraînement ; *combination test* n'y entre jamais ; *center* trois fois pour la dérive ; *clean* et *power dimension* sont des contrôles.
