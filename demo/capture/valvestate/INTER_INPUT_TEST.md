# Test du décalage entre les deux entrées de la Zen Go

Voie produit. **Écrit le 2026-09-30, avant les prises.** Il tranche l'hypothèse de
`predictions_model.md` (section du retard de groupe). Le modèle `vs100r_gc_film_joint_s925_r2`
est en avance d'environ 1 échantillon à 48 kHz (21 µs) sur la piste du départ de boucle, à
toutes les fréquences de 500 Hz à 8 kHz. Pendant la séance, le bouclage (l'entrée du modèle)
était sur l'**entrée 1** à 0 dB, et le départ de boucle (la cible) sur l'**entrée 2** à +20 dB.
Si l'entrée 2 enregistre en retard d'environ 21 µs sur l'entrée 1, l'interface explique tout le
décalage, et le modèle n'a rien à apprendre de plus.

## Prises (environ 5 minutes en tout)

Un seul signal, partagé passivement (câble en Y, ou sortie de la boîte de réamplification
dédoublée), sur les deux entrées en mode **ligne**. Mêmes réglages de session Reaper que la
séance : 192 kHz, 24 bits, même taille de tampon. Signal large bande, 30 s au moins : le
fichier source de la séance convient, ou un bruit rose. Régler le niveau de sortie pour que
l'entrée 2, à +20 dB, crête sous −6 dBFS (le script affiche les crêtes).

| prise | entrée 1 | entrée 2 | câbles |
|---|---|---|---|
| A | 0 dB | +20 dB (gains de la séance) | comme à la séance |
| B | 0 dB | 0 dB | comme à la séance |
| C | 0 dB | 0 dB | câbles **inversés** entre les deux entrées |

Exporter chaque prise en WAV stéréo (canal gauche = entrée 1), ou en deux mono, dans
`datasets/amp_capture/inter_input/` (non commité), nommés `interin_A`, `interin_B` et
`interin_C`.

## Mesure

```
.venv/bin/python3 demo/capture/valvestate/inter_input_delay.py datasets/amp_capture/inter_input/interin_{A,B,C}.wav
```

Sur chaque prise, le script mesure le retard de l'entrée 2 sur l'entrée 1 (positif : l'entrée
2 est en retard). Il donne la médiane des bandes d'octave de 500 Hz à 8 kHz, en échantillons à
48 kHz et en µs, avec la méthode de `group_delay.py`. Il donne aussi le nombre de bandes
valides (cohérence ≥ 0,5) et leur écart-type. Vérification ponctuelle, faite une fois le
2026-09-30 sur un signal synthétique à 192 kHz, et que le script ne rejoue pas : un retard de
4,0 échantillons donne +1,00 à 48 kHz, un retard nul donne 0,00, et 2,6 donne +0,65. Avec
`--pair`, les deux fichiers doivent venir de la même prise.

## Lecture, fixée avant les prises

Les retards d sont en échantillons à 48 kHz. **Une prise ne conclut que si les cinq bandes
sont valides et que leur écart-type vaut au plus 0,3.** Sinon, elle est « non concluante » :
le retard n'est pas pur, ou le signal est insuffisant.

- **Prise A, gains de la séance** :
  - d dans [+0,7 ; +1,3] : **l'interface explique le décalage du modèle** ;
  - |d| ≤ 0,3 : **elle ne l'explique pas**, et le retard pur manquant est dans le modèle ou
    l'ampli ;
  - entre les deux : explication partielle, rapportée telle quelle.
- **Prise B, gains égaux** :
  - si A explique le décalage et que |d_B| ≤ 0,3, le décalage dépend du gain de l'entrée ;
  - si d_B est dans ±0,3 de d_A, c'est un décalage fixe entre les deux voies.
- **Prise C, câbles inversés** :
  - |d_C − d_B| ≤ 0,3 : le décalage tient aux voies d'entrée de l'interface ;
  - d_C ≈ −d_B : il tient à la branche externe entière (bras du Y ou du dédoublement,
    adaptateurs, câble), sans pouvoir séparer ces éléments.

**Conséquence, si A explique le décalage** : aligner la cible dans les données, comme
`--target-advance` pour F-Fuzz, en avançant le départ de boucle de d. Un nouveau run ne part
qu'après une décision de l'utilisateur. Rien n'est réentraîné par ce test.

## Ce que le test ne dira pas

Il mesure l'interface avec ses réglages. Il ne dit rien de l'ampli lui-même. Un décalage de
l'interface qui changerait d'une séance à l'autre, par exemple selon le tampon, ne serait pas vu
par ces trois prises.
