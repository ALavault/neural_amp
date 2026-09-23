# Test D : résultats

Voie produit, systèmes synthétiques M1, aucune revendication sur un appareil réel.
Prédictions : `predictions.md` (commit f6dd3ce, avant exécution). Chiffres :
`results.json`, script `scripts/product_chain_ultrasonic.py`. Toutes les paires et
excitations sont rapportées.

## Synthèse (différence dans la bande 0–21,6 kHz, dB relatifs à la sortie de référence)

| excitation | plancher (témoin identité, max) | médiane 16 paires | pire paire | médiane après baffle 5 kHz | ultrasons de l'étage 1 |
|---|---|---|---|---|---|
| procedural_plucks | −74,7 | −73,2 | −68,3 | −79,6 | −66 à −59 |
| multisine | −55,8 | −50,4 | −42,1 | −62,5 | −45 à −31 |
| two_tone | −62,5 | −66,9 | −61,6 | −67,5 | −62 à −54 |
| colored_noise | −56,2 | −57,1 | −49,4 | −65,4 | −52 à −38 |

Pire paire partout : étage 1 dur (asymmetric_waveshaper ou memory_clipper) suivi d'un
étage 2 sans mémoire (asymmetric_waveshaper ou tanh).

## Verdicts

- **D0 (plancher < −60 dB)** : tenue pour plucks (−74,7) et two_tone (−62,5) ; **réfutée**
  pour multisine (−55,8) et colored_noise (−56,2). Le filtrage seul y laisse un écart de
  cet ordre ; les effets mesurés sur ces excitations ne le dépassent que de 6 à 14 dB. Sur
  two_tone, les paires non linéaires sont sous le plancher : cette excitation ne mesure
  rien.
- **D1 (plucks : médiane entre −60 et −30 dB, une paire au-dessus de −40)** : **réfutée
  vers le bas**. Toutes les paires sont sous −60 dB (−68,3 à −78,8).
- **D2 (baffle : réduction médiane < 10 dB)** : tenue pour plucks (6,4 dB), two_tone
  (0,6) et colored_noise (8,3) ; réfutée pour multisine (12,1).
- **D3 (two_tone et multisine > plucks)** : tenue en valeur (−50,4 et −66,9 contre −73,2),
  mais two_tone est sous son plancher ; seule la comparaison multisine est informative.

## Lecture pour le produit

Le seuil de pertinence fixé d'avance (−30 dB, soit 16 dB sous l'erreur actuelle des
modèles Big Muff à −14 dB) n'est atteint dans aucun cas. Le pire cas, −42 dB sur un
multisine à travers deux écrêteurs durs, reste 28 dB sous l'erreur des modèles. Dans ce
cadre, **transmettre l'ultrasonore d'un étage à l'étage suivant ne change rien d'audible
en pratique**, et la piste C (prolonger les harmoniques au-delà de Nyquist pour les
chaînes) perd sa justification.

## Limites

- Systèmes synthétiques simples ; un fuzz réel à fort gain produit plus d'ultrasons.
  L'effet croît avec l'ultrason de l'étage 1 (multisine, −31 dB au-dessus de 24 kHz,
  donne les plus grands écarts). Un appareil à −20 dB d'ultrasons n'est pas couvert.
- Les « plucks » n'ont que 8 harmoniques (fondamentales 82 à 330 Hz) : un DI de guitare
  réel est plus riche en aigus.
- Filtres des systèmes définis en échantillons, évalués à 192 kHz.
- Plancher de l'instrument limité par la chaîne de décimation gelée (−56 dB sur signaux
  à large bande).
