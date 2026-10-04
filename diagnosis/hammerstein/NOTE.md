# Analyse de Hammerstein de la VS100R et de ses modèles (exploration)

Voie produit, régime exploration, 2026-10-04. Chaque affirmation est marquée **établie** (mesure
citée) ou **supposée**. Instrument : `src/fssr_nam/product/hammerstein.py` ; analyse :
`vs100r_hammerstein.py` → `vs100r_hammerstein.json`.

## Instrument

- Identification d'un modèle de Hammerstein parallèle par sinus glissant exponentiel. La matrice
  qui relie les réponses harmoniques D_k aux filtres G_n est **mesurée** en passant les
  puissances pures de la même entrée dans la même chaîne, au lieu d'être dérivée par
  trigonométrie : phases du sinus glissant non synchronisé, fondus et décimation se compensent.
- **Établi** : sur des systèmes synthétiques aux filtres connus, l'erreur relative médiane est
  d'environ 10⁻³, à 48 kHz comme après décimation depuis 192 kHz (`tests/unit/test_hammerstein.py`).
- Données : les trois sinus glissants de chaque prise VS100R (20 Hz–20 kHz, 5 s, −20, −10 et
  −3 dBFS, à 192 kHz). Référence : l'entrée bouclée de la prise. Analyse à 48 kHz après la
  décimation des données d'entraînement. Modèles : 3 graines à 48 kHz (A48), 3 à 192 kHz
  (B192), alimentés par la même entrée bouclée. 8 prises de test.

## Résultats

**Un modèle de Hammerstein d'ordre 5 ne décrit pas l'OD2 (établi).**
- Les filtres G_n changent fortement avec le niveau : G₁ baisse d'environ 21 dB entre −20 et
  −3 dBFS (prise 17).
- Prédire une autre partie du signal avec les filtres identifiés échoue. Sur le bruit rose
  à −30 dBFS efficace, l'ESR médiane vaut 2,8 (modèle identifié à −20 dBFS), 1,5 (−10) et 1,05
  (−3). Sur la guitare de test, le modèle à −20 dBFS diverge (ESR de l'ordre de 10⁷) et celui à
  −3 dBFS donne 1,3.
- La comparaison utile porte donc sur les **réponses harmoniques D_k** mesurées directement, à
  niveau égal, et non sur les G_n.

**Répétabilité de l'appareil (établi)** : sur les 15 prises du réglage (5 ; 5), l'écart-type
de |D_k| entre prises vaut environ 0,3 dB pour tous les ordres et tous les niveaux (1,1 dB pour
D₄ à −20 dBFS), médiane sur 200 Hz–8 kHz.

**Harmoniques impaires bien reproduites, paires non (établi)** : écart absolu médian entre
modèle et appareil, en dB, sur 200 Hz–8 kHz, 8 prises.

| niveau | bras | D₁ | D₂ | D₃ | D₄ | D₅ |
|---|---|---|---|---|---|---|
| −20 dBFS | A48 | 0,21 | 12,5 | 0,28 | 12,8 | 0,36 |
| −20 dBFS | B192 | 0,35 | 16,6 | 0,57 | 15,9 | 0,75 |
| −10 dBFS | A48 | 0,25 | 4,6 | 0,34 | 5,1 | 0,44 |
| −10 dBFS | B192 | 0,35 | 6,0 | 0,61 | 7,5 | 0,73 |
| −3 dBFS | A48 | 0,27 | 5,0 | 0,41 | 4,7 | 0,52 |
| −3 dBFS | B192 | 0,40 | 7,1 | 0,76 | 5,8 | 0,98 |

- **Sens de l'erreur (établi)** : les modèles produisent **trop** d'harmoniques paires.
  L'appareil a D₂ à −57 dB sous D₁ à −20 dBFS et à environ −35 dB à −10 et −3 dBFS : son
  écrêtage est presque symétrique. Écart signé médian de D₂ : A48 +12,3, +5,0 et +4,7 dB ;
  B192 +16,1, +3,0 et +5,4 dB (−20, −10 et −3 dBFS).
- Les harmoniques impaires sont à 0,2–1 dB de l'appareil, soit du même ordre que sa
  répétabilité. A48 est un peu plus proche que B192 sur tous les ordres.

## Où naissent les harmoniques paires (`gate_harmonics.py`, établi sur un modèle et une prise)

Modèle `vs100r_gc_film_joint_s925_r2`, prise 17, sinus glissants des trois niveaux. Rapport
D₂/D₁ en dB (énergie, 200 Hz–8 kHz), médiane sur les canaux de chaque signal interne.

| niveau | ampli | sorties des portes (blocs 0–7) | somme des skips | après la dernière 1×1 (sortie) |
|---|---|---|---|---|
| −20 dBFS | −53,0 | −32,6 à −9,5 | −23,4 | −40,0 |
| −10 dBFS | −34,5 | −21,1 à −3,4 | −20,8 | −30,9 |
| −3 dBFS | −35,2 | −16,5 à −2,3 | −20,2 | −30,5 |

- **La porte `tanh(a)·sigmoid(b)` crée les harmoniques paires** : dans chaque bloc, D₂/D₁ monte
  de ses entrées a et b à sa sortie, de 4 à 70 dB (bloc 0 : de −67 à −33 dB à −20 dBFS, de −84 à
  −15 dB à −3 dBFS).
- **La tête ne les crée pas** : les LeakyReLU ne changent pas le rapport (à 0,1 dB près).
- **La dernière couche 1×1 les annule en partie** : elle abaisse D₂/D₁ de 11 à 21 dB. L'excès
  mesuré en sortie est le reste d'une annulation imparfaite d'harmoniques paires fabriquées en
  grande quantité à l'intérieur du réseau.

## Ce que cela suggère

- **Établi** : l'excès de D₂ dépend surtout de la prise, donc du réglage, et peu de la graine.
  À −20 dBFS, il va de +5,1 à +26,2 dB selon la prise pour la graine 42, avec le même profil pour
  les graines 43 et 44 : la prise 16 est la plus forte pour les trois (+26 à +30 dB), la prise 30
  la plus faible (+0,3 à +5,6 dB).
- **Supposé** : l'annulation par la dernière couche 1×1 est réglée pour les réglages fréquents
  à l'entraînement et moins bien ailleurs. La prise 16, la plus en excès, est aussi la seule à un
  couple gain–contour non vu. Mesure qui la départagerait : l'excès de D₂ en fonction du nombre
  de prises d'entraînement au même couple, sur toutes les prises (y compris celles d'entraînement).
- **Supposé** : une porte symétrique (impaire) n'aurait rien à annuler. Mesure qui la
  départagerait : un entraînement avec une porte impaire, sur le même protocole. C'est un
  changement de modèle, à préinscrire.
- **Supposé** : poids perceptif faible à bas niveau, plus net aux niveaux de jeu (2ᵉ harmonique
  du modèle vers −30 dB contre −35 dB pour l'ampli). Non vérifié à l'écoute.

## Non fait

- Le bras C (FFT de 288 000 points sans TF32) n'existait pas encore : à ajouter à la fin de sa
  file.
- Aucune analyse par fréquence fine ni par réglage au-delà des médianes.
- L'analyse interne ne porte que sur un modèle et une prise.
