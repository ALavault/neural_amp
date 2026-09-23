# Test D : l'ultrasonore d'un premier étage change-t-il le second dans la bande audio ?

Voie produit, mesure sur systèmes synthétiques connus (M1), aucune revendication sur un
appareil réel. Écrit le 2026-09-23 **avant** toute exécution. Hypothèse :
`reports/neural_amp-research/2026-09-23/brainstorm_supersampling.md`, piste D. Littérature :
aucune mesure trouvée sur une chaîne d'effets (`verif_biblio_supersampling.md`).

## Dispositif (`scripts/product_chain_ultrasonic.py`)

- Tout tourne à 192 kHz. Les filtres des systèmes Wiener/Hammerstein sont définis en
  échantillons : leur réponse en hertz n'est pas celle qu'ils auraient à 48 kHz. Limite
  assumée : le test ne compare que deux versions du **signal intermédiaire**.
- Excitation : `procedural_plucks` (principal, guitare synthétique), `multisine`,
  `two_tone`, `colored_noise`, 2 s à 192 kHz, amplitude du corpus, **limitée à 20 kHz** par
  un passe-bas avant la chaîne (un DI réel n'a rien au-dessus).
- Étage 1 → signal intermédiaire en deux versions :
  - `full` : tel quel, 192 kHz ;
  - `lim` : décimé à 48 kHz par la chaîne gelée `derive_reference_rates` (passe-bas à
    0,9 × Nyquist cible), puis remonté à 192 kHz (`resample_poly`). C'est ce qu'une
    capture ou un modèle à 48 kHz peut transmettre au mieux.
- Étage 2 appliqué aux deux. Les deux sorties sont ramenées à 48 kHz par la même chaîne
  gelée, puis comparées : différence en dB, 10·log10(Σ(z_lim − z_full)² / Σ z_full²), sur
  l'intérieur (0,1 s retirée à chaque bord).
- Paires : étage 1 ∈ {tanh, asymmetric_waveshaper, memory_clipper, hammerstein}, étage 2 ∈
  {tanh, asymmetric_waveshaper, wiener, slow_sag}, soit 16 paires par excitation.
- Témoins : (i) étage 2 = identité (seule la différence de filtrage subsiste) ; (ii)
  « baffle » : passe-bas Butterworth ordre 4 à 5 kHz après l'étage 2.

## Échelle de lecture (fixée avant)

L'erreur actuelle de nos meilleurs modèles Big Muff est un ESR d'environ 0,04, soit
−14 dB. Un effet ultrasonore est **pertinent pour le produit** s'il atteint −30 dB (16 dB
sous cette erreur) ; **négligeable** sous −50 dB ; entre les deux, rapporté tel quel.

## Prédictions

- **D0 (témoin identité)** Différence < −60 dB pour toutes les excitations : la chaîne de
  filtrage elle-même ne crée pas d'écart dans la bande.
- **D1 (effet principal, plucks)** Médiane des 16 paires entre −60 et −30 dB, et au moins
  une paire au-dessus de −40 dB (étage 1 le plus dur, asymmetric_waveshaper ou
  memory_clipper, suivi d'un étage 2 memoryless). Réfutée vers le bas si toutes les paires
  sont sous −60 dB (l'ultrasonore ne compte pas) ; vers le haut si la médiane dépasse
  −30 dB.
- **D2 (baffle)** Le passe-bas à 5 kHz réduit la différence de moins de 10 dB en médiane :
  les produits d'intermodulation d'ordre pair (différences de fréquences) tombent aussi
  dans le grave. Prédiction faible.
- **D3 (excitation)** `two_tone` et `multisine`, plus stationnaires et plus riches en
  hautes fréquences d'entrée, donnent des différences plus grandes que `plucks` (dont
  l'énergie est concentrée sous 3 kHz).

Aucune paire ni excitation n'est écartée après coup ; tout est rapporté.
