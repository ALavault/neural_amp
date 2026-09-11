# Décisions

- `ARCH3-D-001` — Créer une lignée prospective sans modifier ou reprendre v2.
- `ARCH3-D-002` — Utiliser v2 uniquement pour formuler les hypothèses de v3.
- `ARCH3-D-003` — Refuser tout entraînement avant une gate de représentabilité
  calculée exclusivement sur le train.
- `ARCH3-D-004` — Séparer les systèmes primaires progressifs du stress composite
  v2 collé aux rails.
- `ARCH3-D-005` — Étudier une tête de gain non bloquante, un état lent et un TCN
  à champ réceptif étendu sous conditions appariées.
- `ARCH3-D-006` — Limiter l'exploration à trois rounds, 324 GPU-heures et 50 GiB.
- `ARCH3-D-007` — Interdire 192 kHz supposé, FM9, nouvelles captures et holdouts.
- `ARCH3-D-008` — Utiliser un TBPTT causal par chunks de 8192 échantillons avec
  état porté entre chunks et graphe explicitement détaché à chaque frontière.
- `ARCH3-D-009` — Conserver les trois ablations après un banc GPU non scientifique
  recalculé sur la boucle exact-chunk : le round 1 complet plus la compétence
  pessimiste projettent 66,428179 GPU-heures,
  sous le plafond de 324 GPU-heures.
- `ARCH3-D-010` — Au round 1, exiger les gardes gain/corrélation sur chaque système
  et seed aux checkpoints 10k et 15k avant toute comparaison ; sélectionner ensuite
  par ESR médian normalisé condition par condition, avec départage à 1 % par le plus
  petit nombre de paramètres.

- `ARCH3-D-011` — Fermer v3 en `INVALID` sans reprendre la matrice; conserver les 15 trajectoires complètes comme hypothèses historiques uniquement.
