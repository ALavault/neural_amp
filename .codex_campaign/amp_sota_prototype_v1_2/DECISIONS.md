# Décisions

- `SOTA12-D-001` — Conserver `full_island_x2` Kaiser comme dépendance AA déjà
  qualifiée, sans le requalifier dans cette lignée.
- `SOTA12-D-002` — Écarter les approximants et l'equiripple v1.1 : leur gate
  valide était négatif.
- `SOTA12-D-003` — Tester d'abord le modèle complet figé
  `slow_long_tcn_x2` sur trois systèmes, trois seeds et des sources nouvelles.
- `SOTA12-D-004` — Ordonner les gates `dynamic_primary`,
  `two_clippers_primary`, puis `static_primary`; arrêter avant le système suivant
  dès le premier triplet négatif.
- `SOTA12-D-005` — Compter le pré-roll de 14 400 échantillons après la latence
  déclarée de 32, donc scorer à partir de l'index 14 432.
- `SOTA12-D-006` — Si la compétence passe, isoler la valeur lente avec le même
  graphe et la même initialisation, mais une modulation exactement nulle.
- `SOTA12-D-007` — Ne permettre aucun retry, reprise de run, accès physique,
  confirmation, FM9 ou sélection humaine avant les gates correspondants.
- `SOTA12-D-008` — Réserver ADAA analytique, fine-tuning alias-aware et filtre
  demi-bande IIR à une nouvelle lignée motivée par un échec alias/coût ultérieur.
- `SOTA12-D-009` — Exécuter chaque étape scientifique depuis un worktree propre
  et refuser tout fichier décisionnel absent du snapshot de provenance.
- `SOTA12-D-010` — Après la compétence, réévaluer les checkpoints candidats sur
  un split `SLOW_VALUE_EVAL` frais et disjoint; ne pas réutiliser
  `INTERNAL_DEV` pour l'intervalle de valeur lente.
- `SOTA12-D-011` — Préserver le plan croisé : mêmes tirages bootstrap de seed et
  d'épisode dans les trois systèmes fixes, avec égalité exacte des entrées et
  cibles candidat/contrôle vérifiée depuis les NPZ.
- `SOTA12-D-012` — Compter les fixtures produites par les tests comme
  non décisionnelles; elles ne peuvent servir ni à l'entraînement ni à la
  sélection. Le préflight doit conserver zéro sortie synthétique éligible.
- `SOTA12-D-013` — Enregistrer la première invocation du préflight comme
  `INVALID` d'infrastructure : la collecte de `make test` a rencontré quatre
  modules historiques absents du commit. Aucun gate, run ou résultat
  scientifique n'ayant existé, corriger seulement la fermeture du snapshot,
  committer le correctif, puis réinvoquer le préflight inchangé depuis un
  worktree propre.
- `SOTA12-D-014` — Après fermeture du snapshot historique et validation de 340
  tests, accepter la tentative 002 du préflight. Autoriser uniquement le
  triplet `dynamic_primary`; les autres systèmes et toute donnée physique
  restent fermés jusqu'à leur gate littérale.
- `SOTA12-D-015` — Superséder prospectivement la lignée par
  `AMP-QUALITY-TEACHER-v1` avant `dynamic_primary`. Préserver le préflight passé,
  les protocoles, le ledger et le constat de zéro run scientifique; ne pas
  requalifier la tentative d'infrastructure invalide en verdict de campagne.
