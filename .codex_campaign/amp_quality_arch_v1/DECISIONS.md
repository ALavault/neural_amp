# Décisions

- `ARCH-D-001` — Ouvrir une nouvelle lignée d'architecture sans modifier les
  verdicts des lignées R1/R2/QUALITY-AA.
- `ARCH-D-002` — Limiter la revendication SOTA au meilleur comparateur ouvert,
  reproductible et réentraîné localement sur les mêmes splits.
- `ARCH-D-003` — Retenir initialement cinq comparateurs complémentaires : NAM A2 Full,
  Wright LSTM-64, TCN-TFiLM, S4-TFiLM et S6 sélectif.
- `ARCH-D-004` — Fulltone et Big Muff sont développement historique; leurs tests
  ne peuvent pas être présentés comme prospective. Blackstar et UA1176 portent
  la frontière prospective scellée.
- `ARCH-D-005` — Aucun run scientifique n'est autorisé avant intégration des
  rapports d'idées et gel du protocole exécutable.
- `ARCH-D-006` — Intégrer six familles issues des side threads, avec
  `phys_s6_tcn_x2` comme hypothèse principale mais sans privilège de sélection.
- `ARCH-D-007` — Rejeter la super-résolution 48→192 kHz de cette lignée faute de
  vérité terrain physique 192 kHz.
- `ARCH-D-008` — Dimensionner les candidats sans audio par quatre profils
  croissants et retenir, famille par famille, le plus grand profil dont le
  squelette C++ passe p95 RTF ≤0,80 au bloc 64.
- `ARCH-D-009` — Le benchmark aveugle du 28 août 2026 retient `max` pour les six
  familles. Le cas limitant est le cascade à p95 RTF 0,7373604375, 22 218
  paramètres et 32 échantillons de latence; aucune mesure audio ou ESR n'a servi
  à cette décision.
- `ARCH-D-010` — Rejeter `selective_s6_x2` avant gel pour dépassement du budget
  d'entraînement (20,16 h projetées contre 6 h). Conserver le comparateur S6
  officiel, ne pas remplacer la famille rejetée après observation, et qualifier
  les losses sur les cinq candidats techniquement admissibles.
- `ARCH-D-011` — Retirer le comparateur S6 sélectif avant gel car le code primaire
  épinglé n'est pas exécutable et une reconstruction ne serait pas une
  reproduction loyale. Conserver NAM A2, Wright LSTM, TCN-TFiLM et S4-TFiLM;
  la matrice de qualification réserve donc au plus 36 IDs.
- `ARCH-D-012` — Reproduire localement TCN–TFiLM et S4–TFiLM sans la dépendance
  `rational` inutilisée et exiger la parité contre le commit NablAFx. Aligner
  leurs anticipations TFiLM par des délais conservateurs de 508 à 1 270
  échantillons; ne pas leur appliquer la gate de latence du candidat final.
- `ARCH-D-013` — Geler 8 192 échantillons scorés et des contextes propres aux
  familles, puis accepter les onze charges après un préflight sans audio : le
  cas limitant, cascade max, projette 4,598 h/5 000 updates sous le cap de 6 h.
  Si les cinq candidats mécanistes survivent, les variantes NablAFx comptent
  comme 44 subtrajectoires de modèle dans 36 IDs de qualification; sinon les
  candidats rejetés sont supprimés sans remplacement. Les variantes sont
  sélectionnées globalement sur les deux validations.
- `ARCH-D-014` — Arrêter AMP-QUALITY-ARCH-v1 à la gate mécaniste avec
  `NO-GO-ARCH`. Les cinq candidats échouent; aucune donnée physique, loss ou
  comparaison SOTA ne doit être lancée. Conserver les runs et recommander une
  nouvelle lignée compétence-first plutôt qu'une reprise ou un relâchement de
  seuil post hoc.
