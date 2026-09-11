# État

- Lignée : `AMP-QUALITY-ARCH-v1`
- Statut : terminal `NO-GO-ARCH` à la gate mécaniste
- Parent : `FSSR-QUALITY-AA-v2`, terminal `GO-QUALITY-AA-v2`
- Backend AA par défaut : `full_island_x2`, portée synthétique seulement
- Snapshot SOTA : quatre comparateurs exécutables retenus; S6 primaire rejeté
  techniquement avant gel
- Side threads : six familles intégrées; super-résolution explicitement rejetée
- Données : quatre dispositifs 48 kHz présents; tests Blackstar/UA1176 verrouillés
- `EXTERNAL_REPORT_ONLY` : verrouillé
- Squelettes natifs : 24/24 valides; six profils `max` sélectionnés sans audio;
  cascade limitant à p95 RTF 0,7373604375
- Faisabilité d'entraînement : cinq candidats admissibles; `selective_s6_x2`
  rejeté techniquement à 20,16 h projetées/5 000 updates
- Comparateurs : quatre factories exécutables; parité source NablAFx acquise,
  auraloss 0.4.0 épinglé, anticipations TFiLM alignées explicitement
- Formes physiques : 11/11 charges passent; cascade limitant à 4,598 h
  projetées/5 000 updates, sans lecture audio
- Gate mécaniste : 10/10 trajectoires valides, 0/5 candidat admissible; aucun
  run physique autorisé
- Suite : steering requis pour ouvrir une nouvelle lignée compétence-first;
  aucune reprise ou modification de v1
- Validation finale : `make test` 417/417; `make lint` vert sur 217 fichiers
