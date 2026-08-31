# Décisions

- `AQT-D-001` — Superséder v1.2 avant tout run scientifique, sans altérer son
  protocole, son préflight passé ou son ledger.
- `AQT-D-002` — Limiter « SOTA » au meilleur comparateur ouvert,
  reproductible et réentraîné localement parmi S4-TFiLM large, NAM A2 Full et
  WaveNet dense 16×18.
- `AQT-D-003` — Geler ToneTwist au commit
  `76ae7c875781bc7e2cec2df73b395d1a12d9cdbd`; conserver tous les fichiers
  éligibles, leurs groupes source et la conversion dry/wet jointe.
- `AQT-D-004` — Sélectionner le chemin lent une seule fois sur Ampeg seed 0,
  puis un comparateur global unique sur les trois dispositifs de développement.
- `AQT-D-005` — Ouvrir tous les tests en une étape seulement après quatre
  locks; ne financer ni variante ni retry avec un budget résiduel.
- `AQT-D-006` — Exclure CPU, C++, RTF, mémoire runtime, compression et
  distillation. Le teacher et son contrat de streaming alimenteront un cycle
  ultérieur.
- `AQT-D-007` — Compléter l'AdamW sous-spécifié par les valeurs déjà utilisées
  dans la lignée parent : learning rate `0.001`, weight decay `0.0`.
- `AQT-D-008` — Geler les réglages centraux publiés : Ampeg `C5/R5/O6`,
  Rodent `V10/F5/D5/Normal` et Fuzzy Logic `V10/F5`; conserver les réglages
  Fulltone et Big Muff déjà audités.
- `AQT-D-009` — Corriger prospectivement le basename opérationnel Ampeg en
  `C050_R050_L060`; le contrôle publié reste compression 5, release 5, output
  level 6. Cette correction précède toute lecture waveform et ne change ni
  appareil, ni réglage scientifique, ni split, ni seuil.
