# Échecs

- `PREFLIGHT_ATTEMPT_001_INVALID.json` — tentative d'infrastructure invalide :
  `data-audit` et 424 tests ont passé, puis `make lint` a rejeté le groupement
  d'imports du snapshot propre. Aucun gate, run scientifique ou waveform n'a
  été ouvert.
- `PREFLIGHT_ATTEMPT_002_INVALID.json` — tentative d'infrastructure invalide :
  `data-audit` a passé; 424 tests sur 425 ont passé, puis le test d'état exigeait
  encore « préflight non exécuté » après l'enregistrement canonique de la
  tentative 001. Aucun gate, run scientifique ou waveform n'a été ouvert.

Aucun échec scientifique enregistré.

- `DATA_AUDIT_INVALID.json` — gate données `INVALID`. Après dix couples
  Fulltone/Ampeg conformes, le premier couple Big Muff publié a échoué sur
  `synchronization marker is missing`. Dans les fenêtres gelées, le dry atteint
  seulement `3.0517578125e-05` au début et à la fin; le wet reste sous `0.001`,
  face au seuil `0.05`. Les deux fichiers ont 14 994 001 échantillons à
  44,1 kHz, mais le protocole ne permet pas de remplacer le contrôle marqueur
  après observation. Aucun membre `test` ni run scientifique n'a été ouvert.
