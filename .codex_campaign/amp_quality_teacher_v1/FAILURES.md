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
