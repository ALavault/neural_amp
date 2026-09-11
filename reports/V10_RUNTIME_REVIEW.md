# Revue runtime v10

Observation statique pendant la confirmation en cours :

- `run_teacher_v10_confirmation.py` remplace les préfixes de run v9 par v10 et
  écrit bien `seed0`, `seed1`, `seed2`.
- `amp_quality_teacher_v9_registry.parse_confirmation_run_id` conserve une
  expression régulière `quality_teacher_v9_`. Elle ne peut donc pas valider les
  événements `quality_teacher_v10_confirmation_train_*` après les runs.
- Cette divergence n'affecte pas les modèles déjà chargés par le processus; elle
  bloque le verrouillage post-run si elle n'est pas corrigée avant cette étape.

Action autorisée après la fin de l'entraînement : adapter le registre v10 à une
fonction de parsing versionnée (ou à un registre v10 dédié), ajouter un test
synthetic ID/seed, puis valider les 12 événements et leurs artefacts. Ne pas
modifier un run existant ni réécrire le ledger.
