# Entraîner en FOH : non-infériorité à 48 kHz (option (b) de FIX_foh.md)

Voie produit. **Pré-enregistré le 2026-09-24, avant tout run.** Optionnel, en fin de file GPU :
la relecture FOH d'un checkpoint ZOH résout déjà la montée en fréquence sans réentraînement.

## Dispositif

- Trois runs `paired_ssm_wavenet_foh_s{925,736,688}`, avec les drapeaux exacts de
  `paired_ssm_wavenet_s{925,736,688}` plus `--discretization foh`.
- FOH et ZOH ont les mêmes paramètres et consomment les mêmes tirages
  (`test_same_parameters_and_draws_as_zoh`). À graine 42 et partition égale, chaque run FOH part
  donc exactement de l'initialisation de sa jumelle ZOH ; seule la règle de discrétisation
  diffère.
- Effet apparié : d = ln(ESR_ZOH / ESR_FOH), ESR de test du dernier checkpoint, par partition.

## Prédictions

- **Non-infériorité à 48 kHz**, marge ln 1,15 : moyenne des trois d ≥ −ln 1,15 → **non
  inférieur** ; sinon **inférieur**. Aucune prédiction de supériorité : la question est de savoir
  si la FOH coûte quelque chose, pas si elle gagne.
- **Transfert sans correction** (`product_sr_transfer.py --plain-test`) : à 96 kHz, écart au
  modèle 48 kHz du même run ≤ 5·10⁻⁴ dans la bande commune, pour les trois runs. C'est
  l'ordre du plancher non linéaire mesuré sur les modèles ZOH relus (`predictions_floor.md`). À
  44,1 kHz contre l'appareil, ESR ≤ 1,15 × celle de la jumelle ZOH relue en FOH.

## Ce que cela ne dira pas

Le bruit apparié entre deux règles de discrétisation n'est pas connu. Sur trois partitions, le
résidu apparié entre architectures valait 0,066 (`diagnosis/seeds/paired_protocol.md`) : un d
moyen de quelques centièmes n'est pas lisible.

## Écart déclaré avant les runs (2026-09-28)

Le pilote NVIDIA est passé de 595.84 à 595.91.07 (même noyau) avant ces runs. Les jumelles ZOH
ont été entraînées sous l'ancien pilote. La porte au bit de `scripts/product_chain_after_reboot.sh`
vérifie, sous le nouveau pilote, le chemin ZOH SSM-WaveNet Big Muff (`pilotC_decide_seed42`),
pas le chemin FOH, qui n'a pas de référence. Les runs tournent au code a0a4a4f, qui n'a pas le
plancher des pôles de 2d49ac7. Un pôle sous-dépassé y donnerait un NaN, que `run_once` consigne
en échec. Les jumelles ZOH, sans ce plancher elles non plus, n'en ont pas eu.
