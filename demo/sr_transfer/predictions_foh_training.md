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

## Verdict (2026-09-30, trois partitions)

Chiffres recopiés des enregistrements `demo/nablafx_bench/paired_ssm_wavenet_s{925,736,688}.json`
(ZOH), `demo/nablafx_bench/paired_ssm_wavenet_foh_s{925,736,688}.json` (FOH), et des sorties
`demo/sr_transfer/sr_transfer_paired_ssm_wavenet_foh_sXXX_last_plain.json` de
`scripts/product_sr_transfer.py --run-id paired_ssm_wavenet_foh_sXXX --plain-test`. La jumelle
ZOH relue en FOH est `sr_transfer_paired_ssm_wavenet_sXXX_last_asfoh.json` (conversion actuelle
de `FIX_foh.md`, pas `_asfoh_v1`).

**Runs.** Chaque run FOH a les mêmes graine, partition, drapeaux et `val_indices` que sa jumelle
ZOH ; seule `--discretization foh` diffère. Code a0a4a4f, pilote 595.91.07, une session sans
reprise chacun. Le run à 925 a tourné le 29/09 ; ceux à 736 et 688, le 30/09, après environ
16 h d'attente du GPU, pris par un autre projet.

| partition | pas (FOH / ZOH) | arrêt, époque (FOH / ZOH) | bascules de polarité (FOH / ZOH) |
|---|---|---|---|
| 925 | 6 818 / 7 049 | 973 / 1 006 | 72 / 11, 73 |
| 736 | 5 964 / 6 531 | 851 / 932 | aucune / 9, 12 |
| 688 | 8 967 / 7 266 | 1 280 / 1 037 | aucune / 9, 10, 11 |

**Non-infériorité à 48 kHz : INFÉRIEUR.** d = ln(ESR_ZOH / ESR_FOH), ESR de test du dernier
checkpoint :

| partition | ESR ZOH | ESR FOH | d |
|---|---|---|---|
| 925 | 0,04385 | 0,03947 | +0,105 |
| 736 | 0,04999 | 0,04148 | +0,187 |
| 688 | 0,04245 | **0,13668** | **−1,169** |
| moyenne | | | **−0,292** |

Le seuil est −ln 1,15 = −0,140. La moyenne vaut −0,292, en dessous : le libellé pré-enregistré
est « inférieur ». Il tient entièrement au run à 688. Aux deux autres partitions, le run FOH est
plus bas que sa jumelle ZOH ; la moyenne sur ces deux seules partitions (+0,146) n'est pas la
quantité pré-enregistrée.

**Transfert sans correction** (`--plain-test`) :

| partition | 96 kHz, écart au modèle 48 kHz (seuil 5·10⁻⁴) | 44,1 kHz contre l'appareil, FOH / jumelle ZOH relue en FOH (seuil 1,15) |
|---|---|---|
| 925 | 2,94·10⁻⁴ : tenu | 0,03505 / 0,03912 = 0,896 : tenu |
| 736 | 2,59·10⁻⁴ : tenu | 0,03728 / 0,04604 = 0,810 : tenu |
| 688 | 3,16·10⁻⁴ : tenu | 0,13339 / 0,03757 = **3,55 : non tenu** |

- Le critère à 96 kHz est tenu pour les trois runs. Il compare chaque modèle à lui-même à
  48 kHz et ne dépend donc pas de sa qualité : l'écart vaut 2,6 à 3,2·10⁻⁴, sous le plancher
  de 5·10⁻⁴.
- Le critère à 44,1 kHz est tenu à 925 et 736, et non tenu à 688, où il hérite de l'ESR élevée du
  run à 48 kHz.

**Descriptif, run FOH à 688.** Il est moins bon aussi en validation : meilleure perte 0,0345
contre 0,0295 pour sa jumelle. Ce n'est donc pas un écart propre au test. Il a tourné plus
longtemps (8 967 pas contre 7 266, arrêt anticipé à l'époque 1 280 contre 1 037), et sa MR-STFT
de test est aussi plus haute (0,527 contre 0,399). La garde de polarité n'a pas basculé. La
cause n'est pas établie : ce texte ne l'examine pas.

**Ce que le verdict ne dit pas.**

- Une graine, trois partitions. L'écart entre partitions (+0,11, +0,19, −1,17) domine :
  un seul run fait passer le libellé de « non inférieur » à « inférieur ». Le bruit apparié entre
  deux règles de discrétisation n'est pas connu (section précédente).
- La lecture FOH d'un checkpoint ZOH, qui ne demande aucun réentraînement, reste la solution
  de `FIX_foh.md` pour la montée en fréquence. Ce verdict ne la remet pas en cause.
- Les runs FOH tournent au code a0a4a4f, sans le plancher des pôles de 2d49ac7. Aucun n'a produit
  de NaN.

### Examen post hoc du run FOH à 688 (2026-09-30, descriptif)

Deux explications mécaniques ont été testées. Aucune ne tient.

- **Un simple décalage temporel** (comme pour F-Fuzz ou le VS100R) : écarté.
  `demo/sr_transfer/foh_lag.py`, sortie `foh_lag.json` : le retard optimal sortie → cible vaut 0
  dans 10 segments de test sur 12, et −1 dans les 2 autres. Réaligné, l'ESR intérieure ne passe
  que de 0,135 à 0,133, contre 0,040 pour la jumelle ZOH. Le modèle FOH à 688 est réellement
  moins bon, il n'est pas mal calé.
- **Des pôles trop rapides**, avec le code a0a4a4f sans plancher : écarté. Aucun mode n'a
  s = −Re(λ·dt) > 10 ; le s maximal vaut 0,6, contre 0,5 pour la jumelle ZOH, et les médianes
  sont égales (0,0100 et 0,0101). Mesure faite sur le `last.ckpt` des six runs appariés.

Reste descriptif : la perte de validation du run FOH à 688 remonte entre les époques 5 et 10
(0,437 puis 0,469), alors que sa jumelle descend (0,374 puis 0,262). Il ne rattrape jamais tout à fait :
0,035 contre 0,030 à l'époque 1 000. Sa jumelle avait basculé de polarité aux époques 9 à 11,
lui jamais. La cause reste inconnue. Une trajectoire d'entraînement divergente sans mécanisme
propre à la FOH est compatible avec ces faits ; elle n'est pas démontrée.
