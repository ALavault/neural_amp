# Pilote C, extension : la « queue » tient-elle hors échantillon ?

**Pré-enregistré le 2026-09-24, avant tout run des graines 50 à 57.** Prolonge
`pilot_C_common_schedule.md`, dont la dernière section (« La prochaine expérience discriminante »)
écrit la prédiction ; ce document la rend exécutable sans rien y changer, et comble les trois
trous qu'elle laissait : la coupure chiffrée, le cas limite, et une table d'issues qui partitionne
l'espace (leçon du pilote D).

## Question

Sur les graines 42 à 49, le calendrier fixe donne un rapport sd(fixe)/sd(décide) de 0,56. La
lecture post hoc : il ne resserre rien pour les graines dont le pas est divisé tard, et coupe une
queue de catastrophes chez celles où il est divisé tôt (0,54 contre 0,98, coupure à l'époque 200
choisie après coup). Cette lecture tient-elle sur huit graines qu'elle n'a pas vues ?

## Dispositif

- **Graines 50, 51, 52, 53, 54, 55, 56, 57**, fixées par ce commit. Aucune autre graine ne
  remplacera une graine en échec : une run qui échoue est relancée sous un nouveau run_id
  (`_r2`), une seule fois, et l'échec reste dans `demo/RUNS.jsonl`.
- **Bras et drapeaux identiques au pilote C** (`scripts/product_pilotC_queue.sh`) :
  `--model ssm-wavenet --discretization zoh --honor-optim --deterministic --seed S`, sans
  `--split-seed` (l'effet de graine étudié inclut la partition). « Fixe » ajoute
  `--lr-halvings 203 265 422 486 565 608 661 722 --no-early-stopping --max-steps 5950`.
- **Deux phases, dans cet ordre.**
  1. Les huit bras « décide » (`scripts/product_pilotC_ext_decide_queue.sh`).
  2. Un commit qui classe chaque graine (`pilot_C_extension_groups.json`, écrit par
     `pilot_C_extension_verdict.py --classify`). La file de la phase 2 refuse de démarrer si ce
     fichier n'est pas dans `HEAD`.
  3. Les huit bras « fixe » (`scripts/product_pilotC_ext_fixe_queue.sh`).
- Aucune reprise (`--resume`) : une run reprise n'est pas bit à bit égale à une run d'un seul
  tenant.

## Règle d'affectation

- **Grandeur** : époque de la première division du learning rate du bras « décide » de la graine,
  lue dans son `metrics.csv` par `halving_epochs` de `pilot_C_verdict.py` (pas ÷ 7, arrondi).
- **Coupure** : **précoce si < 200, tardive si ≥ 200.** C'est la coupure post hoc de la section
  « Ce que le chiffre agrégé cachait » ; elle devient ici une règle fixée avant les données.
  Vérifiée : elle redonne le classement d'origine (44, 47, 48, 49 précoces ; 42, 43, 45, 46
  tardives).
- **Cas limite** : un bras « décide » sans aucune division est classé tardif et signalé. (Il ne
  devrait pas exister : la patience du plateau, 20 époques, est plus courte que celle de l'arrêt,
  50.)

## Quantités

Pour chaque groupe g d'au moins **3 graines** : R_g = sd(log ESR fixe) / sd(log ESR décide) sur
les graines du groupe, ESR de test du dernier checkpoint. **Aucun rapport n'est rapporté pour un
groupe de moins de trois graines** (réserve déjà écrite : la règle utilise une grandeur du bras
libre, les groupes peuvent être déséquilibrés).

## Prédiction et table d'issues

Prédiction (écrite le 2026-09-22) : **R_précoce ≤ 0,6 et R_tardif ≥ 0,9.**

Chaque rapport tombe dans une classe : bas (≤ 0,6), moyen (entre 0,6 et 0,9), haut (≥ 0,9).

| R_précoce ＼ R_tardif | bas | moyen | haut |
|---|---|---|---|
| **bas** | réfutée : le calendrier resserre partout | partiellement soutenue | **soutenue** |
| **moyen** | réfutée : sens inverse | indécidable | partiellement soutenue |
| **haut** | réfutée : sens inverse | réfutée : sens inverse | réfutée : aucun resserrement |

- **Soutenue** : la lecture par la queue survit hors échantillon.
- **Réfutée, le calendrier resserre partout** : c'est le calendrier lui-même qui resserre, pas la
  coupure d'une queue.
- **Réfutée, aucun resserrement** : l'intermédiaire de 0,56 ne tenait qu'aux graines 48 et 49.
- **Un groupe de moins de trois graines** : prédiction **non évaluable**, rapportée comme telle ;
  pas de repli sur une autre coupure.

## Descriptifs, hors verdict

- Rapport sur les huit nouvelles graines ensemble, à comparer à 0,56.
- Rapport sur les seize graines, **seulement si la porte de reproductibilité du chemin SSM Big
  Muff est franchie** (`diagnosis/seeds/repro_gate.md`) : le banc a changé depuis les graines 42 à
  49 ; sans égalité bit à bit, les deux lots ne se mélangent pas.
- Porte du dispositif, comme au pilote C : chaque bras « fixe » divise exactement aux huit époques
  déclarées.

## Ce que l'extension ne pourra pas dire

- Huit graines, un appareil, une architecture ; groupes de taille inconnue d'avance.
- La coupure à 200 reste celle trouvée après coup sur 42 à 49 ; seule son application aux
  nouvelles graines est hors échantillon.
- La confusion du pilote C demeure : en « fixe », l'ensemble d'entraînement change encore avec la
  graine.
