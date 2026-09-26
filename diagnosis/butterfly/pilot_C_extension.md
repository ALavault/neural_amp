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

## Verdict (2026-09-26, les seize runs faites)

`cd diagnosis/butterfly && ../../.venv/bin/python3 pilot_C_extension_verdict.py`, sortie dans
`pilot_C_extension_results.json`. Classement commité avant tout bras « fixe »
(`pilot_C_extension_groups.json`).

| graine | 1ʳᵉ division libre | groupe | ESR décide | ESR fixe | ln(fixe / décide) |
|---|---|---|---|---|---|
| 50 | 164 | précoce | 0,0596 | 0,0529 | −0,120 |
| 52 | 197 | précoce | 0,0495 | 0,0450 | −0,095 |
| 54 | 126 | précoce | 0,0563 | 0,0425 | −0,283 |
| 57 | 199 | précoce | 0,0641 | 0,0614 | −0,043 |
| 51 | 217 | tardif | 0,0778 | 0,0851 | +0,089 |
| 53 | 206 | tardif | 0,0635 | 0,0665 | +0,047 |
| 55 | 232 | tardif | 0,0603 | 0,0670 | +0,106 |
| 56 | 233 | tardif | 0,0768 | 0,0830 | +0,078 |

- **Porte du dispositif** : les huit bras « fixe » divisent exactement aux huit époques déclarées.
- **R_précoce = 1,53 (haut), R_tardif = 1,02 (haut).**
- **Verdict : réfutée, aucun resserrement.** D'après la table d'issues, l'intermédiaire de 0,56
  des graines 42 à 49 ne tenait qu'aux graines 48 et 49.
- **Fragilité.** Chaque rapport compare deux écarts-types calculés sur quatre points, de l'ordre
  de 0,11 à 0,17 en log : il est instable. Le libellé, lui, ne bouge pas quand on retire une
  graine : de 1,17 à 2,86 chez les précoces, de 0,91 à 1,19 chez les tardives, toujours « haut ».
- **Cas limites.** Les graines 52 et 57 sont précoces à 3 et 1 époque de la coupure. La règle
  fixée d'avance s'applique telle quelle ; aucun verdict n'est recalculé avec une autre coupure.

**La lecture par la queue n'a plus rien à couper.** Aucun bras « décide » des nouvelles graines ne
rechute : le pire ESR libre vaut 0,078, contre 0,158 et 0,178 pour les graines 48 et 49. Dans le
groupe précoce, sd(log ESR décide) vaut 0,11, contre 0,48 pour 44, 47, 48 et 49. Le calendrier
fixe ne resserre pas le gros de la distribution ; il l'élargit. Sur les huit nouvelles graines,
sd(log ESR) passe de 0,15 à 0,26, parce que les précoces y gagnent et les tardives y perdent
(section suivante). La queue qu'il coupait sur 42 à 49 n'est pas réapparue en huit graines.

**Descriptifs, hors verdict :**

- huit nouvelles graines ensemble : R = 1,71 (graines 42 à 49 : 0,56) ;
- seize graines (porte de reproductibilité franchie) : R = 0,68, sans étiquette ;
- l'effet moyen du calendrier fixe disparaît presque : ln(fixe / décide) moyen −0,028 sur les
  nouvelles graines, contre −0,219 sur 42 à 49.

## Après le verdict : post hoc, non pré-enregistré

`pilot_C_extension_channel.py`, sortie dans `pilot_C_extension_channel.json`, écrit après le
verdict.

- **Contrôle** : chaque partition est reproduite, et la perte de validation du bras « décide » y
  est recalculée à 0,1 à 1,2 % de la valeur journalisée. Une mauvaise partition s'écarterait
  d'environ 30 %.
- **Environ sept corrélations** sont calculées sur huit points. Les p ne sont pas corrigés.

**Seize graines sans 48 et 49** (partition post hoc) : R = 1,16 sur les quatorze autres
(sd décide 0,248, sd fixe 0,288). Hors de ces deux rechutes, le calendrier fixe ne resserre rien.

**Les deux lectures post hoc des graines 42 à 49 ne se reproduisent pas.**

| prédicteur → ESR décide | graines 42 à 49 | graines 50 à 57 |
|---|---|---|
| date de la 1ʳᵉ division libre | ρ = −0,81 (p 0,015) | **ρ = +0,71** (p 0,047) |
| part de trames calmes de la partition | ρ = +0,81 (p 0,015) | ρ = +0,31 (p 0,46) |

- La division précoce n'annonce plus un mauvais modèle libre : le signe s'inverse.
- La part de trames calmes n'ordonne plus l'ESR libre. Le « test du canal » de
  `pilot_C_common_schedule.md` (la partition agit par l'ordonnanceur) n'est donc pas reproduit.

**Un motif nouveau, à ne pas lire comme un résultat.** Le signe de ln(fixe / décide) suit le
groupe :
- sur les huit nouvelles graines, les précoces gagnent au calendrier fixe et les tardives y
  perdent ;
- sur les seize graines, 15 suivent ce motif ; seule la 43 fait exception ;
- ρ(1ʳᵉ division libre, gain) = +0,93 (p 0,001) sur 50 à 57, +0,69 (p 0,058) sur 42 à 49 ;
- ρ(1ʳᵉ division libre, ESR fixe) = +0,91 (p 0,002).

Trois réserves, dans l'ordre de leur poids :

1. **Post hoc.** La variable a été choisie après avoir vu le motif des signes.
2. **Longueur d'entraînement confondue.** Le bras « fixe » tourne 850 époques sans arrêt anticipé ;
   les bras « décide » s'arrêtent entre les époques 634 et 918. La date de la 1ʳᵉ division et
   l'époque d'arrêt sont corrélées : ρ = +0,67. L'époque d'arrêt ordonne aussi le gain : ρ = +0,83.
   On ne peut pas séparer « diviser plus tard aide » de « le calendrier fixe donne plus d'époques
   aux graines dont le bras libre s'est arrêté tôt ».
3. **Régression vers la moyenne.** Le gain contient l'issue du bras « décide », ce que
   `pilot_C_common_schedule.md` interdisait justement de corréler. Réponse partielle seulement :
   sur les nouvelles graines, les précoces ont le *meilleur* ESR libre (ρ = +0,71). La régression
   vers la moyenne prédirait donc qu'elles perdent au calendrier fixe, et elles y gagnent.

Prédiction candidate, pour une éventuelle préinscription : **le gain apparié ln(fixe / décide)
décroît avec le retard que le calendrier fixe impose à la première division (203 moins la date
libre), à longueur d'entraînement égale entre les bras.**
