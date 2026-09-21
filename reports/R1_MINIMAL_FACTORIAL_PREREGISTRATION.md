# Préenregistrement minimal FSSR contre baseline

## Question et portée

Question décisionnelle : à données et budget identiques, la formulation de loss
explique-t-elle l'effondrement du modèle structuré-résiduel S3 observé sur Big
Muff, sans dégrader Fulltone ?

Cette expérience départage l'hypothèse mécaniste d'optimisation de S3 de son
invalidateur minimal. A2 est le contrôle black-box apparié. Le résultat ne peut
pas établir H1/H2 ni un verdict `GO-A`/`GO-B`; il autorise seulement, s'il passe,
le diagnostic d'horizon déjà préenregistré.

## Prérequis

- gate de compétence Wright valide et publié dans le registre de gates actif ;
- verrou diagnostique actif et arbre de travail propre ;
- `INTERNAL_DEV` seulement : Fulltone et Big Muff ;
- test scellé, `INTERNAL_VALIDATION` et `EXTERNAL_REPORT_ONLY` inaccessibles.

Tant que ces prérequis ne sont pas satisfaits, aucune trajectoire factorielle
n'est autorisée. Ce document ne lance et ne réserve aucun job GPU.

## Matrice minimale et contrôles

Exécuter exactement les huit trajectoires déclarées dans
`configs/training/r1_factorial.yaml` :

| Appareil | Modèle | Loss | Seed | Run ID |
| --- | --- | --- | ---: | --- |
| Fulltone | A2 | M4 | 0 | `r1_factorial_fulltone_a2_m4_seed0_v1` |
| Fulltone | A2 | Wright | 0 | `r1_factorial_fulltone_a2_wright_seed0_v1` |
| Fulltone | S3 | M4 | 0 | `r1_factorial_fulltone_s3_m4_seed0_v1` |
| Fulltone | S3 | Wright | 0 | `r1_factorial_fulltone_s3_wright_seed0_v1` |
| Big Muff | A2 | M4 | 0 | `r1_factorial_bigmuff_a2_m4_seed0_v1` |
| Big Muff | A2 | Wright | 0 | `r1_factorial_bigmuff_a2_wright_seed0_v1` |
| Big Muff | S3 | M4 | 0 | `r1_factorial_bigmuff_s3_m4_seed0_v1` |
| Big Muff | S3 | Wright | 0 | `r1_factorial_bigmuff_s3_wright_seed0_v1` |

A2 contrôle l'apprenabilité sous les deux losses. S3/M4 est le contrôle négatif
historique apparié. Fulltone empêche une promotion fondée uniquement sur Big
Muff. Les huit conditions utilisent les mêmes sources, quantité d'audio,
précision `float32`, batch, optimiser, régularisations et politique de checkpoint.

## Budget, métrique et observations de validité

- seed : `0` uniquement, sans ajout adaptatif ;
- budget : exactement 5 000 updates par trajectoire ;
- snapshots communs : 200, 1 000 et 5 000 ;
- métrique primaire : ESR de validation au meilleur snapshot commun ;
- sélection de loss : plus faible médiane de l'ESR S3 sur Fulltone et Big Muff ;
- égalité : écart relatif inférieur ou égal à 1 %, alors Wright gagne ;
- diagnostics obligatoires : `gain_error`, énergie de sortie et cible, ratio
  d'énergie résiduelle, normes de gradient par bloc et finitude des gradients ;
- contraste A2/S3 : rapporté aux mêmes snapshots comme contrôle descriptif, sans
  créer de seuil de promotion supplémentaire.

Les fenêtres ne sont pas des unités indépendantes. Aucun résultat test ne sert à
l'entraînement, au checkpoint ou à cette décision.

## Règle de décision et arrêt

La loss sélectionnée est promue si et seulement si les trois contrôles gelés
passent simultanément :

1. Big Muff S3 : `gain_error > -0.5` ;
2. Big Muff S3 : amélioration ESR d'au moins 25 % par rapport à S3/M4 ;
3. Fulltone S3 : régression ESR d'au plus 5 % par rapport à S3/M4.

La matrice est indivisible : aucune décision n'est publiée avant huit artefacts
valides de 5 000 updates. Une erreur de données, environnement, instrumentation,
gradient non fini, artefact incomplet ou ouverture du test rend l'expérience
`invalid`, jamais `refuted`. Après huit runs valides :

- trois contrôles passés : hypothèse d'optimisation `supported`, loss promue et
  horizon autorisé ;
- au moins un contrôle échoué : hypothèse `refuted` pour cette matrice et arrêt
  des stages diagnostiques ultérieurs ;
- preuve incomplète : `invalid` ou `inconclusive`, sans promotion.

Aucune branche échouée ne libère du budget et aucune seed ou architecture n'est
ajoutée. Tous les runs, y compris échecs, sont inscrits au registre global.

## Sources d'autorité

- matrice et seuils : `configs/training/r1_factorial.yaml` ;
- limites 3/8/4/4 : `configs/r1/protocol.yaml` ;
- évaluation fail-closed : `src/fssr_nam/campaign/r1_gates.py` ;
- exécution déclarative : `scripts/campaigns/run_r1_diagnostic.py` ;
- état et décisions : `.codex_campaign/r1/`.
