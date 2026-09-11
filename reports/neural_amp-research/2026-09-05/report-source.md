---
title: "Deep Research — mémoire, validité et failure mining pour neural_amp"
date: "2026-09-05"
lang: fr
---

# Résultat exécutif

La meilleure prochaine expérience n'est pas « un observer plus gros ». Elle doit
d'abord rendre la comparaison symétrique, puis mesurer où une mémoire lente aide
réellement. L'audit du code a trouvé deux facteurs capables de fausser la sélection :
le teacher conservait son état à travers des chunks de 48 000 échantillons, tandis
que les comparateurs étaient remis implicitement à zéro tous les 12 000; sa
validation était pondérée par échantillons, celle des comparateurs par nombre de
segments. Des coordonnées de sortie identiques ne corrigent pas cette différence
d'historique.

La recommandation est donc un benchmark prospectif à contexte commun : reset de
toutes les familles chaque seconde, préroll identique de 14 400 échantillons,
scoring du suffixe seulement et agrégation pondérée par échantillons. Cette
restriction réduit la portée de la revendication : un résultat positif signifie
« meilleur comparateur ouvert réentraîné localement sous le contexte commun
gelé », pas reproduction exacte de toutes les recettes publiées ni supériorité
perceptuelle universelle.

Parmi les quinze pistes générées dans la revue exploratoire, trois sont prioritaires
après un verdict v9 valide : mesurer l'horizon de mémoire utile, remplacer
l'observer lourd par un observer compact à cadence lente, puis tester un encodeur
d'état initial dry-only. Elles sont falsifiables et peuvent être évaluées sur les
splits train/validation sans ouvrir les tests.

# Question, méthode et limites

Question : quelles limites de représentation, d'apprentissage et de mesure
empêchent une émulation dry/wet causale de reproduire un dispositif analogique,
et quelles expériences bornées séparent ces mécanismes ?

La recherche couvre 2016–2026 et conserve 33 sources primaires ou notices
institutionnelles. Les recherches ont ciblé arXiv, DAFx, PMLR, DBLP, ISMIR,
Frontiers et les dépôts universitaires. Le journal complet des requêtes et des
exclusions est dans `sources/search_log.md`; le corpus dédupliqué est dans
`sources/corpus.json`. Le CLI académique prévu n'était pas disponible, donc une
recherche web ciblée a servi de repli. Cette méthode est une revue exploratoire,
pas une revue systématique PRISMA : il n'y a ni export exhaustif des résultats
bruts, ni double screening, ni méta-analyse.

Les quatre DOI déclarés ont été testés. Trois ont fourni leurs métadonnées
Crossref; le DOI KLANN résout mais ses métadonnées ont été corroborées par la
notice Aalto. Aucun DOI absent n'a été inventé. Le registre
`sources/claim_ledger.md` relie chaque conclusion opérationnelle à ses sources et
indique si la preuve est directe, transférée ou issue de l'audit local.

# Synthèse des preuves

## Aucun vainqueur architectural universel

La comparaison multi-effets de Comunità et collaborateurs montre l'intérêt des
modèles state-space et du conditionnement temporel dans son propre cadre, alors
que l'étude comparative de Simionato et Fasciani rapporte des avantages qui
dépendent de la classe d'effet. Ces résultats n'ont ni les mêmes données, ni les
mêmes budgets, ni exactement les mêmes recettes. Ils soutiennent une sélection
par dispositif et protocole gelé, pas un classement universel.
([Comunità et al., 2025](https://www.frontiersin.org/journals/signal-processing/articles/10.3389/frsip.2025.1580395/full),
[Simionato et Fasciani, 2024](https://arxiv.org/abs/2405.04124))

Les RNN modestes restent des baselines fortes pour l'amplification, comme le
montrent les travaux de Wright, mais ces résultats ne garantissent pas la capture
de toutes les dynamiques lentes. Les résultats historiques sur références SPICE
ne sont pas interchangeables avec des captures de matériel réel.
([Wright et al., 2019](https://dafx.de/paper-archive/details/tieFMcaohHBxl_2yPyIGFA),
[Wright et Välimäki, 2020](https://aaltodoc.aalto.fi/items/dc06100e-514d-41b4-b6b4-272912e872b7),
[Damskägg et al., 2018](https://arxiv.org/abs/1811.00334))

## État transporté et gradient appris sont deux questions différentes

Avec une TBPTT de 12 000 échantillons à 48 kHz, le gradient direct couvre 0,25 s.
Un modèle peut transporter un état au-delà, mais cela ne prouve pas que cet état
a reçu un signal d'apprentissage adéquat. Les travaux sur la TBPTT non biaisée et
adaptative établissent le mécanisme général; leur transfert à l'émulation audio
reste une hypothèse à tester.
([Tallec et Ollivier, 2017](https://arxiv.org/abs/1705.08209),
[Aicher et al., 2020](https://proceedings.mlr.press/v115/aicher20a.html))

S4, S4D, LRU et Mamba fournissent différentes paramétrisations de mémoire longue,
mais aucune ne détermine à elle seule l'horizon utile d'une pédale ou d'un ampli.
Les difficultés rapportées sur les compresseurs optiques rendent plausible une
analyse conditionnée par les régimes attaque/release.
([Gu et al., 2021](https://arxiv.org/abs/2111.00396),
[Gu et al., 2022](https://arxiv.org/abs/2206.11893),
[Orvieto et al., 2023](https://arxiv.org/abs/2303.06349),
[Simionato et Fasciani, 2024](https://arxiv.org/abs/2408.12549))

## La métrique moyenne cache les limites génératives

ESR et MAE résument l'erreur mais ne localisent pas sa cause. Une campagne de
failure mining descriptive doit conserver les fenêtres dans les coordonnées
source et les relier à des facteurs calculés uniquement sur le dry : niveau,
crest factor, centroïde spectral et transientness. Le résidu wet peut ensuite
servir de variable réponse, sans sélectionner une nouvelle architecture sur ces
résultats. Les pertes perceptuelles et multi-résolution publiées justifient une
lecture multi-métrique, pas le remplacement post hoc de la métrique primaire.
([Wright et al., 2020](https://dihana.cps.unizar.es/proceedings/ICASSP/2020/pdfs/0000251.pdf),
[Cassidy et al., 2023](https://www.dafx.de/paper-archive/2023/DAFx23_paper_40.pdf))

Une seconde limite possible est l'aliasing. Les travaux récents proposent le
lissage d'activations et l'antidérivée pour certains modèles neuronaux. Cela
justifie un diagnostic spectral conditionnel, mais pas l'ajout d'un nouveau
traitement à la campagne v9 : ce serait une variante scientifique distincte.
([Sato et al., 2025](https://arxiv.org/abs/2505.04082),
[Mikkonen et al., 2025](https://dafx.de/paper-archive/details/ZjW5x1v0V0iVtRp_jt4bzA))

# Failure mining préenregistrable

L'unité descriptive proposée est une fenêtre source fixe sur validation, jamais
un extrait choisi après inspection des tests. Pour chaque fenêtre : calculer ESR
local, MAE, erreur MR-STFT et erreur d'enveloppe; calculer sur le dry niveau RMS,
crest factor, centroïde spectral et transientness. Produire des quantiles et des
courbes de calibration par dispositif. Les comparaisons candidat–comparateur
doivent être appariées sur la même fenêtre.

Quatre signatures départagent des mécanismes :

1. Erreur croissante avec l'horizon depuis un reset, mais pas avec le niveau :
   mémoire ou initialisation d'état.
2. Erreur concentrée sur fortes transitoires et hautes fréquences : branche rapide,
   resampling ou aliasing.
3. Erreur dépendante du niveau avec gain_error négatif : saturation ou calibration
   d'amplitude insuffisante.
4. Erreur de release après transitoire : dynamique lente mal apprise malgré un
   score moyen acceptable.

Ce mining reste descriptif et inéligible à la sélection v9. Une hypothèse nouvelle
issue de ces graphiques exige une lignée prospective séparée.

# Idées de recherche classées

## I01 — Mesurer l'horizon de mémoire utile

Injecter, sur train/validation, des resets contrôlés à plusieurs distances avant
la zone scorée (0,25; 0,5; 1; 2; 4 s) et tracer la récupération de l'erreur par
dispositif. Prédiction : si la voie lente apporte réellement une information
utile, son avantage augmente avec l'historique disponible puis atteint un
plateau. Falsification : courbe plate ou avantage limité aux premières dizaines
de millisecondes. Coût : évaluation seulement sur checkpoints gelés.

## I02 — Observer compact à cadence lente

Remplacer l'observer S4 64×64 par une petite récurrence ou un filtre d'état opéré
à cadence décimée, puis interpoler ses paramètres FiLM. L'analogie vient des
modèles hybrides et filtres différentiables : réserver la capacité à la dynamique
lente et laisser la WaveNet modéliser la non-linéarité rapide.
([Kuznetsov et al., 2020](https://www.dafx.de/paper-archive/details/rA_6fTdLky8YDvH03jdufw),
[Yu et al., 2024](https://arxiv.org/abs/2404.07970),
[Engel et al., 2020](https://arxiv.org/abs/2001.04643))

## I03 — Encodeur d'état initial dry-only

Apprendre un encodeur causal sur un préfixe dry pour initialiser l'état du modèle
avant la fenêtre scorée. Les deep encoder networks fournissent une motivation en
identification non linéaire, mais l'efficacité sur amplificateurs doit être
démontrée. Prédiction : réduction de l'erreur juste après reset sans dégrader le
régime stationnaire. Falsification : aucune réduction appariée, ou dépendance à
des cibles wet indisponibles à l'inférence.
([Beintema et al., 2021](https://proceedings.mlr.press/v144/beintema21a.html))

# Décision pour la campagne

Avant tout nouveau run : qualifier le contrat de contexte commun sur entrées
synthétiques, vérifier gradients finis et mémoire sur les trois comparateurs,
puis geler le snapshot. Ensuite seulement, exécuter la matrice de développement
et le failure mining validation. La confirmation Rodent/Fuzzy Logic et ses tests
restent scellés jusqu'aux locks candidat, comparateur, recettes et checkpoints.

Cette recommandation ne promet pas un GO. Elle vise un verdict interprétable,
y compris si le résultat final est `NO-GO-OBJECTIVE-SOTA`.

# Artefacts associés

- Revue détaillée et bibliographie : `review.md`, `review.bib`.
- Corpus dédupliqué : `sources/corpus.json`, `sources/corpus.md`.
- Journal de recherche : `sources/search_log.md`.
- Registre des conclusions : `sources/claim_ledger.md`.
- PDF de la revue détaillée : `output/pdf/review.pdf`.
