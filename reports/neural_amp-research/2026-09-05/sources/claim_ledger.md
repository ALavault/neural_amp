# Registre claim → source

| ID | Conclusion opérationnelle | Sources | Nature de preuve | Confiance / réserve |
|---|---|---|---|---|
| C01 | Aucun modèle n'est démontré vainqueur universel sur toutes les classes d'effets. | `comunita2025`, `simionato2024a` | Comparaisons publiées, protocoles hétérogènes | Élevée pour l'absence de généralisation; ne compare pas directement v9. |
| C02 | Les RNN compactes sont des baselines crédibles pour l'émulation d'amplis. | `wright2019`, `wright2020`, `damskagg2018` | Résultats directs sur amplis, dont une référence SPICE | Élevée; matériel et données ne sont pas interchangeables. |
| C03 | Un état peut survivre au-delà de l'horizon de gradient TBPTT. | `tallec2017`, `aicher2020` | Théorie/expériences hors audio | Élevée pour le mécanisme, moyenne pour le transfert audio. |
| C04 | Les SSM offrent des mécanismes de mémoire longue, pas une garantie d'horizon utile. | `gu2021`, `gu2022`, `orvieto2023`, `gu2023` | Architecture générale | Élevée sur le mécanisme; preuve audio indirecte. |
| C05 | Les dynamiques attaque/release peuvent révéler des limites masquées par la moyenne. | `simionato2024b`, `steinmetz2021` | Audio dynamique direct | Moyenne à élevée; surtout compresseurs. |
| C06 | Une lecture multi-métrique est justifiée, sans changer post hoc la primaire. | `wrightloss2020`, `cassidy2023` | Pertes et évaluation perceptuelle audio | Élevée pour compléter ESR; aucune équivalence perceptuelle parfaite. |
| C07 | L'aliasing est un mécanisme d'échec plausible des modèles non linéaires. | `sato2025`, `mikkonen2025` | Méthodes antialiasing audio récentes | Moyenne; ne prouve pas que v9 est dominée par l'aliasing. |
| C08 | Un encodeur d'état initial est une hypothèse testable pour réduire les erreurs après reset. | `beintema2021` | Identification state-space transférée | Moyenne; non démontré sur amplis dans ce corpus. |
| C09 | V8 donnait au teacher un historique continu et resetait les comparateurs par segments. | audit local `run_quality_teacher_v6_development.py`, `teacher_comparison.py` | Inspection directe du code gelé | Élevée; corrigé prospectivement en v9. |
| C10 | La validation comparateur v8 surpondérait les tails courts. | audit local `quality_teacher.py`, `quality_teacher_comparator.py` | Inspection directe des agrégations | Élevée; corrigé prospectivement en v9. |
| C11 | NablAFx S4-TFiLM déclarait 1 016 échantillons de latence et l'ancienne perte omettait la fin réelle de chaque segment. | audit local `sota_comparators.py`, `quality_teacher_comparator.py` | Inspection directe du code et des indices | Élevée; v9 flushe la latence vers les coordonnées source. |
| C12 | La revendication doit rester limitée aux recettes ouvertes réentraînées localement sous le contexte v9. | `comunita2025`, audit des recettes locales | Écart constaté entre publication et implémentation locale | Élevée; interdit de dire « reproduction exacte publiée ». |

Les identifiants bibliographiques sont résolus dans `corpus.json` et
`review.bib`. Les conclusions C09–C12 sont des résultats d'audit du dépôt, pas
des affirmations attribuées à la littérature.
