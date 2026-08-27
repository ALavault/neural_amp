# M2 — Reproduction de NAM A2

## Portée

M2 reproduit la chaîne officielle NAM A2 sur un système `tanh` synthétique déterministe. Cette étape valide l’architecture, l’entraînement, l’export, l’inférence causale et le coût CPU de référence. Elle ne mesure pas la fidélité à un dispositif physique et ne constitue aucune preuve pour H1 ou H2.

## Sources figées et architecture réelle

- Trainer NAM `v0.13.0`, commit `f26112906de06ec6b796ad6d1982e29eed83144e`.
- NeuralAmpModelerCore `v0.5.4`, commit `1f42f88535884450104b8711d7595019afa0495b`.
- Configuration officielle `config_model_packed.json`, SHA-256 `4000f7a428b88f2f83a876f817603e6b1017f5a473bacbc9a343a93b3ab42c3e`.

Le modèle packed entraîne simultanément `channels_3` (Lite; appelé « nano » dans Core) et `channels_8` (Full/standard). Chaque sous-modèle utilise 23 couches causales LeakyReLU non gated, les dilatations officielles, une tête de noyau 16 et un champ réceptif de 6 347 échantillons (132,23 ms). Lite contient 1 870 paramètres et Full 12 145. L’inspection dérive respectivement 1 732 et 11 777 MAC linéaires minimales par échantillon, hors activations et additions.

## Données et entraînement minimal

Les fichiers train (384 000 échantillons) et validation (192 000) sont générés séparément à 48 kHz, sans normalisation ni fenêtre partagée. Les cibles appliquent le même `tanh` connu. La recette officielle est conservée: ESR de validation, MRSTFT pondérée à `0.0005`, Adam à `0.004`, weight decay `3.17e-7` et scheduler exponentiel `0.994`. Chaque seed effectue huit epochs, 40 mises à jour.

Le backward CUDA de la réflexion utilisée par MRSTFT n’a pas d’implémentation déterministe. Après deux runs échoués et conservés, le mode Lightning `deterministic="warn"` est utilisé avec toutes les seeds et options cuDNN explicites.

| Seed | Lite ESR | Full ESR | Full ESR initial → final (logger) |
| ---: | ---: | ---: | ---: |
| 0 | 0,915501 | 0,689423 | 0,953033 → 0,683702 |
| 1 | 0,876815 | 0,528333 | 0,949592 → 0,528568 |
| Médiane | 0,896158 | 0,608878 | — |

Les deux courbes descendent sans valeur non finie. Les métriques relues depuis les exports diffèrent légèrement de celles du logger parce que chaque sous-modèle est exporté depuis son meilleur checkpoint individuel.

## Parité numérique et causalité

Le run `m2_a2_cpu_seed0_v2` traite le fichier complet par blocs 1, 16, 64, 128 et selon le motif irrégulier `[1, 7, 64, 3, 128, 17]`.

- erreur maximale Python/C++: `4,77e-7` Lite, `5,96e-7` Full;
- erreur maximale entre tailles C++: `3,58e-7` Lite, `2,98e-7` Full;
- sortie après reset: bit-identique;
- outil officiel `render` et harnais bloc 64: bit-identiques;
- longueur de sortie égale à l’entrée; aucune latence supplémentaire ajoutée.

Le seuil C++ a été corrigé avant gel de `1e-7` à `5e-7` après qu’un run échoué a montré un écart maximal d’environ trois ulps float32. Le fast-path calcule en float32; l’interface hôte Core est en double.

## Coût CPU réel

Mesure Release `-Ofast` avec optimisation interprocédurale, un thread, cœur logique 0 fixé, turbo actif et gouverneur `schedutil`. Les résultats sont spécifiques au double Xeon E5-2630 v3 de ce poste.

| Modèle | Bloc | ns/échantillon médian | ns/échantillon p95 | Facteur temps réel médian |
| --- | ---: | ---: | ---: | ---: |
| Lite | 1 | 1 432 | 1 619 | 14,55× |
| Lite | 16 | 519 | 635 | 40,11× |
| Lite | 64 | 425 | 540 | 49,07× |
| Lite | 128 | 411 | 534 | 50,73× |
| Full | 1 | 3 046 | 3 390 | 6,84× |
| Full | 16 | 3 506 | 4 562 | 5,94× |
| Full | 64 | 2 862 | 3 521 | 7,28× |
| Full | 128 | 2 705 | 3 256 | 7,70× |

À bloc 64, l’outil officiel indépendant mesure 27,44 µs/bloc Lite et 183,06 µs/bloc Full, contre 27,17 et 183,18 µs dans le harnais, ce qui confirme la stabilité locale. Le fast-path officiel accélère le chemin générique de 5,69× pour Lite et 1,09× pour Full sur ce processeur. Les buffers float du fast-path à bloc 64 représentent 179 456 octets pour Lite et 477 696 octets pour Full, hors poids, objets et allocateur.

Les sources figées ne donnent pas de cible A2 publiée normalisée pour ce processeur; aucune concordance inter-machine n’est revendiquée.

## Décision de gate

M2 passe: sources non modifiées, deux seeds convergentes, exports officiels relus, traitement causal par blocs validé, tests Core réussis et benchmark CPU stable. Les échecs `F-M2-001` à `F-M2-011` restent au registre. M3 peut commencer; `EXTERNAL_REPORT_ONLY` reste verrouillé.
