# Où en sommes-nous face à l'état de l'art

> **Erratum (15 septembre 2026).** Ce banc ne suit pas le protocole publié. (1) Les fichiers sont ceux de ToneTwist, mais tronqués à 120/30/30 s par `configs/data/m4_internal.yaml` (publiés : 340/51/60 s) : les modèles ont vu 35 % des données d'entraînement. (2) Le test publié est connu : le fichier `test` complet, évalué dans nablafx par tranches de 5 s (ESR moyen par tranche, sur `last.ckpt`). (3) Notre boucle diffère de celle de nablafx (validation et scheduler toutes les 50 époques, pas d'early stopping, validation sur un fichier séparé). Ces chiffres ne sont donc pas comparables à 0,1076. Le banc conforme est `scripts/product_nablafx_bench.py` (résultats dans `demo/nablafx_bench/`).

Appareil : `electro_harmonix_big_muff`. Tous les modèles sont entraînés et évalués sur les
mêmes six fichiers, avec la même implémentation de métrique
(`fssr_nam.metrics.time.time_metrics`). Aucun chiffre de la littérature
n'est utilisé comme baseline : les modèles de comparaison sont entraînés ici.

## Verdict

Nous sommes DERRIÈRE : NAM A2 Full est à 0.18825 d'ESR de test contre 0.18655 pour S4-TFiLM large, soit 1.01 fois pire.

Mais l'écart de 1 % n'est pas l'information importante. Le banc en livre
trois autres, plus dérangeantes.

**1. Deux architectures sans rien de commun butent au même endroit.**
S4-TFiLM (SSM diagonal + modulation temporelle, 70 193 paramètres) et NAM A2
Full (WaveNet dilaté, 12 145 paramètres) arrivent à 0,18655 et 0,18825 d'ESR
de test, soit 1 % d'écart. Quand deux familles de modèles aussi différentes
convergent vers la même valeur, le plafond est plus probablement dans les
données que dans l'architecture. Changer de modèle ne fera pas tomber ce mur.

**2. L'écart validation/test est énorme et n'est pas le même pour les deux.**
S4-TFiLM passe de 0,01905 en validation à 0,18655 en test, un facteur 9,8.
NAM A2 Full passe de 0,04231 à 0,18825, un facteur 4,4. Le modèle le plus
capable est celui qui généralise le plus mal : il apprend mieux la validation
sans rien gagner sur le test.

**3. Le chiffre publié tombe entre nos deux splits.** La littérature rapporte
0,1076 sur ce Big Muff. Notre S4-TFiLM fait 0,01905 en validation, soit 5,6
fois mieux, et 0,18655 en test, soit 1,7 fois moins bien. Selon le split que
l'on cite, la même expérience « bat l'état de l'art » ou « en est loin ».
C'est exactement pourquoi aucun chiffre publié n'est utilisé ici comme
baseline.

## Comparabilité du protocole

Notre découpage n'est pas celui de la littérature et les chiffres publiés
ne sont donc pas directement opposables aux nôtres : le papier de référence
sur ce Big Muff rapporte 0,1076 d'ESR pour S4-TFiLM `large` sans que nous
sachions quel extrait sert de test. Nous avons mesuré que notre fichier de
test est plus dur que notre fichier de validation (RMS 0,110 contre 0,0708,
facteur de crête 5,00 contre 7,78, flux spectral supérieur de 43 %). C'est
précisément pourquoi la baseline est réentraînée ici : la colonne qui compte
est la comparaison interne, pas l'écart à un nombre publié.

## Résultats

| Modèle | Famille | Paramètres | Époques | Pas | Minutes | ESR val. | ESR test | ns/éch. PyTorch CPU |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| S4-TFiLM large | nablafx (reproduit chez nous) | 70 193 | 5000 | 15000 | 154.4 | 0.01905 | 0.18655 | 379946 |
| NAM A2 Full | NAM (architecture de référence  réutilisée) | 12 145 | 400 | — | 44.0 | 0.04231 | 0.18825 | 257443 |
| NAM A2 Lite | NAM (architecture de référence  réutilisée) | 1 870 | 400 | — | 44.0 | 0.06541 | 0.29406 | 218595 |
| GCN-TFiLM | nablafx (reproduit chez nous) | 267 520 | 5000 | 15000 | 89.5 | 0.38586 | 0.50655 | 204798 |

La colonne de coût est mesurée en PyTorch sur un seul cœur CPU, au bloc 64,
pour tous les modèles : c'est la seule comparaison à armes égales. NAM
dispose en plus d'un moteur C++ optimisé, mesuré à 4 847 ns/échantillon
dans `REPORT.md` ; aucune des baselines n'a d'équivalent.

Budget : les baselines ont reçu 15000 pas d'optimisation, contre
400 époques pour le meilleur run A2 (`product_a2_electro_harmonix_big_muff_seed0_v2`, 44.0 min).

## Compatibilité `.nam`

**S4-TFiLM large** — Non. Le format `.nam` ne décrit que les architectures du projet NAM (WaveNet, LSTM, ConvNet, Linear) ; le registre de parseurs de `NeuralAmpModelerCore` n'a pas d'entrée pour un SSM diagonal ni pour la modulation TFiLM. Preuve : `NAM/model_config.h` lève « No config parser registered for architecture » pour tout nom non enregistré, et les seuls `ConfigParserHelper` du dépôt sont dans `lstm.cpp`, `convnet.cpp`, `linear.cpp`, `container.cpp` et `wavenet/model.cpp`.

**GCN-TFiLM** — Partiellement. Le tronc convolutif dilaté est proche du WaveNet de NAM, mais la modulation TFiLM porte un état LSTM par bloc que le format `.nam` ne sait pas décrire. Un export exigerait soit de retirer TFiLM, soit d'ajouter un parseur côté moteur.

## Fichiers du banc

| Split | Rôle | sha256 | Chemin |
| --- | --- | --- | --- |
| train | input | `dce322582d50d11f` | `datasets/raw/internal_m4/electro_harmonix_big_muff/train_input.wav` |
| train | target | `985ac8a0116332dc` | `datasets/raw/internal_m4/electro_harmonix_big_muff/train_target.wav` |
| validation | input | `d1bad331ea3f9d6d` | `datasets/raw/internal_m4/electro_harmonix_big_muff/validation_input.wav` |
| validation | target | `0b389d65f7b6cd5e` | `datasets/raw/internal_m4/electro_harmonix_big_muff/validation_target.wav` |
| test | input | `ffbdcd1f23d228d9` | `datasets/raw/internal_m4/electro_harmonix_big_muff/test_input.wav` |
| test | target | `9a0a44433ab09e08` | `datasets/raw/internal_m4/electro_harmonix_big_muff/test_target.wav` |
