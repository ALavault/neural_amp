# DIAGNOSIS_ffuzz : plancher d'ESR commun à deux architectures sur ToneTwist F-Fuzz

Voie produit (D-PRODUCT-001), aucune revendication scientifique. Scripts et tables :
`diagnosis/ffuzz/`. Lecture seule sur les runs `param_ssm_wavenet_ffuzz_r2` et
`param_s4_ffuzz_r2` ; les inférences sur leurs checkpoints se font sur CPU. Sections 1 à 3
écrites le 2026-09-24, **avant toute mesure**.

## 1. Écart

- **Métrique** : ESR de test de nablafx, moyenne par lots de 8 segments de 5 s, dernier
  checkpoint. 133 segments par réglage, 11 réglages ; F030 et F070 sont retirés de
  l'entraînement.
- **Valeur** : 0,1366 pour SSM-WaveNet-FiLM (28 385 paramètres) et 0,1377 pour S4-param
  (29 713). Réglage par réglage, les deux architectures sont à moins de 3 % l'une de l'autre,
  de 0,113 (F000) à 0,183–0,188 (F100), et l'ESR croît de façon monotone avec le fuzz.
- **Référence** :
  - le même SSM-WaveNet atteint 0,04 sur Big Muff (`demo/nablafx_bench/paired_ssm_wavenet_*`) ;
  - à ESR égal, les deux architectures diffèrent d'un facteur 1,7 en MR-STFT (0,157 contre
    0,272 sur les réglages vus). L'erreur de forme d'onde est donc commune, pas l'erreur
    spectrale.
  - Aucune valeur publiée n'a été vérifiée pour ce réglage de données.
- **Dispersion** : une run par cellule (graine 42, partition 925). Sur Big Muff, l'écart-type du
  log de l'ESR entre graines vaut 0,27 à 0,53 ; l'écart à Big Muff (facteur 3,4, soit 1,23 en
  log) le dépasse.
- **L'objet du diagnostic est la coïncidence** : un plancher commun à deux modèles dont
  l'erreur spectrale diffère d'un facteur 1,7.

## 2. Comparabilité

- **Jeu de test** : les deux runs partagent le même jeu (`DRY-with-markers/test` ×
  `MultidrivePedalPro-F-Fuzz/test`), soit deux fichiers : `idmt-bass` (358 s) et
  `idmt-gtr4-ib` (314 s).
- **Taux d'échantillonnage** : les fichiers sont natifs à 48 kHz, en flottants 32 bits.
  Aucun rééchantillonnage, contrairement à Big Muff.
- **Métrique et protocole** : identiques entre les deux runs (même script, même commit de
  nablafx, même `metrics_by_setting`).
- **Budget** : 15 000 pas pour chacun, sans arrêt anticipé ; paramètres à 5 % près.
- **Données d'entrée** :
  - deux impulsions de calage au début et à la fin de chaque fichier ;
  - pour tous les fichiers sauf `nam` et `idmt-gtr2`, un gain aléatoire entre −20 et 0 dB
    appliqué toutes les 5 s (README de `DRY-with-markers`).
- **Alignement** : décalage d'enveloppe entrée/cible de −29 à +165 échantillons selon le fichier
  (`demo/param/DEVIATION.md`). C'est une mesure d'enveloppe, pas un retard de forme d'onde.
- **Comparabilité avec Big Muff** : autre appareil, autre jeu de test, autres pertes (0,5/0,5
  contre 1,0/0,1). La référence de 0,04 est un ordre de grandeur, pas une comparaison.

Aucun défaut de comparabilité entre les deux runs. La comparaison avec Big Muff est faible, et la
question porte sur le plancher commun.

## 3. Hypothèses

Écrites avant toute mesure. Chaque hypothèse donne sa mesure discriminante et sa prédiction si
elle est vraie. Les hypothèses de Codex, consulté en lecture seule sans elles, sont ajoutées
en 3b avec leur provenance.

**H1, alignement (données).** Un retard entrée/cible que les modèles causaux ne peuvent pas
reproduire, ou reproduisent mal. Un décalage de d échantillons sur un signal riche en
harmoniques coûte environ 2(1 − ρ(d)) en ESR.
- *Mesure* : par fichier de test, le retard qui maximise l'intercorrélation entre la sortie du
  modèle et la cible ; ESR après ce réalignement.
- *Prédiction si vraie* : un retard non nul (|d| ≥ 1), le même pour les deux modèles ; ESR
  réaligné ≤ 0,5 × ESR actuel.

**H2, bruit ou composante non déterministe de la cible (données).** Un bruit de fond ajouté au
rendu, amplifié par le fuzz, qu'aucun modèle ne peut prédire.
- *Mesure* : spectre et niveau de l'erreur dans les passages où l'entrée est quasi nulle ; part
  de l'énergie d'erreur non corrélée à l'entrée (cohérence entrée/erreur).
- *Prédiction si vraie* : un plancher d'erreur absolu indépendant du niveau d'entrée, qui croît
  avec le réglage de fuzz. L'ESR par segment est alors inversement proportionnelle à l'énergie
  de la cible sur les segments calmes (pente ≈ −1 en log-log).

**H3, agrégation de la métrique (mesure).** L'ESR moyenne est tirée par quelques segments de faible
énergie, par exemple les blocs à −20 dB ou les segments de marqueurs, comme le réglage G000 du
Marshall.
- *Mesure* : distribution des ESR par segment, médiane, ESR pondérée par l'énergie, part de la
  moyenne portée par les 10 % de segments les pires.
- *Prédiction si vraie* : médiane ≤ 0,5 × moyenne, ou plus de 50 % de la moyenne portée par 10 %
  des segments.

**H4, contenu haute fréquence hors d'atteinte (capacité ou repliement).** Le rendu numérique
produit des harmoniques de fuzz, éventuellement suréchantillonnés, que deux modèles à 48 kHz sans
suréchantillonnage replient ou atténuent de la même façon.
- *Mesure* : répartition spectrale de l'énergie d'erreur ; ESR après filtrage passe-bas à 5 kHz
  de la sortie et de la cible.
- *Prédiction si vraie* : plus de 60 % de l'énergie d'erreur au-dessus de 5 kHz, et une ESR
  passe-bas ≤ 0,5 × ESR actuelle.

**H5, marqueurs de calage (données).** Les impulsions de début et de fin tombent dans des segments
de test, et leur réponse de fuzz est mal modélisée.
- *Mesure* : ESR des segments qui contiennent un marqueur, contre les autres.
- *Prédiction si vraie* : ces segments portent plus de 30 % de la moyenne.

**H6, optimisation commune (optimisation).** Les deux modèles s'arrêtent sur le même plateau de
perte imposé par le protocole (15 000 pas, pertes 0,5/0,5).
- *Mesure* : ESR de train contre ESR de test, et pente de la perte de validation sur les 2 000
  derniers pas.
- *Prédiction si vraie* : perte de validation encore décroissante (> 5 % sur les 2 000 derniers
  pas) et ESR de validation proche de l'ESR de test. H6 n'explique pas à elle seule
  l'égalité réglage par réglage ; elle est la moins probable.

### 3b. Hypothèses de Codex, et ce qui était connu au moment du classement

Codex a été consulté en lecture seule, avec les faits et les chemins mais sans les hypothèses
ci-dessus (`task`, 2026-09-24). Il n'a lancé aucune inférence. Il a proposé huit causes ; celles
qui recoupent les miennes sont renvoyées.

- **C1, décalage temporel résiduel** (= H1). Précision ajoutée : un retard d'un échantillon à
  48 kHz donne à lui seul une ESR de 0,11 à 0,19 vers 2,5–3,4 kHz ; et la lag doit être estimée
  sur une autre fenêtre que celle qui est notée.
- **C2, intervalles localement incohérents** (marqueurs, raccords, trous ; = H5, élargie).
  Prédiction : plus de la moitié de l'erreur quadratique dans moins de 5 % des échantillons. Indice
  relevé par Codex : MSE/MAE² ≈ 12, signe d'erreurs concentrées.
- **C3, compromis L1/MR-STFT** (optimisation, nouveau). L'ESR n'est pas la perte entraînée.
  Codex a relevé que l'ESR de validation passe par son minimum tôt (pas 2 015 pour SSM, 2 751
  pour S4, ≈ 0,204) puis remonte légèrement jusqu'au pas 15 000.
- **C4, bruit ou état non observé de la cible** (= H2). Pour tout expliquer, il faudrait une
  composante imprévisible de 11 à 19 % de l'énergie de la cible.
- **C5, pondération par segment** (= H3). Contre-indice mesuré par Codex : aux réglages F000 et
  F100, les 133 segments de test ont tous un RMS supérieur à 0,08. Aucun dénominateur n'est
  quasi nul.
- **C6, couverture d'entraînement réduite** (données, nouveau). `data_to_use = 0.1` ne garde que
  10 % des segments trainval, puis la partition 90/10.
- **C7, limite de représentation commune** (capacité, = H4 élargie).
- **C8, contexte manquant en début de segment** (évaluation, nouveau). Les segments sont notés
  depuis un état nul, alors que la cible a été rendue avec son passé.

**Ce que je savais avant de classer, et qui rend ce classement non aveugle.** Le test de fumée
de `diagnosis/ffuzz/measure.py` (deux segments par réglage, SSM seul, lancé pour vérifier le
script) a montré un retard de +4 échantillons de la cible sur la sortie. Le réalignement y faisait
tomber l'ESR intérieure de 0,040 à 0,0013 et de 0,025 à 0,0017. Ce résultat a été vu avant le
classement ci-dessous, et après l'écriture de H1 et la clôture de la liste de Codex.

**Classement a priori** : H1/C1 d'abord (appuyé par la fumée ci-dessus) ; puis C8 et C2, qui
prédisent une erreur localisée ; puis C3 et C6, communs aux deux architectures par construction
du protocole ; H2/C4, H3/C5 et H4/C7 ensuite ; H6 en dernier.

## 4. Mesures et verdict

Scripts : `diagnosis/ffuzz/measure.py` (test, 1 463 segments par modèle, `segments.json`),
`train_lag.py` (entraînement, `train_lag.json`) et `data_lag.py` (audio seul, `data_lag.json`).

- **H1 / C1, alignement : soutenue, et elle explique presque tout le plancher.**
  - Test : le retard optimal de la sortie à la cible vaut **+4 échantillons dans 1 463 segments
    sur 1 463** pour SSM, et dans 1 459 sur 1 463 pour S4 (les quatre autres : +3). Il est le même
    dans les deux fichiers et à tous les réglages ; la cible arrive 4 échantillons après la
    sortie du modèle.
  - Réaligné, l'ESR intérieure tombe de 0,136 à **0,0013** pour SSM (÷100) et de 0,137 à
    **0,0032** pour S4 (÷43). Par réglage, de F000 à F100 : 0,0006 à 0,0042 pour SSM, 0,0015 à
    0,011 pour S4.
  - Entraînement : **+4 dans 100 segments sur 100** pour chaque modèle (cinq fichiers, F000 et
    F100) ; réaligné, l'ESR passe de 0,05–0,32 à 0,001–0,008. Ce n'est donc pas une
    incohérence entre partitions : les deux modèles, entraînés sur ces cibles, ne produisent
    pas ce retard.
  - Le retard mesuré sur l'audio seul, entrée contre cible, varie de −11 à +4 selon le fichier
    (+3 pour quatre fichiers d'entraînement, +4 pour `nam`, −10 pour `yt-bass`, −2 et +2 en test).
    L'intercorrélation brute mêle la phase propre de l'effet et le contenu du fichier ; ce n'est
    pas une mesure propre de latence. Le retard modèle → cible, lui, est constant.
  - La prédiction écrite (|d| ≥ 1, le même pour les deux modèles, ESR réalignée ≤ 0,5 × ESR)
    est tenue, et très au-delà.
- **H2 / C4, bruit de la cible : réfutée.** Une fois réaligné, le résidu vaut 0,1 à 0,3 % de
  l'énergie de la cible. Aucune composante imprévisible de 11 à 19 % n'existe.
- **H3 / C5, agrégation : réfutée.** Médiane 0,135 contre moyenne 0,137 (SSM), les 10 % pires
  segments portent 20 % du total (seuil 50 %), ESR pondérée par l'énergie 0,140.
- **H4 / C7, aigu hors d'atteinte : réfutée comme cause principale.** 17 % de l'énergie d'erreur
  au-dessus de 5 kHz (seuil 60 %). L'ESR après un passe-bas à 5 kHz vaut 0,111, soit 0,81 fois
  l'ESR (seuil 0,5).
- **H5 / C2, marqueurs : réfutée.** Les 44 segments de début et de fin de fichier portent 3,5 %
  du total (seuil 30 %).
- **C8, contexte manquant en début de segment : réfutée.** L'ESR intérieure, 1 000 échantillons
  retirés de chaque côté, égale l'ESR complète (0,1364 contre 0,1367).
- **H6 / C3, optimisation, et C6, couverture à 10 % : non mesurées, sans objet.** Le retard
  explique le plancher, et les modèles atteignent 0,001–0,008 sur le matériau d'entraînement
  réaligné.

## 5. Contribution retenue

**Prédicat.** Tous les segments, les sept fichiers (entraînement et test), les onze réglages et
les deux architectures : l'écart est uniforme, ce qui explique la coïncidence entre deux modèles
dont la MR-STFT diffère d'un facteur 1,7. La MR-STFT, aveugle à la phase, ne voit pas un retard
pur ; l'ESR le voit entièrement.

**Mécanisme.** Les cibles F-Fuzz rendues portent une latence de 4 échantillons par rapport aux
entrées `DRY-with-markers`, probablement la latence de traitement du plugin, non vérifiée dans sa
documentation. Un modèle causal pourrait la reproduire en retardant sa sortie. Mais ni une somme
d'exponentielles amorties à 4 ou 32 modes par canal suivie de couches sans mémoire (SSM-WaveNet),
ni S4 ne représentent un retard pur ; entraînés à L1 + MR-STFT, les deux convergent vers la
réponse non retardée. Le coût en ESR d'un retard fixe croît avec le contenu harmonique, donc avec
le fuzz : de 0,113 à 0,183.

**Prédiction chiffrée qu'un correctif devra satisfaire.** Compenser la latence dans les données,
en avançant les cibles de 4 échantillons à l'entraînement comme au test, avec le même protocole
et les mêmes graines :

- ESR de test moyenne ≤ 0,01 pour les deux architectures, contre 0,137 aujourd'hui ;
- l'écart entre les deux architectures reparaît : S4 ≥ 1,5 × SSM, comme sur le réalignement a
  posteriori (0,0032 contre 0,0013) ;
- le rapport F100/F000 de l'ESR de SSM reste ≤ 10, contre 6 après réalignement a posteriori.

Le réalignement a posteriori ne remplace pas ce correctif. Il choisit le retard après avoir vu la
sortie, et les modèles ont été entraînés contre des cibles décalées.

## 6. Non expliqué, non vérifié

- **L'origine des 4 échantillons** : latence du plugin Multidrive Pedal Pro, suréchantillonnage
  interne ou chaîne de rendu de ToneTwist ; non vérifiée.
- **Le retard brut entrée/cible varie selon le fichier** (−11 à +4), alors que le retard modèle →
  cible est constant. C'est cohérent avec une phase propre de l'effet qui dépend du contenu, mais
  ce n'est pas démontré ; `yt-bass` (−10) est le plus éloigné.
- **Les autres appareils** : Big Muff et Marshall n'ont pas été soumis à la même mesure. Si leurs
  cibles portaient aussi une latence, les ESR du banc en contiendraient une part. Mesure à faire,
  peu coûteuse (CPU).
- **Les résultats publiés** de NablAFx sur F-Fuzz portent vraisemblablement le même plancher, si
  leur chargeur ne compense rien ; non vérifié.
- Le doublement de la MR-STFT sur les réglages non vus (`demo/param/RESULTS.md`) est une autre
  question : il ne dépend pas de ce retard, auquel la MR-STFT est aveugle.

Correctif : voir skill controlled-fix.
