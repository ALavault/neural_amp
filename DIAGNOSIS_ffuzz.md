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
