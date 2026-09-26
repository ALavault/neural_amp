# Les modèles imitent-ils le bruit de fond dans les passages calmes ? (piste D)

**Écrit le 2026-09-26, avant toute mesure.** Diagnostic en lecture seule sur des checkpoints
existants, CPU, aucun entraînement.

**Pourquoi c'est une question.** Les cibles Big Muff ont un bruit de fond vers −74 dBFS (1er
centile des trames de 20 ms), 20 dB au-dessus de celui du signal d'entrée (DI, −94 dBFS). Un
modèle déterministe ne peut prédire que la part de ce bruit qui vient du DI amplifié. Or le terme
MR-STFT de la perte compare des magnitudes en log, sans regarder la phase. Il pourrait donc
pousser le modèle à produire, dans les passages calmes, une énergie au niveau du bruit de fond qui
n'est pas cohérente avec la cible : soit par un gain trop fort, soit par un signal parasite.

## Dispositif

- **Modèles** : les 32 checkpoints `last.ckpt` du pilote C et de son extension (graines 42 à 57,
  bras « décide » et « fixe »), SSM-WaveNet, tous entraînés avec L1 + 0,1·MR-STFT.
- **Données** : partition de test Big Muff, telle que le banc la charge (`data_module("test")`,
  rééchantillonnée à 48 kHz par nablafx, segments de 5 s). Le modèle est remis à zéro à chaque
  segment, et la première demi-seconde de chaque segment est écartée.
- **Passages calmes**, choisis sur l'**entrée** et non sur la cible, pour ne pas sélectionner des
  réalisations du bruit de la cible : trame de 20 ms dont la puissance du DI est à moins de 10 dB
  du 1er centile du DI, précédée de 200 ms de trames calmes. **Passages forts** : puissance du DI
  au-dessus de la médiane.
- **Spectres** : fenêtres de Hann de 2 048 échantillons, pas de 1 024, entièrement contenues dans
  des trames calmes (ou fortes). On moyenne S_mm, S_yy et S_my (m : sortie du modèle, y : cible).
  Bande 100 Hz à 10 kHz.

## Grandeurs

- **β = Σ|S_my| / Σ S_mm** : gain de la meilleure correction linéaire, fréquence par fréquence,
  de la sortie du modèle vers la cible. Un régresseur pur, qui prédit la moyenne de la cible
  sachant l'entrée, donne β = 1, **quel que soit le bruit de la cible**. Une sortie trop forte ou un
  signal parasite incohérent donnent β < 1. Le signe de la sortie n'intervient pas.
- **L = Σ S_mm / Σ S_yy** : niveau de la sortie rapporté à celui de la cible.
- **C = Σ (|S_my|² / S_mm) / Σ S_yy** : part de la puissance de la cible expliquée linéairement
  par la sortie du modèle, soit une borne basse de la part prévisible du bruit de fond.

## Prédiction et table d'issues

Jugée sur la **médiane des 32 modèles** ; le nombre de modèles dans chaque classe est rapporté.

| β calme | β fort | issue |
|---|---|---|
| ≥ 0,8 | — | **réfutée : régression**, pas de puissance incohérente dans les passages calmes |
| ≤ 0,5 | ≥ 0,8 | **soutenue : excès propre aux passages calmes**. Si de plus 0,5 ≤ L ≤ 2, le modèle imite le niveau du bruit de fond |
| ≤ 0,5 | < 0,8 | **excès général**, pas propre aux passages calmes |
| entre 0,5 et 0,8 | — | indécidable |

A priori : réfutée, à environ 60 %. Un modèle déterministe ne produit rien sans entrée, et un
excès incohérent coûte du L1 partout où il apparaît.

## Ce que cela ne dira pas

- **La cause.** Tous les modèles ont la même perte : une imitation observée ne pourra pas être
  attribuée au terme MR-STFT sans un modèle entraîné en L1 seul.
- Un appareil, une architecture ; données rééchantillonnées de 44,1 à 48 kHz.
- C est une borne basse : une part prévisible que le modèle n'a pas apprise compte comme
  imprévisible.

## Contrôle ajouté le 2026-09-26, après la mesure et avant son exécution

La mesure (section suivante, quand elle sera écrite) montre un excès de sortie dans les passages
calmes. Une relecture Codex en propose une autre cause : les constantes de temps les plus lentes
des modèles vont de 0,09 à 0,50 s. Les états n'ont donc peut-être pas fini de décroître, et la
garde de 200 ms ne porte que sur le DI.

**Contrôle** : même calcul, mais avec le DI mis à **zéro exact sur toutes les trames calmes**
(seuil de 10 dB, avant la garde). On compare, dans les mêmes 225 fenêtres, la puissance de la
sortie dans la bande, P₀, à celle de la mesure, P.

- P₀ / P ≤ −10 dB → l'excès vient de l'**amplification du DI calme** (résidu de quantification) ;
- P₀ / P ≥ −3 dB → il vient des **états, des biais ou du calcul**, et pas du DI calme ;
- entre les deux → cause mixte.

Jugé sur la médiane des 32 modèles.
