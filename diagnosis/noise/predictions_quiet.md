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

La mesure (section « Résultats » ci-dessous) montre un excès de sortie dans les passages
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

## Résultats (2026-09-26, `quiet_frames.py`, `control_zero_di.py`)

Données : 12 segments de test, plancher du DI −99,9 dBFS, 225 fenêtres calmes et 1 038 fortes.

**Issue de la table : soutenue, excès propre aux passages calmes, sur les 32 modèles.**

| | passages calmes | passages forts |
|---|---|---|
| β (médiane ; étendue) | 0,025 (0,019 à 0,034) | 0,985 (0,950 à 0,998) |
| L | 5,3, soit +7,2 dB (2,7 à 8,9) | 0,99 |
| C | 0,005 | 0,96 |
| niveau de la cible | −74,2 dBFS | −24,2 dBFS |
| niveau de la sortie du modèle | −67,0 dBFS (−69,9 à −64,7) | — |

- **Pas d'imitation du niveau** au sens écrit : aucun modèle n'a 0,5 ≤ L ≤ 2. Les modèles sortent
  7 dB **au-dessus** du bruit de fond de la pédale.
- **L'excès est large bande** : +7 à +8 dB par octave de 125 Hz à 1 kHz, +6 dB à 4 kHz,
  +2,7 dB à 8 kHz.
- **Mon a priori (réfutée, à 60 %) était faux.**

**Contrôle DI à zéro : l'excès vient de l'amplification du DI calme.** Avec le DI mis à zéro sur
les trames calmes (13,1 % des échantillons), la puissance de sortie dans les mêmes fenêtres baisse
de **64,4 dB** en médiane (de −53,8 à −69,6 dB). Les états, les biais et le calcul pèsent donc
plus de 50 dB sous l'excès.

**Ce que contient ce DI calme.** Le DI de test est en PCM 16 bits. Ses trames brutes les plus
calmes (1er centile, 44,1 kHz) ne contiennent que −1, 0 et +1 LSB, dont 94 % de zéros. La cible,
aussi en 16 bits, vaut 6,4 LSB RMS dans les 225 fenêtres analysées. Les modèles amplifient donc
un résidu de quantification d'environ −100 dBFS de quelque 33 dB. La sortie obtenue n'a aucune
cohérence détectable avec la cible, et elle est plus forte que le vrai bruit de fond de la pédale.

**Relecture Codex (lecture seule) : corrections retenues.**

- C = 0,0046 et β = 0,025 sont au **plancher de l'estimateur** en l'absence de toute relation :
  environ 1/225 = 0,0044 pour C, et environ 0,026 pour β à L = 5,3. La bonne lecture est
  « aucune cohérence détectable ». C n'est pas une borne basse pour un estimateur fini,
  contrairement à ce qu'écrivait la section « Ce que cela ne dira pas ».
- β n'est pas un gain unique. Pour L = 5,3, β ≤ 0,435 même en cas de relation parfaitement
  cohérente (Cauchy–Schwarz : β²L ≤ C). Le β faible mêle donc gain excessif et incohérence ; c'est
  C qui établit l'incohérence.
- Les 225 fenêtres se recouvrent à 50 %, et 83 % viennent de deux segments (0 et 11). « 32 sur
  32 » établit la stabilité entre checkpoints sur un seul enregistrement, pas 32 réplications.
- Deux phrases du pré-enregistrement étaient fausses. β ≥ 0,8 ne prouverait pas l'absence de
  puissance incohérente. Et un modèle déterministe peut produire une sortie sans entrée, à cause
  de ses biais ; le contrôle montre qu'ici ils ne pèsent rien.

**Ce que le résultat ne dit pas.**

- **Si la pédale a reçu ce résidu.** Si le DI a été requantifié en 16 bits après la capture, la
  pédale ne l'a jamais vu, et la bonne sortie dans ces passages est inconnaissable avec ces
  données.
- **Si le gain des modèles à très bas niveau est juste pour un vrai signal analogique.** La seule
  entrée faible des données est ce résidu.
- **La cause côté entraînement.** On ne peut pas dire si c'est le terme MR-STFT ou une simple
  extrapolation que la perte ne contraint pas (ces passages ne pèsent presque rien en énergie) :
  tous les modèles ont la même perte.

**Conséquences.**

- **Produit.** Nourri d'un DI 16 bits calme, un modèle Big Muff souffle 7 dB au-dessus de la
  vraie pédale. Avec une vraie guitare, c'est son gain à bas niveau qui décidera du bruit
  amplifié, et ce gain n'est pas validé.
- **Piste bruit.** Le « bruit » que produisent ces modèles n'est pas celui de l'appareil : c'est
  leur entrée faible, amplifiée. Enlever le bruit sans rien abîmer passe donc aussi par le gain
  du modèle à bas niveau, pas seulement par les sources de bruit de l'appareil.
- **Séance VS100R.** Les DI de la source viennent de fichiers en flottant 32 bits, et l'ampli
  reçoit exactement le contenu du fichier. Pour ce jeu, la paire entrée faible / cible sera
  donc cohérente, et la même mesure y deviendra lisible.
