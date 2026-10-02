# Brillance à 96 et 192 kHz : repliement retiré, ou compensation perdue ?

Voie produit (D-PRODUCT-001), aucune revendication scientifique. **Écrit le 2026-10-02, avant toute
mesure.** Les règles ci-dessous ne changent plus après les chiffres. Script :
`demo/sr_transfer/brightness.py` (pas encore écrit). Lecture seule, CPU.

## Origine

À l'écoute de la page `demo/listening/results.html`, sur deux segments Big Muff, l'utilisateur
trouve la sortie à 192 kHz « moins brillante, moins dure, peut-être plus proche du vrai ».
L'impression vient de deux extraits écoutés avec les noms visibles. Elle sert d'hypothèse, pas de
donnée : la mesure porte sur tout le jeu de test, et ces deux segments ne sont rapportés qu'à
titre descriptif.

Question ouverte du dépôt (`demo/sr_transfer/RESULTS.md`, « Mécanisme de la montée » ;
`reports/neural_amp-research/2026-09-23/brainstorm_supersampling.md`, pistes A et B) :
- **A, repliement retiré** : à 48 kHz, les non-linéarités du modèle produisent des harmoniques
  au-delà de 24 kHz, qui se replient dans la bande. À 96 ou 192 kHz, une partie de ces
  harmoniques ne se replie plus, et le rééchantillonneur de retour les élimine. La sortie perd
  du contenu inharmonique aigu.
- **B, compensation perdue** : le modèle a appris contre la cible avec son propre repliement. En
  retirant ce repliement, on s'éloigne de la cible.

Les mesures antérieures ne départagent pas. P8 (`RESULTS.md`) mesure une ESR contre l'appareil
avant la découverte du retard du ZOH, qui domine l'ESR. Le plancher à −37 dB
(`predictions_floor.md`) dépend du niveau, mais la dépendance au niveau n'a pas permis
d'attribuer sa cause sur ce fuzz.

**Décision que la mesure change** : faire tourner le SSM à 2× en interne (FOH et Δ rééchelonné)
comme anti-repliement sans réentraînement, ou non.

## Protocole

- **Checkpoints** : ZOH `paired_ssm_wavenet_s{925,736,688}` et FOH
  `paired_ssm_wavenet_foh_s{925,736,688}`, `last`. Les comparaisons se font à poids fixés (même
  checkpoint à 48, 96 et 192 kHz), jamais entre ZOH et FOH, qui sont des entraînements
  différents.
- **Signal de programme** : tous les segments de test Big Muff, chemin `whole` / `interior` de
  `scripts/product_sr_transfer.py`. Aux trois fréquences, rééchantillonnage par le sinc de
  Kaiser de largeur 64 de `demo/sr_transfer/floor.py` (`hq`), à l'aller comme au retour.
- **Témoin du rééchantillonneur** : la sortie du modèle à 48 kHz, envoyée seule à fs puis
  ramenée à 48 kHz par `hq`. Son effet par bande est soustrait des écarts du modèle.
- **Niveaux par bande** : densité spectrale de Welch (8192 points, Hann) sur l'intérieur des
  segments mis bout à bout, intégrée par tiers d'octave, bandes centrées de 63 Hz à 16 kHz,
  en dB.

## Mesures et lectures

**R1, brillance.** Pour chaque checkpoint et fs ∈ {96, 192} kHz, B_fs = moyenne sur les bandes
4 à 16 kHz de (L_fs − L_48) − (L_témoin − L_48), en dB.
- « Moins brillant à fs » si B_fs ≤ −0,5 dB pour au moins 5 checkpoints sur 6.
- « Plus brillant » si B_fs ≥ +0,5 dB pour au moins 5 sur 6.
- Sinon, non tranché, avec les six valeurs.

**R2, proximité au réel.** D_fs = moyenne sur les bandes 1 à 16 kHz de |L_fs − L_réel|, en dB,
où L_réel est le niveau de la cible Big Muff par bande. On compare D_fs à D_48 au même
checkpoint.
- « Plus proche à fs » si D_fs ≤ D_48 − 0,25 dB pour au moins 5 checkpoints sur 6.
- « Plus loin » si D_fs ≥ D_48 + 0,25 dB pour au moins 5 sur 6. C'est la piste B, et c'est le
  résultat qui donnerait tort à l'oreille.
- Sinon, non tranché.
- Rapporté avec : l'écart signé L_48 − L_réel moyen sur 4 à 16 kHz. Si le modèle à 48 kHz n'est
  pas plus brillant que le réel, retirer de l'aigu ne peut pas le rapprocher.

**R3, mécanisme, sur sinus.** Pour f0 ∈ {211, 997, 3001} Hz (fréquences premières en Hz) et deux
niveaux d'entrée : valeur efficace des segments de test, puis 12 dB en dessous. Chaque sinus dure
1,5 s, on analyse la dernière seconde ramenée à 48 kHz : 48 000 échantillons, cases de 1 Hz,
fenêtre rectangulaire, un nombre entier de périodes. Les produits repliés k·f0 mod 48 000 tombent
alors sur des cases hors de la grille harmonique.
- I_fs = énergie hors des cases harmoniques k·f0 et hors de la case 0, sur l'énergie totale, en
  dB.
- H_fs = médiane sur les harmoniques de 4 à 16 kHz de |L_fs(k f0) − L_48(k f0)|, en dB.
- « Repliement » (piste A) si, au niveau du jeu de test, I_192 ≤ I_48 − 10 dB et H_192 ≤ 0,5 dB,
  pour au moins 5 checkpoints sur 6, avec f0 = 997 Hz comme cas principal.
- « Réponse changée » si H_192 > 1 dB pour au moins 5 sur 6 : les harmoniques elles-mêmes
  changent, ce qui relève de la discrétisation ou d'une réponse linéaire, pas du repliement.
- Sinon, non tranché. Les autres f0 et le niveau −12 dB sont rapportés sans lecture propre.
- Écart déclaré avant toute mesure (2026-10-02, en écrivant le script) : chaque sinus dure 1,6 s
  et l'on analyse la fenêtre de 0,5 à 1,5 s, toujours 48 000 échantillons et un nombre entier de
  périodes. La « dernière seconde » d'un sinus de 1,5 s aurait contenu l'effet de bord de fin
  de fichier du rééchantillonneur. Le sinus est engendré directement à chaque fréquence.
- Réserve : 192 kHz n'est pas exempt de repliement. Les harmoniques au-delà de 96 kHz se
  replient encore, ce qui borne la baisse de I_192.

**Ce qui départage A et B.** A prédit R3 « repliement », et R2 « plus proche » si le modèle à 48 kHz
est plus brillant que le réel. B prédit R3 « repliement » et R2 « plus loin ». Une réponse changée
(R3) ouvre une troisième voie, ni A ni B.

**Descriptif, sans lecture** : les segments 10 et 8 écoutés ; le résidu ZOH compensé de son retard
fractionnaire (retards médians de `kappa.json`) ; le ZOH à 96 kHz.

## Ce qui ne pourra pas être conclu

- Rien sur l'audibilité : les seuils sont en dB, pas perceptifs. Une écoute à l'aveugle reste à
  faire, la page a un mode aveugle.
- Rien hors Big Muff ni hors SSM-WaveNet.
- Rien sur l'état d'un modèle réentraîné à 96 kHz.

## Résultats (2026-10-02, `brightness.py`, `brightness.json`)

12 segments de test, valeur efficace d'entrée 0,103. Témoin du rééchantillonneur : 0,000 dB sur
4 à 16 kHz pour les six checkpoints. Le sinc de Kaiser ne colore rien ; tout écart vient du
modèle.

| checkpoint | 48 − réel, 4–16 kHz | B_96 | B_192 | D_48 | D_96 | D_192 | I_48 | I_192 | H_192 |
|---|---|---|---|---|---|---|---|---|---|
| ZOH s925 | −0,01 | −0,33 | −0,53 | 0,08 | 0,27 | 0,38 | −36,6 | −56,5 | 0,97 |
| ZOH s736 | −0,09 | −0,34 | −0,56 | 0,08 | 0,26 | 0,38 | −38,9 | −59,6 | 0,87 |
| ZOH s688 | −0,13 | −0,10 | −0,18 | 0,10 | 0,16 | 0,22 | −37,8 | −69,2 | 0,95 |
| FOH s925 | +0,01 | +0,20 | +0,24 | 0,06 | 0,13 | 0,16 | −35,3 | −58,5 | 0,37 |
| FOH s736 | −0,02 | +0,18 | +0,22 | 0,07 | 0,10 | 0,12 | −37,6 | −56,6 | 0,33 |
| FOH s688 | −0,23 | +0,22 | +0,27 | 0,14 | 0,10 | 0,12 | −37,4 | −47,9 | 0,57 |

En dB ; I et H : sinus à 997 Hz au niveau du jeu de test.

**Lectures pré-inscrites : toutes « non tranché ».**
- R1 (96 et 192) : 2 checkpoints sur 6 à ≤ −0,5 dB à 192 kHz, aucun à 96. Les ZOH perdent de
  l'aigu (−0,18 à −0,56 dB à 192), les FOH en gagnent (+0,22 à +0,27). Les signes s'opposent
  selon la discrétisation.
- R2 (96 et 192) : « plus loin » pour 2 sur 6 seulement (ZOH s925 et s736, +0,30 dB). Les autres
  écarts sont sous 0,25 dB, un seul est négatif (FOH s688, −0,02). Condition préalable : le
  modèle à 48 kHz n'est pas plus brillant que le réel (−0,23 à +0,01 dB sur 4 à 16 kHz) ; ses
  niveaux par bande sont déjà à 0,06–0,14 dB du réel en moyenne sur 1 à 16 kHz. Retirer de l'aigu
  ne pouvait donc pas le rapprocher, et 192 kHz ne le rapproche pas.
- R3 : la condition sur I tient pour les 6 checkpoints (baisse de 10,5 à 31,4 dB), celle sur
  H_192 ≤ 0,5 dB pour 2 sur 6 seulement (les trois ZOH entre 0,87 et 0,97 dB, FOH s688 0,57).
  Aucun H_192 ne dépasse 1 dB. Le repliement existe et 192 kHz en retire l'essentiel, mais les
  harmoniques bougent aussi, plus en ZOH.

**Descriptif (segments écoutés 10 et 8, checkpoint s925).** ZOH à 192 kHz : −0,49 et −0,55 dB
sur 4 à 16 kHz, ce que l'oreille a entendu. Le résidu de la page vaut une ESR de 0,177. Compensé
du retard fractionnaire de `kappa.json`, il tombe à 0,0057 : le retard fait environ 97 % du
résidu ZOH entendu. FOH à 192 kHz : +0,24 dB, ESR 0,0009 face au modèle à 48 kHz.

**Ce qu'on en tire, sans lecture pré-inscrite.**
- L'aigu en moins à 192 kHz est propre au ZOH. Le FOH, qui perd lui aussi son repliement,
  devient un peu plus brillant. Le retrait du repliement n'explique donc pas seul la perte d'aigu
  du ZOH ; le changement de réponse du ZOH quand Δ change (H_192 de 0,9 dB) en est le candidat.
- Les niveaux par bande ne tranchent ni A ni B : le modèle à 48 kHz colle déjà au réel à
  0,1 dB près, et les écarts à 192 kHz restent sous 0,4 dB. Ils ne voient pas la nature
  inharmonique du repliement, que seule l'écoute ou une mesure sur sinus distingue.
- Rapprochement post hoc, à tester : la part inharmonique à 48 kHz (−35 à −39 dB) est du même
  ordre que le plancher à −37 dB entre FOH à 88–192 kHz et 48 kHz (`predictions_floor.md`). Ce
  plancher pourrait être le repliement retiré.
- Décision produit (2× interne) : non tranchée. Le repliement baisse de 10 à 18 dB à 96 kHz sur
  le sinus à 997 Hz, sans rapprocher les niveaux du réel. L'étape suivante est une écoute à
  l'aveugle du FOH à 48 contre 192 kHz : s'il paraît encore moins dur, c'est le repliement ;
  sinon, l'impression venait de l'aigu perdu par le ZOH.
