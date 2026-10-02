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
