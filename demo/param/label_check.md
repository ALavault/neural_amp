# Étiquette ToneTwist `B030_M070_T100_G050` : vérification contre l'original (Q4)

Écrit le 2026-09-24, **avant tout calcul**. Voie produit.

**Constat de départ.** L'enregistrement d'origine (Zenodo 7970723, CC-BY-4.0, archive vérifiée :
md5 conforme à Zenodo, sha256 conforme à `datasets/manifests/catalog.yaml`) nomme ses neuf réglages
de test ; ToneTwist en reprend huit à l'identique. Le neuvième diffère :

| original | ToneTwist |
|---|---|
| B1_M3_T7_G5 | B010_M030_T070_G050 |
| **B3_M7_T1_G5** | **B030_M070_T100_G050** |
| B6.5_M8.5_T3.5_G5 | B065_M085_T035_G050 |

Le motif (1, 3, 7) puis (3, 7, 1) de l'original suggère T1.

**Mesures, dans cet ordre.**
1. *Même enregistrement ?* La cible ToneTwist du réglage suspect contre la sortie haut-parleur
   d'origine B3_M7_T1 : décalage par intercorrélation, puis ESR après gain optimal.
   - Témoin : B010_M030_T070 contre B1_M3_T7, dont l'étiquette concorde.
2. *Quelle valeur de treble le modèle reconnaît-il ?* Le checkpoint `param_ssm_wavenet_marshall`
   est évalué sur ce réglage avec la valeur de treble 1,0 (étiquette ToneTwist) puis 0,1
   (étiquette d'origine), et tout le reste identique. Le modèle a appris le treble sur les
   réglages d'entraînement, qui font varier ce bouton seul (T0, T2, T4, T5, T6, T8, T10) : il
   sert d'instrument indépendant de l'étiquette. Mesure post hoc au sens où le réglage est connu
   comme suspect, mais écrite avant son résultat.

**Prédictions.**
- *Coquille de ToneTwist, audio = B3_M7_T1* : en 1, ESR du réglage suspect du même ordre que le
  témoin et ≤ 10⁻³ ; en 2, ESR(T = 0,1) ≤ 0,5 × ESR(T = 1,0).
- *Audio différent* : en 1, ESR du réglage suspect au moins dix fois celle du témoin. Q4 reste
  alors ouverte, et la mesure 2 dit seulement quelle étiquette le modèle préfère.
- *L'original se trompe et ToneTwist a corrigé* : en 1, même enregistrement ; en 2,
  ESR(T = 1,0) ≤ 0,5 × ESR(T = 0,1).
- Tout autre cas : rapporté tel quel.

## Résultats (2026-09-24, `label_check.py`, chiffres dans `label_check.json`)

**Mesure 1 : c'est le même enregistrement.** La cible ToneTwist est un extrait de 95 s de la sortie
haut-parleur d'origine, commençant 300 s après son début (13 230 000 échantillons à 44,1 kHz).

| paire | corrélation | gain | ESR |
|---|---|---|---|
| **B030_M070_T100 (ToneTwist) contre B3_M7_T1 (original)** | 0,99999999999998 | 0,9999998 | **1,6·10⁻¹³** |
| B010_M030_T070 contre B1_M3_T7 (témoin) | 0,99999999999998 | 0,9999997 | 2,7·10⁻¹³ |

**Mesure 2 : le modèle reconnaît treble = 0,1.** Checkpoint `param_ssm_wavenet_marshall`,
17 segments du réglage suspect :

- ESR 0,3707 avec treble = 1,0, la valeur enregistrée par le banc ;
- **ESR 0,0182** avec treble = 0,1, soit un rapport de 0,049.

**Verdict : prédiction « coquille de ToneTwist » tenue sur les deux critères.** L'audio est celui de
B3_M7_T1 : ESR ≤ 10⁻³, au niveau du témoin. Le modèle, qui a appris le treble sur les réglages
d'entraînement, y reconnaît 0,1 : rapport ≤ 0,5. L'étiquette correcte est **treble = 1 sur 10**,
soit `B030_M070_T010_G050`. Q4 est close ; la réponse est « étiquette fausse », et non plus
seulement « compatible avec une étiquette fausse ».

**Conséquence, descriptive et post hoc** : la même mesure sur `param_s4_marshall` donne une ESR de
0,6314 avec treble = 1,0 et de **0,1229** avec 0,1. Sur les neuf réglages non vus, avec le seul
bouton de ce réglage corrigé et sans rien réentraîner :

| | SSM-WaveNet-FiLM | S4-param |
|---|---|---|
| ESR moyenne, étiquette ToneTwist (enregistrée) | 0,1008 | 0,2833 |
| ESR moyenne, étiquette corrigée | **0,0617** | **0,2268** |
| médiane, ToneTwist puis corrigée | 0,0475 puis 0,0442 | 0,2034 puis 0,1869 |
| rang du réglage corrigé, sur 9 (1 = pire) | 7 | 9 |

Avec l'étiquette corrigée, l'écart entre modèles passe de ×2,8 à ×3,7 en moyenne. Le verdict Q2
(combinaison extrême contre interpolation) ne dépend pas de ce réglage.

**Pour les runs futurs** : `B030_M070_T100_G050` n'est qu'en test. Les runs G2 évalueront ce
réglage avec les deux valeurs et rapporteront les deux, l'étiquette corrigée étant la valeur
primaire (`demo/param/predictions_poles.md`).
