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
