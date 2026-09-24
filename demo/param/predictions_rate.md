# Le modèle paramétrique FiLM change-t-il de fréquence comme le modèle fixe ? (étape 2b)

Voie produit. Écrit le 2026-09-24, **avant tout calcul**.

**Pourquoi c'est une question.** Le FiLM de SSM-WaveNet agit sur les couches 1×1, sans mémoire :
il ne dépend pas de la fréquence d'échantillonnage. Rééchelonner Δ (et relire le modèle en FOH)
devrait donc transférer le modèle paramétrique comme il transfère le modèle fixe de Big Muff. Le
Marshall de ToneTwist est capturé nativement à 44,1 kHz ; nablafx le rééchantillonne à 48 kHz pour
l'entraînement. La cible native à 44,1 kHz sert donc de référence appareil.

**Dispositif.** Checkpoint `param_ssm_wavenet_marshall` (dernier), les neuf réglages de test, dont
le réglage mal étiqueté évalué avec treble = 0,1 (`label_check.md`). Scores sur l'intérieur des
segments, 0,1 s retiré de chaque côté.

- À 44,1 kHz, contre la cible native à 44,1 kHz :
  - `delta` : Δ rééchelonné ;
  - `foh` : relecture FOH, puis Δ rééchelonné ;
  - `around` : rééchantillonnage vers 48 kHz, modèle, retour à 44,1 kHz (hors ligne).
- À 96 et 192 kHz, contre le modèle à 48 kHz dans la bande commune : `delta` et `foh`.

**Prédictions** (médianes sur les neuf réglages) :

- **P1** : à 44,1 kHz, ESR(`delta`) / ESR(`around`) entre 0,85 et 1,05, comme sur Big Muff
  (0,90 à 0,98).
- **P2** : à 44,1 kHz, ESR(`foh`) / ESR(`around`) entre 0,95 et 1,05. La relecture FOH reproduit
  le modèle à 48 kHz à toute fréquence, donc `around` à l'effet du rééchantillonnage près.
- **P3** : à 96 kHz, `foh` réduit l'écart au modèle 48 kHz d'au moins ×10 par rapport à `delta`,
  comme la correction à deux coefficients sur Big Muff (×219 à ×348).
- **P4, sans seuil** : le plancher de `foh` à 96 kHz est rapporté tel quel. Le Marshall a plus de
  gain que le Big Muff, et son repliement non linéaire devrait être plus fort : on s'attend à un
  plancher plus haut que les 2·10⁻⁴ du Big Muff, sans chiffre prédit.

Chaque prédiction est jugée sur la médiane, et le détail par réglage est rapporté.

## Résultats (2026-09-24, `rate_transfer.py`, `rate_transfer.json`)

Écart de mise en œuvre : à 48 kHz, nablafx compte ses segments d'après la longueur du fichier à
44,1 kHz. Il en garde 17 par réglage, soit 85 s sur 95, contre 19 à 44,1 kHz. Seuls les 153
segments natifs qui ont leur jumeau à 48 kHz sont notés.

Médianes sur les neuf réglages :

| | 44,1 kHz contre l'appareil | 96 kHz contre 48 kHz | 192 kHz contre 48 kHz |
|---|---|---|---|
| `delta` | 0,0416 | 0,0516 | 0,1007 |
| `foh` | 0,0435 | 0,0096 | 0,0097 |
| `around` | 0,0444 | — | — |

**Verdicts**, sur la médiane des rapports par réglage, comme écrit :

- **P1 tenue** : `delta` / `around` = 0,96 (fourchette 0,85 à 1,05 ; de 0,87 à 1,14 selon le
  réglage).
- **P2 tenue** : `foh` / `around` = 0,98 (fourchette 0,95 à 1,05 ; de 0,95 à 1,00 selon le
  réglage). La relecture FOH reproduit le modèle à 48 kHz à l'effet du rééchantillonnage près,
  réglages compris.
- **P3 non tenue** : à 96 kHz, `foh` ne réduit l'écart que d'un facteur médian de 7,0 (5,1 à 9,4
  selon le réglage), sous le ×10 prédit. À 192 kHz, la médiane des rapports vaut 13,3, parce que
  `delta` s'aggrave avec la fréquence et pas le plancher.
- **P4** : le plancher de `foh` vaut 0,0096 à 96 kHz et 0,0097 à 192 kHz, soit environ −20 dB.
  Il ne dépend presque pas de la fréquence cible, comme sur Big Muff, mais il est environ 45 fois
  plus haut (2,2·10⁻⁴ sur Big Muff). C'est le sens attendu pour un appareil plus saturé, dont les
  non-linéarités replient davantage. Ce n'est pas vérifié ici par un test de niveau.

**Lecture produit.** Le modèle paramétrique change de fréquence vers le bas sans perte, et sans
toucher aux réglages : le FiLM, sans mémoire, ne s'en aperçoit pas. Vers le haut, la relecture FOH
supprime l'essentiel de l'écart dû à la discrétisation. Il reste un écart de −20 dB entre le
modèle à 48 kHz et le même modèle suréchantillonné. Sur un ampli saturé, c'est la part
vraisemblablement due au repliement à 48 kHz, et donc un argument pour faire tourner le modèle
au-dessus de 48 kHz plutôt qu'une erreur à corriger ; à confirmer par un test de niveau.
