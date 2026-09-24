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
