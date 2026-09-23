# Maintien appris contre repliement appris : résultats

Voie produit, mesure produit. Prédictions : `predictions_hold.md` (commits 61b5d1e et
936b7ef, avant tout chiffre). Chiffres : `sr_transfer_*_last_holdtest.json`. Tous
scores sur l'intérieur des segments ; montée comparée au modèle à 48 kHz dans la bande
commune (ESR). Six checkpoints, aucun écarté.

## Tableau (ESR ; « réd. » = delta / delta_hold)

| run | 44,1 delta | 44,1 hold | 88,2 delta → hold (réd.) | 96 delta → hold (réd.) | 192 delta → hold (réd.) | −20 dB / 0 dB (delta, 96 et 192) | hold 192/96 |
|---|---|---|---|---|---|---|---|
| ssm s925 | 0,0348 | 0,0403 | 0,0649 → 0,0120 (×5,42) | 0,0773 → 0,0144 (×5,38) | 0,1616 → 0,0325 (×4,98) | 0,57 / 0,59 | 2,26 |
| ssm s736 | 0,0450 | 0,0484 | 0,0556 → 0,0134 (×4,14) | 0,0661 → 0,0160 (×4,13) | 0,1375 → 0,0350 (×3,93) | 0,57 / 0,58 | 2,18 |
| ssm s688 | 0,0361 | 0,0381 | 0,0362 → 0,0137 (×2,63) | 0,0432 → 0,0162 (×2,67) | 0,0924 → 0,0321 (×2,88) | 0,56 / 0,57 | 1,99 |
| s4 s925 | 0,0820 | 0,1133 | 0,1317 → 0,0950 (×1,39) | 0,1572 → 0,1142 (×1,38) | 0,3399 → 0,2473 (×1,37) | 0,29 / 0,28 | 2,16 |
| s4 s736 | 0,0785 | 0,0821 | 0,1299 → 0,0885 (×1,47) | 0,1473 → 0,0971 (×1,52) | 0,3073 → 0,2163 (×1,42) | 0,25 / 0,24 | 2,23 |
| s4 s688 | 0,0746 | 0,1558 | 0,1152 → 0,1540 (×0,75) | 0,1468 → 0,1510 (×0,97) | 0,2810 → 0,2905 (×0,97) | 0,34 / 0,37 | 1,92 |

## Verdicts (SSM-WaveNet, graine par graine)

**H1** (réduction ≥ ×5 à 96 et 192 kHz ; réfutée sous ×2) — ×5,38 et ×4,98 (s925),
×4,13 et ×3,93 (s736), ×2,67 et ×2,88 (s688). **Partielle** pour les trois (s925 tenue à
96 kHz, à la limite à 192). La correction causale d'une ligne divise l'écart de montée
par 2,7 à 5,4 : le maintien appris explique la majeure partie de l'écart, pas tout.

**H2** (à 44,1 kHz, `delta_hold` à ±2 % de `delta`) — +16 %, +7,5 %, +5,6 %. **Réfutée**
pour les trois. En descente, le noyau « porte de 48 kHz » dégrade : c'est la limite
déclarée avant les résultats (somme de Riemann, modes rapides), plus sévère quand dt′
dépasse la porte. Règle produit : `delta` sous 48 kHz, `delta_hold` au-dessus.

**H3** (rapport −20 dB / 0 dB de `delta` : 0,5 à 2 si le maintien domine, < 0,2 si le
repliement domine) — 0,56 à 0,59 aux deux fréquences, pour les trois. **Tenue**, près de
la borne basse : l'écart ne s'effondre pas à faible niveau, donc le repliement appris
n'est pas la cause principale ; une part de l'écart (environ 40 %) dépend du niveau.

**H4** (le rapport 192/96 de `delta_hold` s'écarte nettement de 2,25) — 2,26 ; 2,18 ;
1,99. **Réfutée** : le résidu garde la loi en τ². La correction enlève une partie de
l'erreur de discrétisation sans en changer la nature, ce que prévoyait la limite déclarée
(un seul coefficient par mode ne peut pas à la fois garder le DC et rétablir l'avance).

## Contraste S4-TF-L-16

**H5** (réduction entre ×1,5 et ×5) — s736 ×1,52 à 96 kHz (tenue), ×1,42 à 192
(réfutée) ; s925 ×1,38 et ×1,37 (réfutée) ; s688 ×0,97 (réfutée, aucun effet). À 44,1
kHz, le bras dégrade fortement s688 (×2,09). Environ 12,5 % des modes de S4 sont déjà
repliés à 48 kHz : leur « porte » n'a pas d'équivalent continu défini. Le rapport
−20 dB / 0 dB vaut 0,24 à 0,37 pour S4 : l'écart de S4 dépend nettement plus du niveau
(TFiLM, non-linéarités), ce que la correction de discrétisation ne touche pas.

## Ce qu'on en retient

- **Mécanisme.** Pour SSM-WaveNet, l'écart de montée vient surtout de l'avance d'un
  demi-échantillon que le ZOH a figée à 48 kHz, et pas du repliement appris : une
  correction causale de ce seul effet le divise par 2,7 à 5,4, et l'écart ne s'effondre
  pas à −20 dB.
- **Produit.** À 88,2 et 96 kHz, SSM-WaveNet corrigé reste à un ESR de 0,012 à 0,016
  de son propre modèle à 48 kHz (contre 0,036 à 0,077 sans correction).
- **Suite logique, non lancée.** (i) Une correction exacte à deux coefficients par mode
  (garder DC et avance), dont la loi en τ² résiduelle devrait disparaître ; (ii) entraîner
  avec une règle symétrique en entrée (trapèze complet ou maintien du premier ordre,
  `verif_biblio_A_C_D_F.md`), qui supprime l'avance au lieu de devoir la corriger.
