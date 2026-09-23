# Transfert de fréquence par rééchelonnement de Δ — résultats

Voie produit (D-PRODUCT-001) : mesure produit sur le jeu de test Big Muff, aucune
revendication scientifique. Prédictions : `predictions.md` (commit f34cec5, avant tout
chiffre inter-fréquences, non modifié depuis). Chiffres : `sr_transfer_*_last.json`
(bras pré-enregistrés) et `sr_transfer_*_last_posthoc.json` (contrôles ajoutés après la
relecture Codex, sans prédiction). Script : `scripts/product_sr_transfer.py`. Tous les
checkpoints `last` des six runs appariés sont présents ; aucun n'a été écarté.

## Tableau (ESR ; « vs 48k » = cohérence avec le modèle à 48 kHz, pas fidélité)

| run | 48k | 44,1 around | 44,1 delta | 44,1 naive | 44,1 lead | 88,2 delta | 96 delta | 192 delta | 96 naive | 44,1 up2 | 44,1 up4 | pôles repliés 48k |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ssm s925 | 0,0439 | 0,0441 | 0,0396 | 0,2771 | 0,0570 | 0,0650 | 0,0774 | 0,1619 | 1,310 | 0,1354 | 0,2314 | 2/512 |
| ssm s736 | 0,0500 | 0,0502 | 0,0490 | 0,2601 | 0,0632 | 0,0557 | 0,0663 | 0,1377 | 1,307 | 0,1165 | 0,1934 | 0/512 |
| ssm s688 | 0,0425 | 0,0426 | 0,0412 | 0,2499 | 0,0515 | 0,0364 | 0,0434 | 0,0927 | 1,320 | 0,0897 | 0,1437 | 0/512 |
| s4 s925 | 0,0850 | 0,0852 | 0,0841 | 0,2906 | 0,1075 | 0,1326 | 0,1580 | 0,3403 | 1,304 | 0,2067 | 0,3845 | 531/4096 |
| s4 s736 | 0,0811 | 0,0813 | 0,0813 | 0,2888 | 0,0929 | 0,1299 | 0,1473 | 0,3071 | 1,309 | 0,1984 | 0,3541 | 517/4096 |
| s4 s688 | 0,0789 | 0,0791 | 0,0775 | 0,2802 | 0,1328 | 0,1152 | 0,1467 | 0,2806 | 1,311 | 0,2007 | 0,3388 | 509/4096 |

(s4 s688 a fini après l'écriture des prédictions ; ajouté selon la règle qu'elles fixaient.)

## Verdicts

**P1** (`around` à ±5 % de l'ESR 48 kHz) — rapports 1,004 / 1,004 / 1,004 (ssm s925 /
s736 / s688). **Tenue** pour les trois graines. (S4 : 1,002 / 1,003 / 1,003.)

**P2** (`delta` ≤ 1,05 × `around` à 44,1 kHz) — rapports 0,899 / 0,977 / 0,967.
**Tenue** pour les trois graines ; `delta` est même meilleur que `around`. Contrôle post
hoc (Codex, point 5 : zéros ajoutés aux bords par le rééchantillonnage segment par
segment) : sur l'intérieur des segments et avec un rééchantillonnage du fichier entier,
`delta` reste devant (0,0348 contre 0,0393 ; 0,0450 contre 0,0462 ; 0,0361 contre 0,0375).
**Règle produit satisfaite** : à 44,1 kHz, le rééchelonnement de Δ remplace le
rééchantillonneur pour SSM-WaveNet, sans perte mesurable.

**P3** (`naive` > `delta`, rapport entre 1,02 et 1,5) — rapports 7,0 / 5,3 / 6,1.
**Partielle** pour les trois graines : le sens tient, l'ampleur est réfutée, 4 à 5 fois
plus grande que prévu. L'analogie avec les LSTM de Wilczek et al. (0,2 à 0,6 dB) ne se
transpose pas : ne rien faire à 44,1 kHz multiplie l'ESR de ce modèle par 5 à 7.

**P4** (`delta_lead` à ±2 % de `delta` à 44,1 kHz) — rapports 1,44 / 1,29 / 1,25.
**Réfutée** pour les trois graines. Réserve de Codex (point 3) : le bras multiplie C par
exp(λε) sans le terme d'entrée qu'exige une vraie avance temporelle ; c'est une
modification des résidus modaux, pas un décalage. L'échec réfute ce bras, pas
l'hypothèse d'un décalage de branche, qui reste non testée.

**P5** (`delta_vs_around` croît avec fs ; ESR 0,005 à 0,05 à 96 kHz) — 88,2 < 96 < 192
pour les trois graines ; à 96 kHz, 0,0774 / 0,0663 / 0,0434. **Partielle** pour s925 et
s736 (croissance tenue, niveau au-dessus de 0,05), **tenue** pour s688. Post hoc
(Codex, point 6) : comparé dans la bande commune (sortie redescendue à 48 kHz contre le
modèle à 48 kHz), l'ESR est identique au quatrième chiffre (96 kHz : 0,0773 / 0,0661 /
0,0432). L'écart ne vient donc pas du contenu au-dessus de 24 kHz : il est dans la bande
audio.

**P6** (`lead` réduit l'écart d'au moins 30 % à 96 et 192 kHz) — rapports lead/delta
1,16 et 1,04 (s925), 2,47 et 2,22 (s736), 3,35 et 2,62 (s688). **Réfutée** pour les
trois graines : le bras aggrave l'écart. Même réserve que P4.

**P7** (`naive` ≥ 5 × `delta` à 96 et 192 kHz) — rapports 16,9 et 7,6 ; 19,7 et 8,9 ;
30,4 et 13,4. **Tenue** pour les trois graines. Sans rééchelonnement, l'ESR dépasse 1,2
(la sortie est plus éloignée de la référence que la référence ne l'est de zéro).

**P8** (sur-échantillonnage ×2 et ×4 puis retour à 44,1 kHz, contre l'appareil ; aucun
sens prédit) — rapports à `around` : ×2 3,07 / 2,32 / 2,10 ; ×4 5,25 / 3,85 / 3,37.
**Sans objet** pour les trois graines (aucun sens n'était prédit) ; le résultat observé
est net : faire tourner le modèle à 2 ou 4 fois la fréquence d'entraînement, sans le
réentraîner, dégrade la fidélité à l'appareil d'un facteur 2 à 5.

**P9** (S4-TF-L-16 : gain naive → delta plus petit que SSM-WaveNet ; `delta_vs_around` à
96 kHz ≥ 3 × SSM-WaveNet, par découpage) — gain S4 3,5 / 3,6 / 3,6 contre SSM 7,0 / 5,3 /
6,1 : première partie tenue pour les trois. Rapport 96 kHz S4/SSM 2,04 (s925), 2,22
(s736), 3,38 (s688). **Partielle** pour s925 et s736, **tenue** pour s688.

## Ce qu'on en retient pour le produit

- **Descente à 44,1 kHz : résolue pour les deux architectures.** Δ rééchelonné égale ou
  bat le rééchantillonnage hors ligne (SSM 0,899 à 0,977 ; S4 0,980 à 0,999), sans latence
  ni filtre. S4, dont environ 12,5 % des modes sont déjà repliés à 48 kHz, transfère
  aussi bien à 44,1 kHz : le repliement des pôles ne se manifeste pas à cette distance.
- **Ne rien faire coûte cher** : ESR ×5 à ×7 à 44,1 kHz, au-delà de 1,2 à 96 kHz.
- **Montée (88,2 à 192 kHz) : non résolue.** L'écart au modèle 48 kHz est dans la bande
  audio, croît avec fs et vaut 2 à 3 fois plus pour S4. Le sur-échantillonnage interne
  (modèle à 88,2 kHz puis retour) dégrade la fidélité à l'appareil. Pour un hôte à
  88,2/96 kHz, la voie sûre reste le rééchantillonnage autour du modèle à 48 kHz.
- **Mécanisme de la montée : ouvert.** Deux pistes non testées : (i) le modèle a appris
  avec son propre repliement à 48 kHz (les harmoniques qui se repliaient et qu'il
  compensait ne se replient plus) ; (ii) un décalage de branche dépendant de fs, que le
  bras `lead` ne teste pas correctement (Codex). Mesure qui les départagerait : l'écart
  dans la bande commune pour un signal d'entrée faible (non-linéarités peu actives,
  repliement négligeable) — (i) prédit un écart qui s'effondre, (ii) un écart qui reste.

## Relecture Codex (task, lecture seule, 2026-09-23)

1. Rééchelonnement ZOH correct pour les deux SSM ; la docstring disait à tort que toute la
   mémoire de S4 est un SSM (TFiLM à blocs de 128). **Corrigée.**
2. Restauration des paramètres correcte ; `around` dépendait de l'ordre des bras.
   **`set_rate(48 kHz)` explicite ajouté**, sans effet sur les chiffres (ordre inchangé).
3. Le bras `lead` n'est pas une avance temporelle. **Docstring corrigée, réserve portée
   dans P4 et P6** ; les JSON ne sont pas recalculés.
4. Moyenne des scores conforme à nablafx ; la porte à 48 kHz affiche sans tolérance.
   **Laissé tel quel** : les six portes sont dans les JSON (écarts ≤ 0,1 %).
5. Bords de segment non mesurés. **Mesurés post hoc** (P2) : conclusions inchangées.
6. Contenu hors bande compté comme erreur. **Mesuré post hoc** (P5) : il ne compte pas.
