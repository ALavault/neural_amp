# Loi de déclenchement de l'ordonnanceur à patience

**Pré-enregistré le 2026-10-02, avant toute mesure de ce document.** Lecture seule, journaux
existants, CPU. Script : `diagnosis/butterfly/patience_law.py`, sortie `patience_law.json`.

## Question et cadre

`ReduceLROnPlateau` (mode min, seuil relatif θ = 10⁻⁴, patience 20) divise le pas quand 21
époques passent sans record de la perte de validation `loss/val/tot`. L'hypothèse : l'époque de la
première division est un temps de premier passage aléatoire, fixé par le rapport entre la dérive
de la perte et son bruit, et non par θ. Deux modèles de records sont possibles :

- **tendance + bruit indépendant** (Franke, Wergen et Krug 2010) ;
- **marche avec dérive**, où le bruit est cumulatif (Wergen, Bogner et Krug 2011).

Le choix dépend de la nature des fluctuations d'une époque à l'autre.

## Ce qu'on sait déjà

- `plateau_margin.md`, correction du 2026-10-02 : en fluctuation locale, le seuil est 370 à 680
  fois sous le bruit, et les marges de déclenchement valent 0,00 à 0,54.
- Une relecture critique du 2026-10-01 a vu une autocorrélation de 0,2 à 0,4 au retard 1 des
  résidus, après un détendançage grossier, sur quelques runs.

## Runs

- Les 5 enfants « décide » de la fourche 100 (`butterfly_ssm_seed42_f100_decide_k0` à `k4`).
  L'ordonnanceur de l'enfant hérite de l'état du parent : la courbe analysée est celle du parent
  (`butterfly_ssm_seed42_parent`, époques 0 à 100) suivie de celle de l'enfant. Les époques de
  l'enfant sont comptées depuis la fourche.
- Les 16 runs `pilotC_decide_seed42` à `seed57`, depuis l'époque 0.
- Époque de première division observée : changement de `lr_by_epoch`
  (`demo/butterfly/<run>.json`) pour les enfants, de la colonne `lr-AdamW/pg1` de `metrics.csv`
  pour le pilote C.

## Étapes et seuils

**Porte de rejeu.** `plateau_simulation.halvings(courbe, 1e-4, 20)`, appliqué à la vraie courbe,
doit reproduire l'époque observée de première division pour au moins 19 runs sur 21. Une seule
convention de décalage (0 ou 1 époque) est permise, la même pour toutes les runs. Sinon, la mesure
est déclarée non calculable comme spécifiée.

**Étape 1, nature des fluctuations.**
- Sur chaque run, on prend le log de la perte de validation sur les 120 époques qui précèdent la
  première division observée (pour le pilote C, à partir de l'époque 20 au plus tôt).
- On retire une tendance quadratique en log(époque + 1), ajustée sur cette fenêtre.
- VR(10) = Var(r[t+10] − r[t]) / (10·Var(r[t+1] − r[t])) sur les résidus r.
- Lecture sur la médiane des 21 runs :
  - VR(10) ≤ 0,3 → **bruit indépendant** autour d'une tendance (VR ≈ 1/k attendu) ;
  - VR(10) ≥ 0,7 → **bruit cumulatif** (VR ≈ 1 attendu) ;
  - entre les deux → **mixte**, et les deux modèles sont simulés et rapportés.

**Étape 2, loi prédite de la première division.**
- Par run : tendance quadratique en log(époque + 1) ajustée du début de la fenêtre à la première
  division observée, puis extrapolée. Écart-type σ des résidus sur les 60 dernières époques de la
  fenêtre.
- 2 000 courbes simulées :
  - modèle indépendant : log L = tendance + σ·ε, ε gaussien indépendant ;
  - modèle cumulatif : marche gaussienne dont le pas a l'écart-type des différences des résidus.
- La vraie courbe est gardée jusqu'au début de la fenêtre, pour que l'état de l'ordonnanceur soit
  le vrai ; les simulations partent de là. Le même rejeu donne la première division de chaque
  courbe. On relève les quantiles à 10, 50 et 90 %.
- **Fuite déclarée** : la fenêtre d'ajustement se termine à la division observée.

**Lectures.**
- **Couverture** : part des 21 runs dont l'époque observée tombe dans leur intervalle prédit
  [q10 ; q90].
  - ≥ 0,60 (au moins 13 sur 21) → la loi de premier passage du modèle retenu est **soutenue** ;
  - ≤ 0,40 (au plus 8) → **réfutée** ;
  - entre les deux → intermédiaire.
- **Indépendance au seuil** : on refait l'étape 2 avec θ = 10⁻³ et θ = 10⁻⁵. Si la médiane des
  21 médianes prédites bouge d'au plus 5 époques → **indépendance au seuil soutenue** ; sinon,
  non soutenue.

## Ce que cela ne dira pas

Une architecture, un appareil, une configuration de l'ordonnanceur. La tendance est un ajustement
paramétrique choisi d'avance : une loi qui ne couvre pas peut venir de la tendance et non du
bruit.
