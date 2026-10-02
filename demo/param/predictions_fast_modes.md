# Modes rapides du bras D : que portent-ils ?

Voie produit. **Pré-enregistré le 2026-10-02, avant toute mesure de ce document.** Lecture seule,
checkpoints existants, CPU. Script : `demo/param/fast_modes.py`, sortie `fast_modes.json`.

## Question

Dans le bras D (pôles conditionnés, tête additive), C et D ne dépendent pas des boutons. Tout gain
réglable plat en fréquence doit donc passer par des modes devenus sans mémoire (s ≫ 1).
- Les modes rapides servent-ils vraiment la sortie ?
- Les modes commutés (s < 1 pour un réglage, s > 10 pour un autre) sont-ils commandés par le
  bouton Gain, qui vaut 5 sur tous les réglages de test ?

## Ce qu'on sait déjà

- Comptes de modes (`pole_speed.json`) : rapides pour au moins un réglage 9, 4, 1, 4 et 4 ;
  commutés 4, 1, 0, 2 et 2 (D à 925, 736, 688 `_r2`, 744, 369).
- Une relecture critique du 2026-10-01, sur les checkpoints de 925 et 736 seulement, a trouvé :
  - les modes commutés surtout commandés par G, via Δα ;
  - les modes « rapides partout » quasi morts (gain de 10⁻⁴ à 10⁻⁶) ;
  - le mode de 688 `_r2` mort.

  Ni ablation ni calcul de D + K[0] n'ont été faits.

## Mesure

**Checkpoints** : `g2_poles_additive_s925`, `s736`, `s688_r2`, `s744` et `s369`.
- Réglages : les 28 réglages d'entraînement (module de données `trainval` du Marshall) et les 9
  réglages non vus de test.
- B030_M070_T100_G050 est évalué à treble 0,1, comme dans `poles_verdict.py`.

**Mode rapide.** s = e^(log_A_real + Δα)·e^(log_dt + Δlog_dt) dépasse 10 pour au moins un réglage
d'entraînement. Il est commuté si son s minimal est inférieur à 1.

**C1, ablation.**
- Pour chaque mode rapide, C[canal, mode] est mis à zéro (parties réelle et imaginaire) à
  l'inférence.
- On calcule l'ESR des 9 non vus avec `poles_verdict.unseen`, et r = ESR ablatée / ESR de base par
  réglage.
- Lecture par mode, sur la médiane des 9 :
  - |ln r| < 0,01 → **mort** ;
  - ln r ≥ ln 1,1 → **porteur** ;
  - entre les deux → mineur.

**C2, terme instantané.**
- Par bloc, canal et réglage d'entraînement : D_c + Σ_modes 2·Re(C·b), avec b = (a − 1)/λ et le
  pôle décalé du réglage (ZOH). C'est le premier échantillon K[0] de la réponse du canal, plus D.
- Étendue par bouton : pour chaque bouton k, max − min sur les réglages de son axe d'entraînement
  (les autres boutons au centre), dans les canaux qui portent un mode commuté.
- Lecture : **G domine** si étendue_G ≥ 2 × max(étendue_B, étendue_M, étendue_T) dans au moins
  2/3 de ces canaux ; sinon, non.

**C3, attribution de Δα.**
- La tête additive donne avant le tanh une somme exacte de fonctions d'un seul bouton :
  f_k(u_k) = W[:, bloc_k]·relu(h_k(u_k)).
- Pour chaque mode commuté : étendue de f_k sur les valeurs d'entraînement du bouton k, et part
  du bouton k = étendue_k / Σ des étendues.
- Lecture : **commandé par G** si la part de G est ≥ 0,5 pour au moins 2/3 des modes commutés ;
  sinon, non.

## Ce que cela ne dira pas

L'ablation mesure l'effet d'un mode sur la sortie à l'inférence, pas ce que l'entraînement aurait
fait sans lui. Une graine, cinq partitions. Les réglages non vus sont joués sur un autre DI que
l'entraînement.
