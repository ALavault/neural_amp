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

## Résultats (2026-10-02, `fast_modes.json`)

Les comptes retrouvent ceux de `pole_speed.json` : 22 modes rapides (9, 4, 1, 4 et 4 à 925, 736,
688 `_r2`, 744 et 369), dont 9 commutés (4, 1, 0, 2 et 2). Notation bloc/canal/mode ; ln r est la
médiane sur les 9 non vus.

**C1, ablation : 19 morts, 2 mineurs, 1 porteur.**
- Porteur : 744, 0/0/2 (commuté), ln r = +0,2597 (ESR des non vus × 1,30).
- Mineurs : 736, 2/9/2 (commuté), +0,0470 ; 369, 6/11/2 (commuté), +0,0313.
- Tous les modes rapides non commutés sont morts (|ln r| ≤ 0,0042) ; 6 des 9 commutés le sont
  aussi (de −0,0049 à +0,0038).

**C2 : G domine** D + K[0] dans **6 des 8** canaux qui portent un mode commuté (seuil : 2/3, soit
au moins 6). Les deux exceptions :
- 925, 0/0 : étendue G 1,643, T 0,893, B 0,302, M 0,072 ;
- 369, 6/11 : étendue T 0,366, M 0,063, G 0,027, B 0,016.

Dans les 6 autres, l'étendue de G va de 0,102 à 1,803, et le plus grand des trois autres
boutons de 0,001 à 0,144.

**C3 : commandés par G.** La part de G dans Δα est ≥ 0,5 pour **7 des 9** modes commutés (de 0,72
à 0,93) ; seuil 2/3, soit au moins 6. Exceptions :
- 925, 0/10/2 : G 0,22, T 0,40, B 0,25, M 0,14 ;
- 369, 6/11/2 : G 0,24, T 0,35, M 0,29, B 0,12.

**Ce qui se dit.**
- Les modes rapides du bras D sont presque tous inertes à l'inférence. Leur dérive ne porte pas
  la sortie, à une exception près (744, +30 % d'ESR).
- Les modes commutés et le terme instantané des canaux qui les portent suivent surtout le bouton
  Gain. Or G vaut 5 sur tous les réglages de test : ce mécanisme ne peut pas expliquer
  l'avantage de D sur les non vus, qui diffèrent par B, M et T.
- Descriptif, hors critère : le seul porteur (744, 0/0/2) est commandé par G (part 0,93).
  Son Δα dépend donc surtout de G, qui ne varie pas sur les réglages de test ; ce qu'il apporte
  à ces réglages n'est pas analysé ici.
