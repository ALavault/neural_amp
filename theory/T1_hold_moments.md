# T1 — Moments du maintien : écart signé du ZOH, nul en FOH

Preuve de l'entrée T1 de `THEORY.md` (statut : voir le registre). Contrôle :
`theory/t1_check.py` (symbolique et numérique sur le code réel).

## Objet et correspondance

`src/fssr_nam/models/ssm_wavenet.py`, `DiagonalSSMLayer` @ 2d49ac7. Pour un canal et un mode :

| ici | code |
|---|---|
| λ = −e^{α} + iω | `_lambda()` : `-exp(log_A_real) + 1j * A_imag` |
| Δ (en secondes d'échantillon : l'unité de temps est l'échantillon à la fréquence d'entraînement) | `exp(log_dt)` |
| z = λΔ, a = e^{z} | `pole(z)` (plancher Re z ≥ −80, voir Portée) |
| c | `_complex(C)` |
| ZOH : b = (a − 1)/λ, K[k] = 2 Re(c b a^k) | `_b`, `_kernel` |
| FOH : β₁ = Δ φ₂(z), β₀ = Δ e^{z} φ₂(−z), φ₂(u) = (e^u − 1 − u)/u² ; K[0] = 2 Re(c β₁), K[k] = 2 Re(c(β₁ a^k + β₀ a^{k−1})) | `_foh_inputs`, `_foh_kernel` |
| sortie y = K ∗ x + D x | `forward` |

Le mode continu associé a la réponse h(t) = 2 Re(c e^{λt}), de gain continu ∫h = 2 Re(w) avec
w = −c/λ, et de premier moment ∫t h = 2 Re(c/λ²). Mesuré en échantillons, ce premier moment vaut
2 Re(c/(λ²Δ)).

## Énoncé

Énoncé sur la **discrétisation idéale** (arithmétique exacte, noyaux infinis, formules fermées de
la table ci-dessus) ; le code flottant n'est contrôlé que dans le domaine de `t1_check.py` (voir
Portée). Pour Re λ < 0 (toujours vrai ici) et Δ > 0 :

1. **Gain continu.** Σ_k K[k] = 2 Re(w) en ZOH comme en FOH.
2. **FOH.** Σ_k k K[k] = 2 Re(c/(λ²Δ)) : le premier moment discret égale celui du mode continu.
3. **ZOH.** Premier moment continu − premier moment discret = 2 Re(w φ(z)), avec
   φ(z) = −1/z − 1/(e^{−z} − 1) = 1/2 + z/12 − z³/720 + O(z⁵).
4. **Canal.** Avec plusieurs modes et D, quand le gain continu G = Σ_m 2 Re(w_m) + D est non nul,
   le retard de groupe à fréquence nulle (premier moment / gain) du ZOH diffère de celui du
   système continu de Σ_m 2 Re(w_m φ(z_m)) / G échantillons (retard continu moins retard ZOH,
   quantité signée) : une avance pour un mode réel isolé, sans voie directe (point 5) ; aucune
   garantie de signe pour le canal général. Celui du FOH n'en diffère pas (0).
5. **Modes réels** (ω = 0, z = −s, s > 0, Re c ≠ 0, r > 1). L'avance vaut (1 − g(s))/s = 1/s − 1/(e^s − 1), dans
   ]0 ; 1/2[, avec g(x) = x/(e^x − 1). Si l'on remplace Δ par Δ/r (fréquence r fois
   plus haute, sans autre correction), le modèle rééchelonné prend un retard, compté en
   échantillons à la fréquence d'origine, de (g(s/r) − g(s))/s, soit
   κ_mode = 2 (g(s/r) − g(s)) / (s (1 − 1/r)) fois la jauge (1 − 1/r)/2, avec **κ_mode ∈ ]0 ; 1[**,
   → 1 quand s → 0 et → 0 quand s → ∞.

## Preuve

Séries géométriques, |a| < 1 : Σ a^k = 1/(1 − a), Σ k a^k = a/(1 − a)².

1. ZOH : Σ K = 2 Re(c b/(1 − a)) et b/(1 − a) = −1/λ. FOH : Σ K = 2 Re(c(β₁ + β₀)/(1 − a)), et
   β₁ + β₀ = (a − 1)/λ (identité vérifiée par sympy), d'où le même résultat.
2. FOH : Σ_k k K[k] = 2 Re(c [β₁ a/(1 − a)² + β₀ Σ_{k≥1} k a^{k−1}]) = 2 Re(c (β₁ a + β₀)/(1 − a)²),
   et (β₁ a + β₀)/(1 − a)² = 1/(λ²Δ) (sympy).
3. ZOH : Σ k K[k] = 2 Re(c b a/(1 − a)²). La différence 1/(λ²Δ) − b a/(1 − a)² vaut (−1/λ) φ(z)
   (sympy), et la série de φ est celle de l'énoncé (sympy).
4. Linéarité des moments ; D contribue au gain et pas au premier moment.
5. Pour z = −s : φ(−s) = 1/s − 1/(e^s − 1) = (1 − g(s))/s. Le retard en échantillons d'origine
   après Δ → Δ/r est (1 − g(s))/s − (1/r)(1 − g(s/r))/(s/r) = (g(s/r) − g(s))/s. Or
   g′(x) + 1/2 = e^x (sinh x − x)/(e^x − 1)² > 0 et
   g′(x) = −(x e^x − e^x + 1)/(e^x − 1)² < 0 pour x > 0 (le numérateur s'annule en 0 et a pour
   dérivée x e^x > 0) ; les deux identités sont vérifiées par sympy. Donc
   0 < g(s/r) − g(s) = −∫_{s/r}^{s} g′ < (s − s/r)/2, d'où κ_mode ∈ ]0 ; 1[. Le même argument
   (avec r → ∞, g(s/r) → 1) donne l'avance dans ]0 ; 1/2[. La décroissance de l'avance en s est
   seulement observée sur une grille (voir Vérifications) et ne fait pas partie de l'énoncé.

## Vérifications

- Symbolique (`t1_check.py`, `symbolic`) : les identités des points 1, 2, 3 et 5 et la série de φ.
- Numérique (`numeric`), sur le vrai `DiagonalSSMLayer` en float64, paramètres tirés au hasard
  (log_A_real ∈ [ln 0,05 ; ln 3], ω ∈ [−3 ; 3], log_dt ∈ [ln 0,02 ; ln 1,5], C et D gaussiens),
  noyaux de 60 000 échantillons :
  - 96 modes isolés et 24 canaux complets par discrétisation (6 tirages × 4 canaux × 4 modes) : gain, avance prédite (ZOH) et
    avance nulle (FOH) ; écart maximal 1,74·10⁻¹⁰ ;
  - modes réels, r = 2 et 4 : κ_mode mesuré par les premiers moments des noyaux avant et après
    Δ → Δ/r, égal à la formule (écart 4,5·10⁻¹¹), et dans ]0 ; 1[ pour chaque mode ;
  - avance des modes réels dans ]0 ; 1/2[ sur s ∈ [10⁻⁴ ; 10²], et décroissante sur cette grille
    (observation, hors énoncé).
  Le contrôle compare des valeurs signées, pas des normes : une erreur de signe ou de facteur
  dans φ le ferait échouer (c'est arrivé au premier essai, sur une troncature du noyau).

## Portée

- **Fréquence nulle seulement.** Les énoncés portent sur le premier moment du noyau, c'est-à-dire
  le retard de groupe en ω = 0. Les retards par bande d'octave mesurés par
  `demo/sr_transfer/kappa.py` (500 Hz à 8 kHz) n'en découlent pas directement.
- **Une couche linéaire.** Rien ici ne porte sur le réseau entier (portes, non-linéarités,
  cascade) : c'est l'objet de T4.
- Les retards (rapport premier moment / gain) supposent un gain non nul : G ≠ 0 pour un canal,
  Re c ≠ 0 pour un mode réel isolé. Contre-exemple sinon (relecture Codex) : λ = −1, Δ = 1,
  c = i donne un noyau identiquement nul. Près d'une annulation, le rapport est instable.
- **Code flottant.** Les identités exactes portent sur l'objet idéal. Le code s'en écarte par
  `phi2` (polynôme tronqué pour |z| < 10⁻⁴), `pole` (plancher Re z ≥ −80), `foh_late` (limite
  Δ/z² sous ce plancher) et l'arrondi : pour |z| très petit (λ = −1, Δ = 10⁻⁸ en float32,
  Δ = 10⁻²⁰ en float64), a s'arrondit à 1, le noyau ZOH devient nul et le noyau FOH a une queue
  constante, sans moments convergents (observé par Codex sur le code réel). Le contrôle
  numérique couvre s ∈ [10⁻³ ; 4,5] environ, pas ces bords.
- Le commentaire d'en-tête de `ssm_wavenet.py` décrit une sortie issue de l'état précédent et
  sans facteur 2 ; les fonctions exécutées (`_kernel`, `_foh_kernel`, `_foh_step`) suivent la
  table ci-dessus. Écart signalé, code et commentaire non modifiés.
- Le point 5 ne vaut que pour les modes réels : pour un mode complexe, Re(w φ(z))/Re(w) n'est pas
  borné (Re w peut s'annuler), et les modes entraînés sont surtout complexes.
- Les noyaux décalés par les boutons (`_shifted_kernel`), la parité avec `forward` et
  l'inférence récurrente ne sont pas contrôlés ici.
- Le code calcule en complex64 en usage normal ; le contrôle est en float64.

## Rétrodictions et prédictions

- **Rétrodiction.** Le « demi-échantillon » de `demo/sr_transfer/FIX_foh.md` et
  `RESULTS_hold.md` est la limite s → 0 du point 3 ; le point 5 le précise pour la contribution
  normalisée d'un mode réel isolé : un mode lent porte une demi-jauge entière, un mode rapide
  presque rien ; cela ne dit pas quels modes dominent dans le réseau.
- **Rétrodiction.** Les modèles entraînés en FOH (`paired_ssm_wavenet_foh_s{925,736,688}`) ont un
  ESR de 2,8·10⁻⁴ à 4,0·10⁻⁴ face à leur propre sortie à 48 kHz après rééchelonnement à 88,2, 96
  et 192 kHz (`demo/sr_transfer/sr_transfer_paired_ssm_wavenet_foh_s925_last_plain.json`), contre
  0,036 à 0,077 pour le ZOH sans correction : compatible avec un retard de transfert nul.
- **Prédiction T1-P1**, mesurée ; résultat et portée dans `THEORY.md`, T1-P1.
