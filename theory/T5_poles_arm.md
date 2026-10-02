# T5 — Ce que conditionne le bras « pôles »

Preuve de l'entrée T5 de `THEORY.md` (le statut est dans le registre). Contrôle :
`theory/t5_check.py` (symbolique et numérique sur le code réel).

## Objet et correspondance

`src/fssr_nam/models/ssm_wavenet.py` @ 2d49ac7, `SSMWaveNet(num_controls > 0, cond="poles",
discretization ∈ {"zoh", "foh"})`. Notations : bloc i, canal c, mode m, réglage u.

| ici | code |
|---|---|
| u | `controls` (batch, num_controls) |
| (δ_{i,c}(u), ε_{i,c,m}(u)) = 2,3 · tanh(g(u)) | `pole_shifts` : `POLE_SCALE * tanh(pole_net(controls))` |
| Δ_{i,c}(u) = e^{ℓ_{i,c} + δ_{i,c}(u)} | `_shifted_inputs` : `exp(log_dt + dlog_dt)` |
| λ_{i,c,m}(u) = −e^{α_{i,c,m} + ε_{i,c,m}(u)} + i ω_{i,c,m} | `_shifted_inputs` : `complex(-exp(log_A_real + dlog_a), A_imag)` |
| a = e^{λΔ} ; ZOH : b = (a − 1)/λ ; FOH : β₁, β₀ | `pole`, `_shifted_inputs` |
| K_{i,c}[k] (noyau du canal) | `_shifted_kernel` |
| D_{i,c}, c_{i,c,m} | `D`, `_complex(C)`, non décalés |
| sortie du canal : K ∗ x + D x | `DiagonalSSMLayer.forward` |

## Énoncé

Énoncé sur le modèle idéal : arithmétique exacte, noyaux infinis, formules fermées de la table.
Le code flottant n'est contrôlé que dans le domaine de `t5_check.py` (voir Portée).

1. **Voie unique.** La sortie du réseau dépend de u seulement à travers les familles
   (Δ_{i,c}(u))_{i,c} et (Re λ_{i,c,m}(u))_{i,c,m}. Deux réglages qui donnent les mêmes
   familles donnent la même sortie, pour toute entrée. Ni ω (`A_imag`), ni c, ni D, ni aucun
   paramètre des couches 1×1, des portes ou de la tête de sortie ne dépendent de u.
2. **Gain continu d'un canal.** Σ_k K_{i,c}[k] + D_{i,c} = D_{i,c} + Σ_m 2 Re(−c_{i,c,m}/λ_{i,c,m}(u)),
   en ZOH comme en FOH. Ce gain ne dépend pas de Δ_{i,c}(u).
3. **Régime permanent du réseau.** Pour une entrée constante x₀, la sortie converge vers une
   limite F(x₀, u). Cette limite dépend de u seulement à travers les (Re λ_{i,c,m}(u)). En
   particulier, un réglage qui ne déplace que les Δ (ε ≡ 0) laisse F inchangé.
4. **Résidu lié au pôle.** Le résidu continu de chaque mode, w_{i,c,m}(u) = −c_{i,c,m}/λ_{i,c,m}(u),
   vérifie w λ = −c pour tout u. Pour un mode réel (ω = 0), déplacer le pôle d'un facteur e^{ε}
   multiplie la contribution du mode au gain continu par e^{−ε}. Le bras ne peut donc déplacer
   la coupure d'un mode réel sans changer, en sens inverse, son gain continu.

## Preuve

**1.** Lecture du code @ 2d49ac7.
- Avec `cond="poles"`, le constructeur ne remplit que `pole_net`. `film_net` reste `None` : la
  branche `if cond == "film"` est la seule qui l'affecte.
- `SSMWaveNet.forward` ne lit `controls` qu'à deux endroits : `self.film_net(controls)`, gardé
  par `film_net is not None`, et `self.pole_shifts(controls)`. Les `films` restent donc `None` et
  `GatedSSMBlock.forward` saute la modulation (`if film is not None`).
- `shift` n'entre que dans `self.ssm` (`DiagonalSSMLayer.forward`), puis dans `_shifted_kernel`,
  puis dans `_shifted_inputs`. Ce dernier ne lit `shift` que sous la forme `log_A_real + dlog_a`
  et `log_dt + dlog_dt`. `A_imag` est seulement étendu (`expand_as`). `C` est lu dans
  `_shifted_kernel` sans décalage, et `D` dans `forward`.
- Le noyau est donc une fonction de (e^{α+ε}, ω, e^{ℓ+δ}, c), c'est-à-dire de (Re λ(u), Δ(u)) à
  paramètres fixés.
- Les autres opérations (`input_conv`, `pre_gate`, `tanh`·`sigmoid`, `res_conv`, `skip_conv`,
  `output_net`) n'ont pas d'argument dépendant de u. La sortie est leur composition. ∎

**2.** À u fixé, le noyau du canal est celui d'une couche sans décalage de paramètres
(α + ε, ω, ℓ + δ, c). La table de correspondance le montre : `_shifted_inputs` calcule
a, b, β₁ et β₀ par les mêmes formules que `_a`, `_b` et `_foh_inputs`. T1 (énoncé 1) donne
Σ_k 2 Re(c b a^k) = 2 Re(−c/λ) en ZOH, et la même somme en FOH, puisque β₁ + β₀ = (a − 1)/λ.
L'expression ne contient pas Δ. On somme sur les modes et on ajoute D. ∎

**3.** Par récurrence sur les blocs.
- Pour une entrée constante, `input_conv` (1×1) produit un signal constant.
- Si l'entrée d'un canal SSM converge vers h∞, sa sortie converge vers G_{i,c}(u) h∞, G étant
  le gain de l'énoncé 2. En effet, le noyau est absolument sommable (|a| < 1, puisque
  Re λ < 0 et Δ > 0). Écrivons x[n] = h∞ + e[n] avec e → 0. Alors (K ∗ e)[n] → 0 par convergence
  dominée, et (K ∗ h∞)[n] → (Σ K) h∞.
- Les couches 1×1, la porte tanh·sigmoid et la tête de sortie sont sans mémoire et continues.
  Elles préservent la convergence, et leur limite est leur valeur en la limite. Les sommes
  résiduelle et `skip_sum` la préservent aussi.
- La limite F est donc une composition de fonctions continues fixes et des gains G_{i,c}(u).
  Par l'énoncé 2, ces gains ne dépendent de u qu'à travers Re λ(u). ∎

**4.** On a w λ = −c par définition. Si ω = 0, λ = −e^{α+ε} et
2 Re(w) = 2 Re(c) e^{−(α+ε)}. ∎

## Hypothèses et portée

- H1 `cond="poles"`, discrétisation ZOH ou FOH [mathématique ; le constructeur refuse `free`].
- H2 noyaux infinis [mathématique ; le code tronque à la longueur du segment]. Les énoncés 2
  et 3 ne valent que lorsque les modes sont éteints dans la longueur du segment. Un mode
  lent (|Re z| petit) peut ne pas l'être, et le régime permanent n'est alors pas atteint.
- H3 formules fermées exactes [mathématique ; le code s'en écarte par `pole`, qui plafonne
  Re z ≥ −80. En ZOH, b est calculé avec le même a plafonné, et l'énoncé 2 reste exact. En FOH,
  β₁ et β₀ viennent du z exact : l'écart est d'ordre e^{−80}. Le code calcule aussi `phi2` en
  série et arrondit en float32 et complex64.]
- Portée :
  - L'énoncé 1 dit par quoi passe u, pas ce qui est appris.
  - Les énoncés 2 et 3 portent sur la fréquence nulle seulement. Aux autres fréquences, Δ
    change bien la réponse du canal : il en dilate l'axe des fréquences, à l'erreur de
    discrétisation près.
  - En temps discret, le résidu effectif c·b(λ, Δ) et le premier échantillon K[0] varient avec
    δ et ε. L'énoncé ne contredit donc pas la mesure de `D + K[0]` dans
    `demo/param/predictions_fast_modes.md`.
  - L'énoncé 3 porte sur un régime permanent de signal constant, pas sur l'audio. Les signaux
    du banc n'ont pas de composante continue.

## Correspondance avec la mesure G2.2

Le verdict G2.2 (`demo/param/predictions_poles.md`) compare le bras D (pôles) au bras A (FiLM).
L'énoncé 1 dit que le bras D ne teste pas seulement « déplacer des coupures ». Il couple aussi les
gains continus des modes à leurs pôles (énoncé 4). Il ne peut changer aucun gain continu de canal
par Δ seul (énoncé 2). Tout effet de gain réglable à basse fréquence passe donc par ε, ou par les
non-linéarités en aval, qui reçoivent des signaux dépendant de u.
