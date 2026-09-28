# Vitesse des modes, gain et dérive : pourquoi un pôle s'annule, et comment l'empêcher

Note théorique, voie produit, écrite le 2026-09-28 après le NaN de `g2_poles_additive_s688`
(`predictions_poles.md`, « Écart de G2.2 »). Mesures : `pole_speed.py`, `pole_speed.json`.

## 1. Un mode ZOH et ses trois régimes

Chaque mode du SSM diagonal a un pôle continu λ = −e^α + iω (α = `log_A_real`, ω = `A_imag`), un pas
dt = e^δ (δ = `log_dt`). On note z = λ·dt et **s = −Re(z) = e^α·dt**, la décroissance par
échantillon. Le mode discrétisé en ZOH s'écrit :

- pôle discret : a = e^z ;
- entrée : h[n] = a·h[n−1] + b·x[n], avec b = (a − 1)/λ ;
- sortie : 2·Re(C·h[n]) ;
- noyau : K[k] = 2·Re(C·b·a^k), gain continu (DC) : 2·Re(−C/λ).

Trois régimes, selon s :

- **s ≪ 1 : mémoire longue.** a ≈ 1 − s, b ≈ dt : le mode intègre sur environ 1/s échantillons.
- **s ≈ 1 : mémoire d'environ un échantillon.**
- **s ≫ 1 : sans mémoire.** a = e^−s ≈ 0 et b ≈ −1/λ. Le noyau se réduit à
  K[0] ≈ 2·Re(−C/λ), et K[k ≥ 1] = O(e^{−k·s}). Le mode est un **gain instantané**, exactement
  comme le terme direct D. Au-delà de s ≈ 10 (e^−10 ≈ 4,5·10⁻⁵ par échantillon), il n'a plus
  aucune dynamique à 48 kHz.

## 2. Sans mémoire, la vitesse devient un gain

Dans le régime s ≫ 1, et si |ω| ≪ e^α, le gain du mode vaut g ≈ |C|/|λ| ≈ |C|·e^{−α}. Donc :

- **∂ ln g / ∂α = −1** : α n'est plus un paramètre de dynamique, c'est un **logarithme de gain** ;
- **∂g / ∂δ ≈ 0** : dt a disparu de la sortie, puisque b ≈ −1/λ ne dépend plus de dt ;
- **(C, λ) est redondant** : seul C/λ compte. Réduire l'apport du mode se fait aussi bien en
  diminuant |C| qu'en accélérant le pôle (α ↑), et rien dans la perte ne préfère l'un à l'autre.

## 3. Pourquoi l'optimisation dérive jusqu'au sous-dépassement

Trois ingrédients se combinent :

1. **Gradient algébrique.** Le gradient de la perte par rapport à α vaut
   |∂L/∂α| ≈ |∂L/∂K[0]|·g. Il décroît comme le gain, c'est-à-dire comme e^{−α}, donc seulement
   comme 1/s. Il ne s'annule pas vite.
2. **Invariance d'échelle d'Adam.** Adam fait des pas m̂/(√v̂ + ε) : tant que le gradient garde le
   même signe et dépasse ε = 10⁻⁸, **le pas reste de l'ordre du taux d'apprentissage, quelle que
   soit la taille du gradient**. Pour le mode fautif de D à 688 (|C| = 0,023), le gain vaut environ
   10⁻⁴, loin au-dessus de ε : la dérive continue à vitesse constante.
3. **Aucune force de rappel.** Avec `--honor-optim`, α, ω et δ n'ont pas de weight decay.

La dérive ne s'arrête donc qu'au sous-dépassement de a = e^{−s}, en float32 : 0 pour s > 103,3, et
valeurs dénormalisées dès s > 87,3. Le NaN de D à 688 et le mode à s = 101,7 du bras « pôles joint »
en sont l'aboutissement.

## 4. Pourquoi seulement avec le conditionnement des pôles

Mesures (`pole_speed.json`, 28 réglages d'entraînement du Marshall) :

- **Bras FiLM (4 runs) : aucun mode au-delà de s = 10.**
- **Bras « pôles » (4 runs)** : de 1 à 9 modes sans mémoire pour au moins un réglage, et de 0 à 4
  modes **commutés**, lents (s < 1) pour un réglage et sans mémoire (s > 10) pour un autre.

**Lecture.** La tête de conditionnement décale α et δ chacun jusqu'à ±2,3, ce qui multiplie s
jusqu'à e^{4,6} ≈ 100 d'un réglage à l'autre. Un mode ordinaire au centre peut donc devenir sans
mémoire pour un réglage extrême. Là, d'après la section 2, le décalage agit comme un **gain dépendant
du réglage**. La tête dispose alors d'un second mécanisme, qui n'est pas « déplacer une coupure
comme un correcteur » mais **allumer ou éteindre un mode selon les boutons**. Rien dans la perte ne
le décourage, et la section 3 explique qu'il peut dériver sans fin. Hors conditionnement, un mode
n'a aucun intérêt à devenir un gain, puisque D remplit ce rôle.

## 5. Remèdes

### 5.1 Garde numérique (faite, `2d49ac7`)

s est bornée à 80 avant l'exponentielle. Cela supprime le NaN sans changer un bit au-dessus du
seuil. **Cela ne traite pas la cause** : la dérive continue jusqu'au seuil, puis le gradient par a
est coupé.

### 5.2 Modes normalisés en gain continu (remède recommandé)

On remplace le poids d'entrée b = (a − 1)/λ par **b' = 1 − a**, c'est-à-dire qu'on prend comme
paramètre C' = −C/λ, le gain DC du mode :

- noyau : K[k] = 2·Re(C'·(1 − a)·a^k) ;
- gain DC : 2·Re(C'), indépendant de λ et de dt.

Propriétés :

- **Même classe de modèles.** C = −C'·λ est une bijection pour λ ≠ 0. Seule change la géométrie
  de l'optimisation : le gain appartient à C', la dynamique à λ.
- **Plus de redondance.** Sans mémoire, K[0] ≈ 2·Re(C') ne dépend plus de α. Le gradient restant
  vaut |∂K[0]/∂α| = 2·|C'|·|∂a/∂α| = 2·|C'|·s·e^{−s} : il s'annule en e^{−s}, non plus en 1/s. Il
  passe sous ε (10⁻⁸) vers s ≈ 20. **La dérive s'arrête d'elle-même bien avant le
  sous-dépassement** (s = 87 à 103), même avec Adam.
- **Changement de fréquence inchangé.** Il s'agit toujours du ZOH exact du même système continu :
  le rééchelonnement de Δ et la relecture FOH restent valables.
- **Conditionnement plus lisible.** La tête ne peut plus régler un gain à travers la vitesse d'un
  pôle, seulement une dynamique. « Les réglages déplacent les pôles » devient littéral.

### 5.3 Borne lisse de la vitesse

s = ρ·tanh(e^{α+δ}/ρ), avec ρ ≈ 10 : tout mode reste en deçà du régime sans mémoire. C'est simple,
mais cela change la classe de modèles (plus de gain instantané porté par un mode, D le reprend). Et
le gradient près de la saturation, en sech², reste amplifié par Adam : la borne tient, mais le
mode peut s'y coller.

### 5.4 Rappel doux

Une pénalité Σ softplus(s − ρ)² rend une force de rappel. C'est un hyperparamètre de plus, et
cela ne supprime pas la redondance de la section 2.

## 6. Prédictions pour un futur bras (à préinscrire avant tout run, sans plan ici)

Un bras D avec des modes normalisés en gain (5.2) départagerait les deux mécanismes de la section 4 :

- **ESR des réglages non vus inchangé** (à la marge de non-infériorité ln 1,15 de G2) : l'avantage
  de D tient au déplacement des pôles ;
- **ESR dégradé** : il tenait en partie à la commutation de gains, que 5.2 interdit ;
- **dans les deux cas, attendu** : aucun mode au-delà de s ≈ 20, et plus aucun mode commuté.

## 7. Implications pratiques

- **Moteur C++** : un pôle vers 10⁻⁴⁰ fait tourner l'état dans les nombres dénormalisés, que les
  processeurs traitent très lentement, d'où des pics de charge en temps réel. Il faut activer
  flush-to-zero et denormals-are-zero, ou appliquer 5.2.
- **Changement de fréquence** : à 192 kHz, s est divisée par 4 (99 → 25). Un mode sans mémoire le
  reste, et la conversion est sans effet sur lui.
- **Interprétation de G2** : voir la lecture à vérifier au verdict dans `predictions_poles.md`.
