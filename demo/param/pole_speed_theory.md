# Vitesse des modes, gain et dérive : pourquoi un pôle s'annule, et comment l'empêcher

Note théorique, voie produit, écrite le 2026-09-28 après le NaN de `g2_poles_additive_s688`
(`predictions_poles.md`, « Écart de G2.2 »). Mesures : `pole_speed.py`, `pole_speed.json`.
**Réécrite après une relecture Codex en lecture seule, qui a corrigé les points 3 à 7 de la
première version** : classe de modèles sous conditionnement, argument sur Adam, seuils,
plages, portée du futur bras.

## 1. Un mode ZOH et ses régimes

Chaque mode du SSM diagonal a un pôle continu λ = −e^α + iω (α = `log_A_real`, ω = `A_imag`) et
un pas dt = e^δ (δ = `log_dt`). On note :

- z = λ·dt ;
- **s = −Re(z) = e^α·dt**, la décroissance de l'enveloppe par échantillon ;
- θ = ω·dt, la rotation par échantillon.

Le mode discrétisé en ZOH s'écrit :

- pôle discret : a = e^z = e^{−s}·e^{iθ} ;
- entrée : h[n] = a·h[n−1] + b·x[n], avec b = (a − 1)/λ ;
- noyau : K[k] = 2·Re(C·b·a^k), d'état initial nul ;
- gain continu : Σ K[k] = 2·Re(−C/λ).

Les régimes :

- **s ≪ 1** : enveloppe à mémoire longue (environ 1/s échantillons). On n'a a ≈ 1 − s et b ≈ dt
  que si, en plus, |θ| ≪ 1 : un mode peut être lentement amorti et fortement oscillant.
- **s ≈ 1** : mémoire d'environ un échantillon.
- **s ≫ 1** : **sans mémoire**. |a| = e^{−s} ≈ 0, b ≈ −1/λ, K[0] ≈ 2·Re(−C/λ), et les termes
  retardés décroissent en e^{−k·s}. Le mode agit comme un gain instantané. À s = 10, l'état ne garde
  que 4,5·10⁻⁵ de lui-même par échantillon.

## 2. Sans mémoire, la vitesse agit sur un gain

Dans le régime s ≫ 1, on distingue l'enveloppe du gain, g = |C|/|λ|, et le gain continu réel,
G = 2·Re(−C/λ), qui peut s'annuler ou changer de signe selon la phase.

- ∂ ln g / ∂α = −e^{2α}/(e^{2α} + ω²) → −1 quand e^α ≫ |ω| : **α agit asymptotiquement comme un
  logarithme de gain**.
- La sensibilité à δ disparaît asymptotiquement : b ≈ −1/λ ne dépend plus de dt, et c'est α,
  plus que δ, qui module le gain.
- **Redondance asymptotique** : seul C/λ compte. Réduire l'apport du mode passe aussi bien par |C|
  que par α. Ce n'est pas une invariance exacte à vitesse finie.

## 3. Ce qui peut entretenir la dérive

- **Le gradient ne s'éteint pas vite.** La dérivée du tap instantané par rapport à α est de l'ordre
  du gain, qui ne décroît qu'en e^{−α}, soit en 1/s. Mais le gradient que reçoit l'optimiseur,
  ∂L/∂α, dépend aussi des autres taps, des activations et du gradient amont. Il n'est pas mesuré
  ici.
- **Adam.** Ses pas valent m̂/(√v̂ + ε), rapportés à l'histoire récente du gradient (β₂ = 0,999,
  environ 1 000 pas). **Une dérive à pas de l'ordre du taux d'apprentissage est possible si le
  signe du gradient reste constant et que sa taille ne chute pas brutalement par rapport à son
  histoire**, mais ce n'est pas automatique. Après de grands gradients, un petit gradient de même
  signe donne des pas minuscules : Codex calcule 3·10⁻⁵ fois le taux d'apprentissage pour 100 pas à
  10⁻⁶ après 1 000 pas à 1. La dynamique réelle du mode fautif n'est pas mesurée.
- **Pas de force de rappel.** Avec `--honor-optim`, α, ω et δ n'ont pas de weight decay.
- **Ordre de grandeur.** Pour le mode fautif de D à 688, au pas 9 300, |C/λ| vaut de 1,5 à
  1,75·10⁻⁴ et le gain réel G de 1,2 à 1,4·10⁻⁴ (vérifié par Codex). C'est un gain ; ce n'est pas le
  gradient de la perte.
- **Seuils float32** : exp(−s) devient dénormalisé au-delà de s = 87,34, et **s'arrondit à zéro
  au-delà de s ≈ 103,97** (150·ln 2 ; 103,28 = 149·ln 2 correspond au plus petit dénormalisé). La
  valeur exacte dépend du traitement des dénormalisés par le matériel.

## 4. Observé seulement dans les bras « pôles »

Mesures (`pole_speed.json`, 28 réglages d'entraînement du Marshall) :

- **bras FiLM (4 runs)** : aucun mode au-delà de s = 10 ;
- **bras « pôles » (4 runs)** : de 1 à 9 modes au-delà de s = 10 pour au moins un réglage, et de 0
  à 4 modes « commutés », c'est-à-dire s < 1 pour un réglage et s > 10 pour un autre.

**Portée.** Ces comptes décrivent des **changements de régime**, pas leur effet sur la sortie ou
sur l'ESR : un mode peut franchir ces seuils en restant négligeable, masqué ou compensé. « Commuté »
est le nom d'un critère numérique, pas la preuve qu'une contribution utile s'allume ou s'éteint. Et
« observé seulement dans ces quatre runs » ne prouve pas que le conditionnement soit nécessaire.

**Mécanisme plausible, non démontré.** La tête décale α et δ chacun dans [−2,3 ; 2,3]. s varie
donc jusqu'à un facteur e^{4,6} ≈ 100 autour de sa valeur de base, et **jusqu'à e^{9,2} ≈ 9 900 entre
deux réglages** (borne théorique ; le réglage central n'est pas contraint à un décalage nul). Un
mode peut ainsi entrer dans le régime sans mémoire pour certains réglages, où le décalage de α
module un gain. La tête dispose alors d'un moyen de régler un gain par réglage, en plus de déplacer
des dynamiques.

## 5. Remèdes

### 5.1 Garde numérique (faite, `2d49ac7`)

s est bornée à 80 avant l'exponentielle. Le NaN disparaît sans changer un bit au-dessus du seuil.
Au-delà, le gradient ne passe plus par le module de a, mais il passe encore par le dénominateur λ
de b. **Elle ne traite pas la cause.**

### 5.2 Modes normalisés en gain continu

On remplace le poids d'entrée b = (a − 1)/λ par **b' = 1 − a**, c'est-à-dire qu'on prend comme
paramètre C' = −C/λ :

- noyau K[k] = 2·Re(C'·(1 − a)·a^k), gain continu 2·Re(C'), indépendant de λ et de dt ;
- **à réglage fixe**, C ↔ C' est une bijection (λ ≠ 0) : même fonction, autre géométrie
  d'optimisation ;
- **sous conditionnement des pôles, ce n'est plus la même classe de modèles.** C est partagé entre
  réglages alors que λ en dépend : l'équivalence exigerait C'(u) = −C/λ(u). Avec un C' fixe, **le
  gain continu du mode devient indépendant des boutons**. C'est un changement fonctionnel du
  conditionnement, qui retire précisément le moyen de régler un gain par la vitesse ;
- **gradient résiduel** : à C', δ et ω fixés, ∂K[0]/∂α = 2·s·Re(C'·a), soit au plus 2·|C'|·s·e^{−s}
  en module. C'est une borne sur un seul tap, pas sur ∂L/∂α. Elle passe sous 10⁻⁸ vers s ≈ 12 à 22
  selon |C'| (de 10⁻⁴ à 1). **Elle rend la dérive beaucoup moins probable, sans la garantir** ; ε
  n'est pas un seuil d'arrêt ;
- **changement de fréquence** : rééchelonner dt reste cohérent, puisque c'est un ZOH exact avec un
  coefficient d'entrée continu B' = −λ. Mais la relecture FOH demande d'adapter aussi les poids,
  β'_j = −λ·β_j. Et `zoh_to_foh` refuse déjà les pôles conditionnés. La compatibilité n'est pas
  acquise sans adaptation ;
- **ce qu'elle n'interdit pas** : un mode peut encore avoir s < 1 pour un réglage et s > 10 pour un
  autre. Elle fixe le gain continu, pas la réponse à toutes les fréquences, ni l'apport du mode
  après les non-linéarités.

### 5.3 Borne lisse de la vitesse

s = ρ·tanh(e^{α+δ}/ρ), avec ρ ≈ 10. Elle empêche les très grandes vitesses, **pas le comportement
quasi instantané** : près de ρ = 10, le mode ne garde que 4,5·10⁻⁵ de son état par échantillon. Elle
ne rend pas le gain conditionnel à D, qui est partagé entre réglages. Recalculée après un changement
de fréquence, elle ne commute pas avec dt ↦ dt/4.

### 5.4 Rappel doux

Une pénalité λ_r·Σ softplus(s − ρ)² donne bien une force de rappel : sa dérivée par rapport à α
vaut 2·λ_r·s·softplus(s − ρ)·σ(s − ρ) > 0. Mais son poids λ_r est à fixer, elle ne garantit aucune
borne stricte, et elle garde la redondance de la section 2.

## 6. Un futur bras : ce qu'il pourrait et ne pourrait pas dire (sans plan ici)

Un bras D avec des modes normalisés (5.2) **ne départagerait pas à lui seul** « déplacer des
coupures » et « régler des gains ».

- Un ESR dégradé pourrait venir de la capacité conditionnelle retirée, du conditionnement numérique
  ou de l'apprentissage.
- Un ESR inchangé montrerait seulement que la modulation du gain continu par la vitesse n'était
  pas nécessaire à cette performance.

Une comparaison mieux posée comprendrait **deux bras** :

- modes normalisés **plus une tête explicite de gain par mode, fonction des boutons**, qui rend la
  capacité retirée sans passer par la vitesse des pôles ;
- modes normalisés sans cette tête.

À préinscrire, avec les comptes de `pole_speed.py` et une mesure de l'apport de chaque mode à la
sortie, et pas seulement de son régime.

## 7. Implications pratiques (non mesurées ici)

- **Moteur C++.** Un coefficient a dénormalisé n'implique pas un état dénormalisé : l'entrée
  maintient l'état à une amplitude normale, et le moteur natif ajoute l'entrée à chaque récurrence.
  Des opérandes dénormalisés peuvent ralentir certaines opérations selon le processeur. Activer
  flush-to-zero et denormals-are-zero est la pratique documentée. La normalisation 5.2 ne les
  remplace pas.
- **192 kHz.** s est divisée par 4. Un mode à s = 99 reste quasi instantané (24,75), mais un mode à
  s = 11 passe à 2,75 (|a| ≈ 0,064) : tous les modes sans mémoire à 48 kHz ne le restent pas. L'effet
  est négligeable pour l'exemple extrême en ZOH ; en FOH, le terme d'entrée retardée peut rester
  d'ordre 1/s par rapport au gain.
- **Interprétation de G2** : voir la lecture à vérifier au verdict dans `predictions_poles.md`.
