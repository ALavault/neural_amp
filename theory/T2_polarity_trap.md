# T2 — Le piège de polarité de la perte L1 + MR-STFT, le long du gain de sortie

Preuve de l'entrée T2 de `THEORY.md` (statut : voir le registre). Contrôle :
`theory/t2_check.py` (symbolique, et numérique sur la vraie perte et les vraies cibles).

## Objet et correspondance

La perte d'entraînement du banc, `scripts/product_nablafx_bench.py` (`main`, `FlexibleLoss`) :
L(ŷ, y) = w₁ L1(ŷ, y) + w₂ MRSTFT(ŷ, y), avec

- `l1_loss` de nablafx : moyenne de |ŷ − y| sur tous les échantillons du lot ;
- `mrstft_loss` de nablafx, qui appelle `auraloss.freq.MultiResolutionSTFTLoss` avec ses valeurs
  par défaut (`third_party/auraloss/auraloss/freq.py`) : trois résolutions (FFT 1024, 2048, 512 ;
  pas 120, 240, 50 ; fenêtres de Hann 600, 1200, 240), moyenne sur les résolutions de
  SC + log-magnitude ;
  - magnitude |X| = √max(|STFT|², η), η = 10⁻⁸ (on note η le plancher et ε la perturbation du gain) ;
  - SC = ‖|Y| − |X|‖_F / ‖|Y|‖_F, normes sur tout le lot ;
  - log-magnitude = moyenne de |log|X| − log|Y|| sur toutes les cases du lot.

Poids : (1 ; 0,1) pour le protocole Big Muff publié (`PUBLISHED`, S4-TF et défaut), (0,5 ; 0,5)
pour `s4-param`.

Chemin étudié : ŷ = a·y, a = −1 + ε, dans l'espace des sorties. Si la sortie du modèle est
affine en les paramètres de sa dernière couche (activation finale `none`), multiplier ses poids et
son biais par |a| parcourt exactement ce chemin à partir d'une solution ŷ = −y. Les deux
processeurs du banc finissent par un tanh impair (`output_act="tanh"` par défaut,
`output_net[4]` ; `PolarityGuard`) : la négation est exacte, le changement d'échelle ne l'est
pas (voir Portée).

## Énoncé

Soit, pour chaque résolution r, f_r la fraction des cases STFT de la cible de puissance ≥ η et
ρ_r = ‖|Y| sur ces cases‖_F / ‖max(|Y|, √η)‖_F, et m = moyenne_r (ρ_r + f_r) ∈ [0 ; 2]. Alors

L(−1 + ε) − L(−1) = −w₁ E|y| ε + w₂ m |ε| + O(ε²).

On suppose w₁, w₂ > 0, y ≠ 0 et (H2) qu'aucune puissance STFT de la cible n'est exactement
égale à η. La pente à droite vaut w₂ m − w₁ E|y| et la pente à gauche −(w₂ m + w₁ E|y|) < 0.
Donc :

- si **w₁ E|y| < w₂ m**, a = −1 est un minimum local strict du chemin de gain ;
- si **w₁ E|y| > w₂ m**, ce n'en est pas un : réduire le gain en valeur absolue fait descendre
  la perte ;
- à l'**égalité**, c'en est encore un minimum strict dès qu'une case est au-dessus du plancher
  (f̄ > 0) : les termes du premier ordre s'annulent et le terme logarithmique laisse
  w₂ f̄ (−log(1 − ε) − ε) ≈ w₂ f̄ ε²/2 > 0 à droite (contre-exemple de Codex au « si et
  seulement si » strict de la première version).

Conséquence, pour une activation finale identité : une solution exactement inversée n'est
**pas** un minimum local du modèle complet dès que w₁ E|y| > w₂ m (direction de descente sur la
dernière couche). La réciproque n'est pas établie : un minimum le long du gain n'en est pas un
dans toutes les directions.

## Preuve

- **L1.** |a y − y| = |a − 1| |y| ; pour a = −1 + ε, |a − 1| = 2 − ε (|ε| < 2). Donc
  L1 = (2 − ε) E|y| : pente −E|y| des deux côtés.
- **MR-STFT.** La STFT est linéaire : STFT(a y) = a STFT(y), et la magnitude ne voit que
  g = |a| = 1 − ε pour |ε| < 1. Pour une case de puissance P :
  - P ≥ η/g² et P ≥ η (case gardée des deux côtés) : |X| − |Y| = (g − 1)|Y| et
    log|X| − log|Y| = log g ;
  - P < min(η, η/g²) : les deux sont au plancher, contribution nulle ;
  - le lot est fini : si aucune puissance n'égale exactement le plancher (H2), aucune case ne
    franchit le seuil pour |ε| assez petit, et les deux cas précédents sont les seuls.

  D'où, exactement dans ce voisinage, SC_r = |g − 1| ρ_r et log-magnitude_r = |log g| f_r,
  avec |log g| = |ε| + O(ε²). (Une première version invoquait une densité bornée des
  puissances près du plancher ; Codex a montré que ce n'est pas suffisant pour la norme de
  Frobenius, en |ε|^{3/2} dans une idéalisation continue. L'hypothèse finie la remplace.)
- **Somme.** L(−1 + ε) − L(−1) = −w₁ E|y| ε + w₂ m |ε| + O(ε²) ; les deux pentes s'en déduisent
  (limites unilatérales calculées par sympy dans `t2_check.py`).
- **Modèle complet, activation finale identité.** (W, b) → ((1 − ε)W, (1 − ε)b) multiplie la
  sortie par 1 − ε. Si w₁ E|y| > w₂ m, la dérivée à droite le long de cette direction est négative
  en ŷ = −y, qui n'est donc pas un minimum local.

## Hypothèses

- H1 [mathématique] ŷ = −y exactement. Une solution réelle ŷ = −y + e lisse le pli de la MR-STFT
  en a = −1 sur une échelle fixée par e ; l'énoncé ne dit rien en dessous de cette échelle.
- H2 [mathématique pour l'énoncé ; contrôlée sur les cibles réelles par `t2_check.py`] aucune
  puissance STFT de la cible n'est exactement égale à η = 10⁻⁸. Le voisinage où l'énoncé est
  exact a pour taille la distance relative des puissances au plancher ; le script rapporte la
  fraction des cases qui le franchissent pour un gain dans [1 − h ; 1 + h].
- H3 [code] perte du banc telle que décrite (auraloss par défaut, η = 10⁻⁸, réduction « mean ») ;
  la pondération A-weighting, l'invariance d'échelle ou un terme de phase (désactivés) changent
  l'énoncé.

## Vérifications

- Symbolique : pentes à droite et à gauche de la perte modèle, par `sympy.limit`.
- Numérique : vraies cibles de test Big Muff (12 segments de 240 000 échantillons,
  E|y| = 0,01494), vraie `FlexibleLoss` du banc en float64 (en entraînement : float32).
  Résultats de `theory/t2_check.py` :

  | poids | échelle c de la cible | pente droite mesurée (h = 10⁻⁴) | prédite | piège |
  |---|---|---|---|---|
  | 1 / 0,1 | 1 | +0,17811 | +0,17811 | oui |
  | 1 / 0,1 | 10,33 (0,8 c₀) | +0,04541 | +0,04540 | oui |
  | 1 / 0,1 | 16,15 (1,25 c₀) | −0,04138 | −0,04138 | non |
  | 0,5 / 0,5 | 1 | +0,95780 | +0,95778 | oui |
  | 0,5 / 0,5 | 103,34 (0,8 c₀) | +0,22781 | +0,22778 | oui |
  | 0,5 / 0,5 | 161,47 (1,25 c₀) | −0,20655 | −0,20657 | non |

  m = 1,9305 à l'échelle 1 ; échelle de référence c₀ = w₂ m(y)/(w₁ E|y|) = 12,92 (1 / 0,1) et
  129,18 (0,5 / 0,5). Ce n'est pas le point exact de bascule : m dépend de l'échelle de la
  cible, puisque le plancher est fixe (m = 1,9984 à c = 10,33 ; 1,9993 à 16,15). Les essais à 0,8 c₀ et 1,25 c₀
  encadrent la bascule sans la localiser. Pour une cible fixe, le critère sur le rapport des
  poids, w₁/w₂ > m/E|y|, n'a pas cette dépendance.
  Pentes à gauche et à droite : écarts ≤ 2,5·10⁻⁴. Cases franchissant le plancher pour un gain
  dans [1 − 10⁻³ ; 1 + 10⁻³] : 1,55·10⁻⁴ à l'échelle 1, ≤ 5,7·10⁻⁶ aux autres échelles ; aucune
  exactement au plancher.
- **Cas d'égalité** (w₂ = 0,1, w₁ = w₂ m/E|y|, f̄ = 0,9305) : incrément à droite 4,306·10⁻⁶
  (prédit w₂ f̄ (−log(1 − h) − h) = 4,684·10⁻⁶) à h = 10⁻², 4,288·10⁻⁸ (prédit 4,656·10⁻⁸) à
  h = 10⁻³ ; à gauche 3,861·10⁻⁴ (prédit 2 w₂ m h = 3,861·10⁻⁴) à h = 10⁻³. L'incrément droit est
  positif comme prévu. L'écart de 8 % au coefficient prédit n'est pas isolé ; une explication
  possible, non vérifiée, est l'effet à l'ordre 2 des cases qui franchissent le plancher pour
  h = 10⁻³ ou 10⁻², hors du voisinage exact de la preuve.
- Le journal de la dernière exécution est `/fastdata/lavaulta/tmp/t2_check.log` (hors dépôt) ;
  les chiffres ci-dessus en sont recopiés. Le contrôle échoue si une pente s'écarte de plus de 2 %
  de w₂ m + w₁ E|y|, ou si le signe de la pente à droite diffère de la prédiction (aux deux pas) ; il encadre la
  bascule dans les deux pondérations.

## Non-vacuité

La condition n'est ni toujours vraie ni toujours fausse : sur les cibles Big Muff réelles, la
marge vaut ×12,9 en 1 / 0,1 et ×129 en 0,5 / 0,5, et un changement de niveau de la cible (ou du
rapport w₁/w₂) d'un facteur connu la renverse, comme le montre le tableau.

## Portée

- **Tête tanh (cas du banc).** Avec ŷ = tanh(u), réduire la dernière couche de (1 − ε) déplace
  la sortie de −(1 − ŷ²) artanh(ŷ) ε = −ŷ (1 − 2ŷ²/3 + O(ŷ⁴)) ε au lieu de −ŷ ε. L'écart relatif
  est de l'ordre de (2/3) ŷ² ; sur la cible Big Muff, E[y²|y|]/E|y| = 0,0413 (crête 0,4265,
  calculés par `t2_check.py`), soit environ 3 % pondéré par |y|. Approximation non démontrée, non contrôlée : le corollaire sur le
  modèle complet n'est pas établi pour la tête tanh.
- Une direction (le gain de sortie), une solution exacte, un lot de cibles. Ni la dynamique de
  l'optimiseur, ni la probabilité de tomber dans le piège.
- E|y| et m dépendent du lot ; la moyenne sur les lots d'entraînement n'est pas calculée ici.
- Le poids relatif w₂/w₁ n'est donc pas sans dimension : son effet sur ce piège dépend du niveau
  de la cible.
