# FIX_foh — discrétisation symétrique en entrée (maintien du premier ordre)

Voie produit (D-PRODUCT-001), aucune revendication scientifique. Écrit le 2026-09-24,
**avant tout changement de code** ; les sections 1 à 4 ne sont plus modifiées après lecture des
résultats. Diagnostic source : `demo/sr_transfer/RESULTS_hold.md`, qui tient lieu de fichier
DIAGNOSIS. Il fournit le prédicat, le mécanisme et la prédiction chiffrée, tenue (H6).

## 1. Spécification

**Prédicat** (du diagnostic) : montée en fréquence, de 88,2 à 192 kHz, d'un SSM-WaveNet ZOH
entraîné à 48 kHz et rééchelonné par Δ.

**Mécanisme** : le ZOH maintient l'échantillon *futur* x[n] sur l'intervalle ]t(n−1), t(n)]. Le
modèle apprend donc, à 48 kHz, une avance d'un demi-échantillon, qui ne se rééchelonne pas avec Δ.
Écart en τ², réduit de ×182 à ×557 par la correction à deux coefficients par mode, jusqu'à un
plancher d'ESR ≈ 2·10⁻⁴ (−37 dB) face au modèle à 48 kHz.

**Ce que le correctif doit atteindre** :

- (a) **Relecture FOH d'un checkpoint ZOH** (`zoh_to_foh`) : sortie identique à 48 kHz, et égale
  à la correction à deux coefficients aux autres fréquences. Argument analytique (section 3) :
  l'égalité est exacte, donc les écarts attendus sont des erreurs d'arrondi.
  - Sur les trois checkpoints SSM appariés (`paired_ssm_wavenet_s{925,736,688}`), les ESR
    `--as-foh` doivent égaler celles de `*_twotap.json` **à 10⁻³ près en relatif**, bras par bras.
  - Valeurs de référence, pour les graines 925, 736 et 688 :

    | bras | 925 | 736 | 688 |
    |---|---|---|---|
    | 44,1 kHz, contre l'appareil | 0,0391189 | 0,0460333 | 0,0375714 |
    | 88,2 kHz | 2,206e-4 | 2,147e-4 | 1,989e-4 |
    | 96 kHz | 2,223e-4 | 2,261e-4 | 1,969e-4 |
    | 192 kHz | 2,901e-4 | 3,079e-4 | 2,348e-4 |

    À 88,2, 96 et 192 kHz, la comparaison se fait contre le modèle à 48 kHz, dans la bande commune.
  - Identité à 48 kHz : ESR ≤ 10⁻¹².
- (b) **Entraînement avec la règle FOH** : optionnel, en fin de file GPU. Il n'a de sens qu'en
  non-infériorité à 48 kHz, puisque (a) résout déjà la montée sans réentraînement. Les
  prédictions (marge ln 1,15, appariement aux runs ZOH des partitions 925, 736 et 688) seront
  écrites dans un fichier à part, avant ces runs.

**Invariants hors prédicat**, avec leur valeur actuelle et leur tolérance :

- Chemins `free` et `zoh`, avec et sans FiLM : **bit à bit inchangés**. Les empreintes sha256
  (paramètres après initialisation, tirages aléatoires consommés, sortie FFT, 50 pas de `step`)
  sont prises au commit 87a9466, avant le changement, dans
  `tests/regression/test_ssm_wavenet_bitwise.py`. Tolérance nulle.
- Suite `make test` : 708 tests verts avant le changement. Tolérance : aucun rouge.
- Runs en file : exécutées depuis le worktree figé (`scripts/product_runner_worktree.sh`),
  inaffectées par construction.

## 2. Correctif

Une seule intervention : une troisième valeur de `discretization`, `"foh"`, dans
`DiagonalSSMLayer`, et une fonction de conversion `zoh_to_foh`.

- Récurrence : h[n] = a·h[n−1] + β₁·x[n] + β₀·x[n−1], avec z = λ·dt, a = e^z,
  φ₂(z) = (e^z − 1 − z)/z², β₁ = dt·φ₂(z), β₀ = dt·a·φ₂(−z).
  - φ₂ est calculé en complex128, par la série 1/2 + z/6 + z²/24 si |z| < 10⁻⁴.
  - β₀ + β₁ = (a − 1)/λ, le coefficient ZOH : le gain DC est inchangé.
- Noyau FFT : 2·Re(c·β₁·a^m) plus le même terme en β₀, décalé d'un échantillon.
- `step()` garde x[n−1]. `reset_state()` le remet à zéro.
- Mêmes paramètres et même initialisation que `"zoh"`, donc mêmes tirages aléatoires.
- `zoh_to_foh(model)` convertit sur place chaque couche ZOH :
  - c′ = c·b/(β₁ + β₀/a) ;
  - D′ = D + Σ_modes 2·Re(c·b − c′·β₁) ;
  - `discretization = "foh"`.

  La réponse impulsionnelle à 48 kHz est alors identique : les termes m ≥ 1 par c′, le terme m = 0
  par D′.

Hors de ce correctif, et dans des commits distincts :

- les options `--as-foh` et `--plain-test` de `scripts/product_sr_transfer.py` ;
- `--discretization foh` dans le banc, simple choix supplémentaire.

## 3. Argument de non-régression

- **`free` et `zoh`** : le correctif n'ajoute que des branches. Les expressions existantes de
  `_a`, `_b`, `_kernel`, `forward` et `step` ne sont pas réécrites, et la validation accepte
  `"foh"` dans la même branche que `"zoh"`, qui crée les mêmes paramètres dans le même ordre.
  - L'identité tient par construction.
  - Le test d'empreintes la vérifie au bit sur CPU.
  - Sur GPU, elle découle du fait que le code exécuté est le même.
- **Relecture FOH d'un modèle ZOH, égalité avec la correction à deux coefficients** : argument
  analytique.
  1. Le modèle relu est un système continu (pôles λ, résidus c′, terme direct D′) que la FOH
     discrétise **exactement pour toute entrée affine**, puisque l'interpolation linéaire d'une
     rampe est la rampe.
  2. Sa réponse discrète reproduit donc, à toute fréquence, le gain DC et le premier moment de ce
     système. À 48 kHz, ils valent ceux du mode ZOH d'origine, par construction de c′ et D′.
  3. La correction à deux coefficients est **définie** comme l'unique section
     (p + q·z⁻¹)/(1 − a′·z⁻¹) qui a ce gain DC et ce premier moment, avec le pôle a′ = e^(λ·dt′).
  4. La relecture FOH est une section de même forme, avec le même pôle : elle lui est égale.
  5. Le terme D′ ne diffère d'une fréquence à l'autre qu'en m = 0, ce que la forme (p, q) absorbe.
- **Entraînement FOH** : l'argument ne se ferme pas. Une autre règle de discrétisation est un
  autre biais inductif, et son effet sur l'entraînement est inconnu. **Contrôle empirique
  obligatoire** : la non-infériorité de (b).

## 4. Contrôle

Niveau retenu : invariants et relance hors GPU des trois checkpoints de référence. Le coût est
faible, sur CPU, et suffit pour (a). (b) demande des runs GPU, pré-enregistrés à part.

- **Invariants**, dans `tests/numerical/test_ssm_discretization.py`, en float64 :
  - finitude ;
  - parité FFT / pas à pas, avec des blocs de longueur irrégulière et une remise à zéro ;
  - causalité ;
  - DC exact : réponse à une constante = −2·Re(Σ c/λ) + D, en régime établi ;
  - exactitude entre fréquences sur une entrée affine par morceaux : à 48 et à 96 kHz, la sortie
    FOH échantillonne le même système continu aux instants communs ;
  - **paire discriminante** : sur une rampe, ZOH et FOH diffèrent d'une avance d'un
    demi-échantillon, et la FOH seule donne la réponse continue exacte ;
  - identité à décalage nul de log_dt ;
  - gradients finis de β₀ et β₁ pour z → 0, et gradient à |z| = 10⁻⁴ ± ε continu entre la série
    et la forme fermée ;
  - `zoh_to_foh` : identité à 48 kHz, et égalité avec `two_tap_kernel` (importée de
    `scripts/product_sr_transfer.py`, l'instrument qui a produit H6) à 44,1, 88,2, 96 et 192 kHz.
    C'est ce test qui établit l'équivalence avancée par l'agent de conception (écart ≤ 7·10⁻⁹),
    qui n'avait laissé aucun fichier.
- **Empreintes** : `tests/regression/test_ssm_wavenet_bitwise.py`, inchangées.
- **Relance de référence** : `scripts/product_sr_transfer.py --run-id paired_ssm_wavenet_sXXX
  --as-foh` pour 925, 736 et 688, comparées aux `*_twotap.json` avec la tolérance de la section 1.
- **Ce que ce contrôle ne couvre pas** : l'entraînement FOH ; S4-TF, que la relecture ne concerne
  pas, puisque ses pôles sont repliés et son TFiLM fixé en échantillons.

## 5. Exécution

- Code : `src/fssr_nam/models/ssm_wavenet.py` ; tests : ci-dessus.
- Relance : trois commandes CPU, enregistrements `sr_transfer_paired_ssm_wavenet_sXXX_last_asfoh`
  (nouveaux identifiants, dans `demo/sr_transfer/` et `demo/RUNS.jsonl`).
- Relecture Codex (`review`) avant d'annoncer terminé.

## 6. Rapport (2026-09-24)

**Invariants.**

- Empreintes bit à bit des chemins `free`, `zoh` et `zoh+FiLM` : inchangées.
- Suite complète : 728 tests verts avant l'ajout du code « pôles ».
- Tests FOH (`tests/numerical/test_ssm_discretization.py`) : tous verts, après correction de deux
  constructions de test fautives :
  - la causalité exigeait une tolérance nulle, alors que la FFT laisse un arrondi de 5·10⁻¹⁶ ;
  - le test inter-fréquences partait de x[0] ≠ 0. La FOH suppose alors une rampe depuis 0 sur
    l'intervalle précédent, qui dure dt à 48 kHz et dt/2 à 96 kHz : ce ne sont pas les mêmes
    entrées.

**Écarts à ce qui était écrit aux sections 2 et 4.**

- `two_tap_kernel` a été déplacé sans changement de `scripts/product_sr_transfer.py` vers
  `src/fssr_nam/models/ssm_wavenet.py`, pour que les tests l'importent sans charger le banc.
- La continuité entre la série et la forme fermée de φ₂ est testée sur ses valeurs, de part et
  d'autre de |z| = 10⁻⁴, et non sur ses gradients. La finitude des gradients près de z = 0,
  elle, est testée.
- Les options `--as-foh` et `--plain-test` et le choix `foh` du banc partent dans le même commit
  que le correctif. Ce commit contient aussi le code « pôles » de l'étape 3, entremêlé dans le
  même fichier.

**Équivalence avec la correction à deux coefficients : établie, et analytique.** Le noyau FOH
converti et celui de la correction à deux coefficients, cette dernière recalculée entièrement en
float64, coïncident à 4 à 8·10⁻¹³ près à 44,1, 88,2, 96 et 192 kHz. Contre `two_tap_kernel`
tel qu'il a mesuré H6, en complex64, l'écart est de 1 à 2,5·10⁻⁷, soit l'arrondi float32.
L'affirmation de l'agent de conception (« écart ≤ 7·10⁻⁹ ») est donc confirmée ; l'écart réel en
float64 est même plus petit.

**Deux défauts trouvés en route.**

1. **Aliasing de C**, trouvé par le test float64. Sur une couche float64, `C.double()` est C
   lui-même, que `copy_` écrasait avant le calcul de D. Corrigé par `.clone()`.
2. **Relecture Codex, P2.** La conversion recalculait b = (a − 1)/λ en float64. Or le modèle a
   été entraîné avec le b de la couche float32, qui perd des chiffres par annulation sur les modes
   lents. Dans un cas extrême (log_dt = −15), l'écart d'identité atteignait une ESR de 5,8·10⁻³.
   La conversion part désormais des coefficients calculés par la couche, promus ensuite en
   float64. Un test float32 la couvre ; il échoue sur l'ancienne conversion.

**Relance de référence** (`compare_asfoh.py` ; enregistrements `*_asfoh.json`, et `*_asfoh_v1.json`
pour la conversion avant le point 2) :

| graine | identité à 48 kHz, ESR | écart relatif maximal aux quatre bras à 2 coefficients |
|---|---|---|
| 925 | 4,0·10⁻¹¹ | 9,4·10⁻⁴ (88,2 kHz) |
| 736 | 2,3·10⁻¹¹ | 9,97·10⁻⁴ (96 kHz) |
| 688 | 1,5·10⁻¹¹ | 7,7·10⁻⁴ (192 kHz) |

- **Égalité avec la correction à deux coefficients : tenue**, pour les douze bras, sous la
  tolérance de 10⁻³. Le bras le plus proche de la limite est à 0,997·10⁻³.
- **Identité à 48 kHz : critère non tenu tel qu'écrit** (≤ 10⁻¹²). Le témoin
  (`asfoh_precision.py`) montre que ce seuil était inatteignable en float32 :

  | graine | FOH float64 contre ZOH float64 | FOH float32 contre ZOH float32 | ZOH float32 contre ZOH float64 |
  |---|---|---|---|
  | 925 | 1,3·10⁻²⁸ | 3,8·10⁻¹¹ | 2,4·10⁻¹⁰ |
  | 736 | 5,6·10⁻²⁹ | 1,6·10⁻¹¹ | 2,1·10⁻⁹ |
  | 688 | 3,2·10⁻²⁹ | 1,0·10⁻¹¹ | 7,8·10⁻¹⁰ |

  La conversion est exacte en float64. En float32, son écart est 10 à 100 fois plus petit que
  l'écart du modèle d'origine avec lui-même d'une précision à l'autre. Le seuil de la section 1
  est laissé tel quel ; l'erreur était de l'avoir fixé sans regarder la précision de travail.

**Conséquence produit.** Un checkpoint ZOH existant se relit en FOH sans réentraînement ; il se
transfère alors à toute fréquence par le seul rééchelonnement de Δ, avec la qualité de la
correction à deux coefficients (−37 dB face au modèle à 48 kHz). Le moteur natif n'a donc à
implémenter que la FOH.

**Incertitudes restantes.**

- Le plancher de −37 dB reste non attribué (étape 2e).
- L'entraînement FOH (b) n'est pas fait : optionnel, en fin de file.
