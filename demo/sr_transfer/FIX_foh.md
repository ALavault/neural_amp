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

## 6. Rapport

À remplir après la relance.
