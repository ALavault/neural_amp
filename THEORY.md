# Registre des énoncés théoriques

Seule source du statut des énoncés ; les autres documents citent `T<n>` sans recopier le statut.
Statuts : **D** démontré (preuve complète, contrôle symbolique, contrôle numérique sur le code
réel, relecture indépendante) ; **E** esquisse ; **S** supposé ; **M** mesuré sous
préinscription, sans preuve ; **PH** post hoc ; **R** réfuté.

## T1 — Moments du maintien : avance du ZOH, avance nulle du FOH

Statut : D (2026-10-02) ; à revérifier si `src/fssr_nam/models/ssm_wavenet.py`
(`DiagonalSSMLayer`, `phi2`, `pole`, `foh_late`) change
Énoncé : pour un mode diagonal stable (λ, Δ, c), z = λΔ, w = −c/λ : le noyau ZOH et le noyau
FOH ont le gain continu 2 Re(w) ; le premier moment du FOH égale celui du mode continu ; celui du
ZOH est en avance de 2 Re(w φ(z)), φ(z) = −1/z − 1/(e^{−z} − 1) = 1/2 + z/12 − z³/720 + O(z⁵).
Pour un mode réel (z = −s, Re c ≠ 0), l'avance vaut 1/s − 1/(e^s − 1) ∈ ]0 ; 1/2[ échantillon, et le
retard pris quand Δ → Δ/r (r > 1) vaut κ_mode·(1 − 1/r)/2 échantillon d'origine, avec
κ_mode = 2 (g(s/r) − g(s))/(s (1 − 1/r)) ∈ ]0 ; 1[, g(x) = x/(e^x − 1).
Hypothèses : H1 Re λ < 0, Δ > 0 [mathématique ; vrai par construction, λ = −e^{α} + iω] ;
H2 noyaux infinis [mathématique ; le code les tronque à la longueur du signal] ; H3 arithmétique
exacte et formules fermées [mathématique ; le code s'en écarte par `phi2` en série,
`pole`, `foh_late` et l'arrondi, et pour |z| ≲ 10⁻⁸ en float32 a s'arrondit à 1] ; H4 gain non
nul (G ≠ 0 pour un canal, Re c ≠ 0 pour un mode réel) [mathématique]
Portée : fréquence nulle seulement (premier moment), pas les retards par bande ; une couche
linéaire, pas le réseau ; le rapport par canal exige un gain continu non nul ; la borne κ_mode < 1
ne vaut que pour les modes réels (les modes entraînés sont surtout complexes) ; noyaux décalés par
les boutons non contrôlés ; le code tourne en complex64, le contrôle en float64
Objet : `src/fssr_nam/models/ssm_wavenet.py:DiagonalSSMLayer` (`_kernel`, `_foh_kernel`,
`_foh_inputs`) @ 2d49ac7 ; correspondance dans la preuve
Preuve : `theory/T1_hold_moments.md`
Vérifications : symbolique `theory/t1_check.py` (identités des poids FOH, moments, série de φ,
encadrement de g′) ; numérique `theory/t1_check.py` (vrai `DiagonalSSMLayer` en float64,
96 modes et 24 canaux par discrétisation, paramètres génériques tirés au hasard, écart
1,74·10⁻¹⁰ ; κ_mode des modes réels à r = 2 et 4, écart 4,5·10⁻¹¹) ; relecture : Codex gpt-6-astra, 2026-10-02, deux passes (xhigh puis high). Passe 1 : étapes
algébriques justes ; à corriger : gain nul (contre-exemple c = i, ajouté en H4), objet idéal
contre code flottant (H3 et Portée), comptes 96 modes et 24 canaux (corrigés) ; commentaire
d'en-tête de `ssm_wavenet.py` en désaccord avec le code exécuté (signalé, non modifié). Passe 2 :
chaque correction juste et suffisante, r > 1 à reporter (fait) ; « preuve complète, statut
démontré possible »
Utilité : dit quelle quantité la correction de transfert de fréquence doit annuler (l'avance du
premier ordre, portée surtout par les modes lents) et pourquoi la relecture ou l'entraînement en
FOH l'annule exactement à fréquence nulle (`demo/sr_transfer/FIX_foh.md`) ; borne la part d'un
mode réel dans κ (T4)
Rétrodictions : le « demi-échantillon » de `demo/sr_transfer/RESULTS_hold.md` (limite s → 0) ;
l'ESR de 2,8 à 4,0·10⁻⁴ des modèles entraînés en FOH face à eux-mêmes après rééchelonnement
(`demo/sr_transfer/sr_transfer_paired_ssm_wavenet_foh_s*_last_plain.json`), contre 0,036 à 0,077
pour le ZOH non corrigé
Prédictions nouvelles :
- T1-P1 (mesurée le 2026-10-02, soutenue ; inscrite le 2026-10-02) : sur `paired_ssm_wavenet_foh_s{925,736,688}`,
  avec la méthode de `demo/sr_transfer/kappa.py` (rééchelonnement de Δ à 96 et 192 kHz, retour
  à 48 kHz, retard par bande d'octave, Welch 4096, cohérence ≥ 0,5), la médiane des κ aux bandes
  500, 1 000 et 2 000 Hz (3 checkpoints × 2 fréquences) vérifie |κ| ≤ 0,15 ; réfutée si
  |κ| ≥ 0,5. Aucune prédiction à 4 et 8 kHz, où l'erreur du FOH n'est plus petite
  (ωΔ ≈ 0,5 à 1). L'ESR existante borne déjà le retard global ; l'apport est la résolution par
  bande.
  Précisions inscrites le 2026-10-02, avant toute mesure (script `demo/sr_transfer/kappa_foh.py`,
  pas encore écrit). Checkpoint `last`, segments de test du banc comme `kappa.py:bigmuff`.
  Lecture primaire : pour chaque (checkpoint, fréquence), médiane des retards aux bandes 500,
  1 000 et 2 000 Hz, divisée par (1 − 48 000/fs)/2 ; puis médiane des 6 κ ; soutenue si
  |κ| ≤ 0,15, réfutée si |κ| ≥ 0,5, intermédiaire entre les deux ; une bande sans valeur
  (cohérence) est omise et déclarée. Lecture secondaire, sans verdict propre : médiane des 18 κ
  par bande. Bandes 4 et 8 kHz rapportées sans lecture. Portes, sans lesquelles aucune lecture
  n'est faite : chaque module SSM chargé a `discretization == "foh"` ; l'ESR à 48 kHz contre la
  cible est à 2 % relatifs de `arms["48k"]["esr"]` du `_last_plain.json` du même checkpoint ;
  l'ESR entre la sortie ramenée de 96 ou 192 kHz et la sortie à 48 kHz (intérieurs) est
  ≤ 10⁻³ (ZOH non corrigé : 0,036 à 0,077).
  Résultat (`demo/sr_transfer/kappa_foh.json`, commit 34c0032) : portes tenues (FOH sur les
  trois checkpoints, ESR à 48 kHz égale au record, aller-retour de 2,6 à 4,3·10⁻⁴). κ primaire
  0,036, soutenue. Lecture secondaire 0,034, aucune bande omise. Les 6 κ par
  (checkpoint, fréquence) vont de 0,031 à 0,043, les 18 κ par bande de 0,003 à 0,059. Sans
  lecture : à 4 et 8 kHz, κ de 0,041 à 0,157. Les ZOH aux mêmes graines (`kappa.json`), qui sont
  d'autres entraînements, ont des κ de 1,9 à 2,6 aux trois bandes prédites (1,8 à 2,6 sur les
  cinq). Non prédit : les 18 estimations par bande sont positives sur ces segments. Aucune
  incertitude n'est estimée, donc le signe n'est pas établi, et l'énoncé ne dit rien de lui.
  Relecture Codex gpt-6-astra, lecture seule, 2026-10-02 : règle, portes et chiffres conformes ;
  deux formulations corrigées (plage ZOH par bandes, signe non établi).
Historique : 2026-10-02 — → E : preuve et contrôles écrits ; 2026-10-02 E → D : relecture
en deux passes, corrections appliquées

## T2 — Piège de polarité de L1 + MR-STFT le long du gain de sortie

Statut : D (2026-10-02) ; à revérifier si `scripts/product_nablafx_bench.py` (perte),
`third_party/auraloss/auraloss/freq.py` ou `third_party/nablafx/nablafx/evaluation/functions/`
changent
Énoncé : avec L = w₁ L1 + w₂ MRSTFT (auraloss par défaut, plancher η = 10⁻⁸ sur la puissance) et
ŷ = (−1 + ε) y : L(ε) − L(0) = −w₁ E|y| ε + w₂ m |ε| + O(ε²), où m = moyenne sur les
résolutions de (ρ_r + f_r) ∈ [0 ; 2] (f_r : fraction des cases de la cible au-dessus du
plancher ; ρ_r : leur part de la norme de Frobenius). Avec w₁, w₂ > 0, y ≠ 0 et aucune
puissance STFT de la cible exactement au plancher : ŷ = −y est un minimum local strict du chemin
de gain si w₁ E|y| < w₂ m (et encore à l'égalité si une case dépasse le plancher, par le terme
d'ordre 2 du logarithme) ; ce n'en est pas un si w₁ E|y| > w₂ m. Pour un modèle à activation
finale identité, w₁ E|y| > w₂ m implique que ŷ = −y n'est pas un minimum local du modèle
complet.
Hypothèses : H1 ŷ = −y exactement [mathématique] ; H2 aucune puissance STFT de la cible
exactement égale à 10⁻⁸ [mathématique ; contrôlée sur les cibles réelles par
`theory/t2_check.py`, qui rapporte aussi la fraction de cases franchissant le plancher] ; H3 perte du banc telle que lue dans le code
[mathématique, code @ 2d49ac7]
Portée : une direction, une solution exacte, un lot ; rien sur la dynamique de l'optimiseur ni
sur la probabilité d'inversion ; le corollaire sur le modèle complet n'est pas établi pour la
tête tanh du banc (écart d'environ 3 % pondéré par |y| sur la cible Big Muff, non contrôlé) ;
contrôle en float64, entraînement en float32
Objet : `scripts/product_nablafx_bench.py:main` (`FlexibleLoss`, `l1_loss`, `mrstft_loss`) et
`auraloss.freq.STFTLoss` @ 2d49ac7
Preuve : `theory/T2_polarity_trap.md`
Vérifications : symbolique `theory/t2_check.py` (pentes unilatérales, `sympy.limit`) ; numérique
`theory/t2_check.py` (vraie perte du banc, 12 cibles de test Big Muff, poids 1 / 0,1 et
0,5 / 0,5, cibles mises à l'échelle à 0,8 et 1,25 fois l'échelle de référence
c₀ = w₂ m(y)/(w₁ E|y|) : pentes mesurées égales aux prédites, signe de la pente à droite
basculant entre les deux ; cas d'égalité : incrément à droite positif) ; relecture : Codex gpt-6-astra, 2026-10-02, deux passes (xhigh puis high). Passe 1 : contre-exemple au
« si et seulement si » au cas d'égalité (accepté, vérifié indépendamment sur une cible
synthétique ; l'égalité donne encore un minimum strict), H2 remplacée par « aucune puissance au
plancher », c₀ n'est qu'une échelle de référence, `PolarityGuard` hors de l'énoncé, journal
incomplet (relancé). Passe 2 : corrections justes et suffisantes, chiffres conformes au journal ;
notation ε/η, marge calculée sur le lot de test, écart de 8 % au cas d'égalité à garder comme
hypothèse (corrigés) ; « preuve complète, égalité comprise, statut démontré possible »
Non-vacuité : sur les cibles Big Muff, m = 1,93 et E|y| = 0,0149 : le piège tient avec une marge
×12,9 en 1 / 0,1 et ×129 en 0,5 / 0,5. À cible fixe, ces marges sont exactes pour un
changement du rapport des poids ; pour une cible mise à l'échelle c, la condition porte sur
m(cy), qui varie avec c, et le contrôle montre la bascule entre 0,8 et 1,25 fois ces facteurs
Utilité : dit que le poids relatif L1 / MR-STFT n'est pas sans dimension (son effet sur ce piège
suit le niveau de la cible), que le piège existe le long du gain au protocole Big Muff publié
(sans dire si `PolarityGuard` est nécessaire : un minimum le long d'une direction n'en est pas
un dans toutes), et quel rapport de poids le ferme pour un appareil donné (w₁/w₂ > m/E|y|)
Rétrodictions : la run inversée `ssmzoh_seed42` (`paper/icassp2027/data/polarity_ssmzoh_seed42.json`)
est compatible avec un piège le long du gain (marge ×12,9 calculée sur le lot de test ; la
marge sur les lots d'entraînement n'est pas calculée), pas expliquée (l'énoncé n'en donne pas
la probabilité)
Prédictions nouvelles :
- T2-P1 (non mesurée, demande des entraînements, hors du goal en cours) : SSM-WaveNet sur
  Big Muff, sans `PolarityGuard`, avec w₁ E|y| ≥ 1,5 w₂ m (par exemple w₁ = 20 pour w₂ = 0,1),
  ne finit inversé pour aucune graine sur cinq.
  Non lancée (décision du 2026-10-02) : telle quelle, la prédiction ne départage rien. Runs Big
  Muff SSM-WaveNet entraînés sans garde aux poids 1 / 0,1 (`demo/nablafx_bench`, `RUNS.jsonl`) :
  en ZOH, configuration actuelle, un seul run, `ssmzoh_seed42`, inversé ; en discrétisation
  `free`, trois runs (`ssmwavenet_seed42` à `44`), aucun inversé. Taux en commun p ≈ 1/4, donc
  (1 − p)⁵ ≈ 0,24 : « 0/5 inversées » s'observerait une fois sur quatre sans aucun effet. Le
  taux en ZOH est inconnu (n = 1). Indice descriptif : les runs ZOH gardés
  (`ssmzoh_guard_seed4*`) ont 0 à 5 retournements chacun, jusqu'à l'époque 20, mais un
  retournement du garde n'est pas une inversion finale.
  Esquisse d'un plan qui départagerait : deux bras en ZOH sans garde, aux mêmes graines, l'un
  aux poids 1 / 0,1 (témoin), l'autre à w₁ = 20 ; comparaison des comptes d'inversions finales
  par un test exact. Pour p ≈ 0,4, il faut environ 12 graines par bras, soit à peu près 85 h de
  GPU. Confondant à déclarer : le seuil d'écrêtage du gradient (1) ne se rééchelonne pas avec
  la perte.
Historique : 2026-10-02 — → E : preuve et contrôles écrits ; 2026-10-02 E → D : relecture
en deux passes, critère corrigé au cas d'égalité

## T3 — Déclenchement de l'ordonnanceur à patience

Statut : M pour l'indépendance au seuil, S pour la loi de premier passage (2026-10-02) ; à
revérifier si `diagnosis/seeds/plateau_simulation.py` ou la configuration de
`ReduceLROnPlateau` (mode min, seuil relatif 10⁻⁴, patience 20) changent
Énoncé : (a) l'époque de la première division ne dépend pas du seuil θ dans [10⁻⁵ ; 10⁻³] ;
(b) c'est le temps de premier passage d'une tendance plus un bruit indépendant de l'époque à
l'époque.
Hypothèses : H1 résidus de log L de validation indépendants d'une époque à l'autre [mesurée :
VR(10) médian 0,166, `diagnosis/butterfly/patience_law.json`] ; H2 tendance quadratique en
log(époque + 1) [supposée]
Portée : 21 runs SSM-WaveNet Big Muff (5 enfants de la fourche 100, 16 runs du pilote C) ; une
architecture, une configuration de l'ordonnanceur
Objet : idéal (simulation) ; rejeu `plateau_simulation.halvings`, porte 21 sur 21
Preuve : aucune ; mesure `diagnosis/butterfly/patience_law.md`
Vérifications : mesure préinscrite (commit 2b447f3) ; (a) médiane des médianes prédites décalée
de 0 et −1 époque contre un critère de 5 ; (b) couverture 12/21 = 0,57, intermédiaire (soutien
≥ 0,60, réfutation ≤ 0,40), les 9 écarts tous vers une division plus tardive que prévu
Utilité : ne pas régler θ ; traiter les décisions précoces de l'ordonnanceur comme aléatoires
entre graines ; (b) reste à corriger (tendance ou mémoire du bruit) avant de prédire une loi
Historique : 2026-10-02 — → M (a), S (b) : résultats de `patience_law.md`

## T4 — Décalage du réseau au transfert de fréquence (κ)

Statut : M (2026-10-02) pour la forme ; à revérifier si `src/fssr_nam/models/ssm_wavenet.py`
(discrétisation ZOH) change
Énoncé : un SSM-WaveNet ZOH entraîné à 48 kHz, rééchelonné par Δ à fs′, prend un retard de
κ (1 − 48 000/fs′)/2 échantillon à 48 kHz, avec κ propre au checkpoint.
Hypothèses : H1 forme (1 − T′/T) [mesurée : R1 de `demo/sr_transfer/predictions_kappa.md`,
rapport κ(192)/κ(96) de 0,991 à 1,031 sur 4 checkpoints] ; H2 κ constant pour une architecture
[réfutée : R3, κ de 1,90 à 2,53 entre trois graines Big Muff]
Portée : ZOH, Δ rééchelonné sans correction, 96 et 192 kHz ; pas le FOH (T1-P1) ; retard pur
(R4) sur Big Muff, dispersif sur le VS100R (500 Hz moins retardé)
Objet : `demo/sr_transfer/kappa.py` @ fa44f45
Preuve : aucune pour le réseau. T1 borne κ_mode ∈ ]0 ; 1[ pour un mode réel ; κ > 1 partout
exige donc une cascade de couches, des modes complexes ou des signes mêlés, non départagés
(la mesure par bloc, R5, est invalide par construction)
Vérifications : mesure préinscrite (2b447f3) ; κ VS100R 1,964, en accord avec l'estimation
indirecte 1,86 (R2)
Utilité : taille du retard non corrigé à 96 et 192 kHz pour le produit (0,25 κ et 0,375 κ
échantillon, soit 0,48 à 0,95 échantillon) ; une correction doit être calculée par checkpoint,
pas fixée par architecture
Historique : 2026-10-02 — → PH ; 2026-10-02 PH → M : mesure préinscrite (forme tenue, κ
dépendant du checkpoint)

## T5 — Ce que conditionne le bras « pôles » (C et D fixes)

Statut : E (2026-10-02) ; à revérifier si `src/fssr_nam/models/ssm_wavenet.py`
(`SSMWaveNet.__init__`, `forward`, `pole_shifts`, `GatedSSMBlock.forward`,
`DiagonalSSMLayer.forward`, `_shifted_inputs`, `_shifted_kernel`) change
Énoncé : pour `SSMWaveNet(cond="poles").forward` en ZOH ou en FOH (convolution à état initial
nul, réglage u constant sur le segment) :
(1) la sortie dépend de u seulement à travers les Δ_{i,c}(u) et les Re λ_{i,c,m}(u). Deux
réglages qui donnent ces mêmes familles donnent la même sortie. ω (`A_imag`), C, D et les autres
couches ne dépendent pas de u.
(2) Le gain continu de chaque canal SSM vaut D + Σ_m 2 Re(−c_m/λ_m(u)) et ne dépend pas de Δ(u).
(3) Pour une entrée constante, la limite de la sortie du réseau (convolution prolongée
indéfiniment) ne dépend de u qu'à travers les Re λ(u). Un réglage qui ne déplace que les Δ la
laisse inchangée.
(4) Le résidu continu w_m(u) = −c_m/λ_m(u) vérifie w λ = −c. Pour un mode réel, la contribution
au gain continu vaut 2 Re(c) e^{−(α+ε)} : un décalage ε la multiplie par e^{−ε} si Re c ≠ 0, et
elle est nulle si Re c = 0. Rien n'est dit du gain total d'un canal.
Hypothèses : H1 `cond="poles"`, ZOH ou FOH [mathématique ; lecture du code, énoncé 1] ;
H2 noyaux infinis [mathématique ; sur un segment de longueur L, la somme tronquée d'un mode ZOH
vaut 2 Re(−c (1 − a^L)/λ)] ; H3 formules fermées exactes [mathématique ; le code s'en écarte par
`pole` (Re z ≥ −80 ; exact en ZOH ; en FOH, `foh_late`, écart relatif d'ordre e^{−80} sur le
gain modal complexe, pas sur sa partie réelle), `phi2`
en série et l'arrondi float32/complex64]
Portée : (2) et (3) portent sur la fréquence nulle seulement. À une fréquence non nulle, Δ seul
peut changer le gain d'un canal (mode réel normalisé, Ω = 0,01 : |H| = 0,707 à Δ = 0,01, 0,894 à
Δ = 0,02) ; il ne le change pas toujours (C = 0, ou contributions qui se compensent). (3) porte sur une entrée constante, et sa limite peut ne pas être atteinte sur un
segment si un mode est lent. Le chemin récurrent (`step`), qui dépend de l'historique de l'état,
n'est pas couvert. En temps discret, c·b(λ, Δ) et K[0] peuvent varier avec δ et ε. (1) dit par quoi passe
u, pas ce qui est appris.
Objet : `src/fssr_nam/models/ssm_wavenet.py` @ 2d49ac7 ; correspondance dans la preuve
Preuve : `theory/T5_poles_arm.md` (1 par lecture du code ; 2 par T1, énoncé 1 ; 3 par
récurrence sur les blocs ; 4 immédiat)
Vérifications : symbolique `theory/t5_check.py` (gain −c/λ et dérivée nulle en Δ, en ZOH et en
FOH ; mode réel). Numérique `theory/t5_check.py` : vrai `SSMWaveNet(cond="poles")` en float64,
3 blocs × 4 canaux × 3 modes, paramètres et têtes tirés au hasard, ZOH et FOH, 3 graines.
- N1 : deux réglages donnent des sorties distinctes (écart maximal de 0,091 à 0,251 selon le cas), mais le second,
  évalué avec les décalages du premier, donne exactement la sortie du premier.
- N2 : somme du noyau + D contre la forme fermée, écart relatif ≤ 4,6·10⁻¹³. Avec Δ seul
  déplacé, ≤ 8,3·10⁻¹⁶ en ZOH et ≤ 5,9·10⁻¹³ en FOH. Avec les pôles seuls déplacés, écart
  ≥ 8·10⁻².
- N3 : dernier échantillon de la réponse à une constante, 2¹⁵ échantillons. Avec Δ seul
  déplacé, écart ≤ 7,2·10⁻¹⁵. Avec les pôles seuls, ≥ 2,1·10⁻². Le même déplacement de Δ seul
  change la réponse au bruit d'au moins 1,5·10⁻³.
- N4 : modes réels isolés, à cinq valeurs de ε, somme finie = 2 Re(c) e^{−(α+ε)} à ≤ 9,4·10⁻¹⁴
  près ; somme nulle si Re c = 0.
- N5 : autour du plancher et au-delà (Re z = −79,9 ; −80,1 ; −202 ; −800, sous le −709 que
  `foh_late` évite), somme finie = 2 Re(−c/λ) à ≤ 1,7·10⁻¹⁶ près en relatif ; float64 ne mesure
  pas l'écart d'ordre e^{−80}. Mode lent (λ = −1, Δ = 10⁻⁵, L = 2¹⁵) : somme tronquée égale à 10⁻⁹
  près à une référence calculée sur 50 chiffres puis arrondie en float64 (ZOH 0,279406427 =
  1 − a^L ; FOH 0,279402824), contre une limite de 1. Chaque valeur comparée est contrôlée finie
  (N1 à N5).
Relecture : Codex gpt-6-astra, 2026-10-02, passe 1 (xhigh). (1) tenu pour `forward`, mais le
chemin récurrent dépend de l'historique : l'objet a été restreint (accepté). (2) correct ;
`foh_late` à citer, et e^{−80} est relatif (accepté). (3) récurrence valide une fois la
convergence dominée détaillée (fait), mais « s'éteindre dans le segment » confondait limite et
troncature (reformulé avec le facteur 1 − a^L). (4) surénoncé : contre-exemples Re c = 0 et
coupure discrète déplacée par Δ (énoncé restreint). Utilité : glissement de la fréquence nulle aux
basses fréquences, contre-exemple chiffré vérifié (corrigé). Contrôles manquants pour (4) et pour
les cas limites (N4 et N5 ajoutés). Son exécution de `t5_check.py` a échoué dans son bac à sable
(`snap-confine`, caches). Il a rejoué la branche sans checkpointing des noyaux et reproduit les
chiffres. Verdict : D impossible en l'état. Passe 2 (high) : restriction à `forward`, (2), (3) et
troncature justes ; restent la condition Re c ≠ 0 dans l'Utilité (corrigé), « à toute fréquence
non nulle » faux pour C = 0 (corrigé en « peut »), e^{−80} relatif au gain modal complexe
seulement (précisé), N4 sans contrôle de finitude (ajouté), N5b FOH sans référence (référence en
50 chiffres ajoutée, valeur FOH égale à celle calculée par Codex), N5a sans cas autour de −80 ni
sous −709 (ajoutés). Verdict : pas encore D. Passe 3 (high) : Re c ≠ 0, borne e^{−80}, N4 et
N5a suffisants ; restaient une phrase universelle dans la preuve (« Aux autres fréquences, Δ change
bien la réponse », corrigée en « peut »), des NaN qui pouvaient passer N2 et N3 (max(0, NaN) = 0 en
Python ; contrôle de finitude ajouté avant chaque agrégation) et la précision « 50 chiffres »
revendiquée pour N5b (ramenée à 10⁻⁹). Verdict : conserver E. Passe 4 (high) : N5b et la phrase
de la preuve sont justes ; le contrôle de finitude ne portait que sur les opérandes (0/0 donne NaN,
1e308 − (−1e308) donne inf ; le résultat et le dénominateur sont maintenant contrôlés) ;
« varient » ramené à « peuvent varier », coupure observable seulement si la contribution est non
nulle, bornes de N1 précisées. Verdict : conserver E. Passe 5 (high) : finitude et formulations
validées ; un seul point bloquant, « donc la coupure » (une contribution non nulle ne garantit pas
de coupure dans la bande : λ = −1, c = 1/2, Δ = 2 donne un minimum de |H| de 0,762, vérifié ;
la phrase est limitée au déplacement du pôle discret) ; mineur : variante `sine` à citer (fait).
Passe 6 à faire.
Utilité : « conditionner les pôles » ne teste pas seul « déplacer les coupures »
(`demo/param/predictions_poles.md`, verdict G2.2). Si Re c ≠ 0, le décalage ε d'un pôle réel
change aussi sa contribution au gain continu (4). Le gain d'un canal à fréquence nulle ne change pas par Δ seul
(2). Cela ne vaut qu'à fréquence nulle, voir Portée. L'analyse C de `predictions_fast_modes.md`
mesure par quelle voie passent les boutons.
Rétrodictions : aucune mesurée.
Prédictions nouvelles : aucune. L'énoncé est structurel, des identités du code. Il dit comment
lire une mesure du bras D (toute variation du gain d'un canal à fréquence nulle entre réglages se
lit dans ε), il ne prédit pas de résultat d'entraînement.
Historique : 2026-10-02 — → E ; 2026-10-02 : énoncé précisé en quatre points, preuve et
contrôles écrits ; 2026-10-02 : passe 1 de Codex, objet, énoncé 4 et Utilité restreints, N4 et
N5 ajoutés ; 2026-10-02 : passe 2, surénoncés résiduels et contrôles N4 et N5 corrigés ; 2026-10-02 : passe 3, phrase universelle,
finitude de N2 et N3 et précision de N5b corrigées ; 2026-10-02 : passe 4, finitude des
résultats et deux formulations corrigées ; 2026-10-02 : passe 5, phrase sur la coupure retirée
(E, passe 6 à faire)
