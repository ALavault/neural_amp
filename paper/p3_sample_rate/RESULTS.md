<!-- Généré par paper/render.py depuis RESULTS.src.md. -->

# P3 — Transfert de fréquence d'échantillonnage de SSM-WaveNet : résultats

<!-- Conditions par défaut, sauf mention dans le résultat : voie produit (D-PRODUCT-001), publiable étiquetée exploratoire (D-PRODUCT-002), captures scellées non lues. Big Muff ToneTwist S050_V100, protocole NablAFx. SSM-WaveNet (8 blocs, 16 canaux, 4 états par canal), discrétisation ZOH, entraîné à 48 kHz ; checkpoints `last` des runs appariés `paired_ssm_wavenet_s925`, `paired_ssm_wavenet_s736`, `paired_ssm_wavenet_s688` (graine 42, partitions 925, 736, 688). Évaluation CPU de checkpoints existants. « Descente » : à 44 100 Hz contre la capture native de l'appareil. « Montée » : à 88 200 Hz, 96 kHz et 192 kHz, sortie ramenée à 48 kHz et comparée au même modèle à 48 kHz dans la bande commune, sur l'intérieur des segments : c'est une cohérence avec le modèle à 48 kHz, pas une fidélité à l'appareil. Exceptions : les bras d'origine de `demo/sr_transfer/predictions.md` (P1 à P9, fichiers `*_last.json`) sont notés sur les segments entiers, et en montée (`*_vs_around`, P5, P6, P7, P9) ils comparent à fs′ avec `around` (le modèle à 48 kHz rééchantillonné autour), sans retour à 48 kHz. -->

## Établi

- **Descente à 44 100 Hz sans perte par rééchelonnement de Δ.** Rapport ESR `delta` / `around` (rééchantillonnage hors ligne autour du modèle à 48 kHz) de 0,899 à 0,977 (P2 : au plus 5 % au-dessus) ; `around` à 1,004 à 1,004 fois l'ESR à 48 kHz (P1 : ±5 %).
  - Preuve : F-0026, E-0025 ; `demo/sr_transfer/RESULTS.md` ; `demo/sr_transfer/sr_transfer_paired_ssm_wavenet_s925_last.json` (et s736, s688).
  - Statut : préinscrit (`demo/sr_transfer/predictions.md`), P1 et P2 tenues sur les 3 partitions. Contrôle post hoc des bords (intérieur des segments, fichier entier rééchantillonné) : rapport 0,885 à 0,975, conclusion inchangée.
  - Réserve : l'avance de `delta` sur `around` n'est pas un meilleur transfert (H8, section Retiré ou réfuté).
- **Ne pas rééchelonner Δ n'est pas une option en montée.** Avec Δ inchangé, ESR contre le modèle à 48 kHz de 1,22 à 1,32 à 96 et 192 kHz, soit 7,6 à 30,4 fois celle de `delta`.
  - Preuve : E-0025 ; `demo/sr_transfer/RESULTS.md` (P7) ; `demo/sr_transfer/sr_transfer_paired_ssm_wavenet_s925_last.json` (et s736, s688).
  - Statut : préinscrit (P7, au moins 5 fois), tenu sur les 3 partitions.
- **L'écart de montée non corrigé ne s'effondre pas à bas niveau.** Rapport ESR(−20 dB) / ESR(0 dB) de `delta` de 0,56 à 0,59 à 96 et 192 kHz, dans la plage 1/2 à 2 fixée pour « le maintien domine » (sous 1/5 : « le repliement appris domine »).
  - Preuve : E-0025 ; `demo/sr_transfer/RESULTS_hold.md` (H3) ; `demo/sr_transfer/sr_transfer_paired_ssm_wavenet_s925_last_holdtest.json` (et s736, s688).
  - Statut : préinscrit (`demo/sr_transfer/predictions_hold.md`), H3 tenue sur les 3 partitions, près de la borne basse : une part de l'écart dépend du niveau.
- **Correction à deux coefficients par mode : la montée revient à un plancher.** Par mode, c·(b0 + b1·z⁻¹)/(1 − a′z⁻¹), a′ = e^{λΔ′}, gain continu et premier moment égaux à ceux du mode à 48 kHz. ESR contre le modèle à 48 kHz de 2,0·10⁻⁴ à 3,1·10⁻⁴ (-37,1 à -35,1 dB) à 88 200 Hz, 96 kHz et 192 kHz ; réduction de l'écart non corrigé de ×182 à ×557 ; identité à 48 kHz, ESR de 1,2·10⁻⁹ à 5,6·10⁻⁹.
  - Preuve : F-0026, E-0025 ; `demo/sr_transfer/RESULTS_hold.md` (H6) ; `demo/sr_transfer/sr_transfer_paired_ssm_wavenet_s925_last_twotap.json` (et s736, s688).
  - Statut : préinscrit (H6 de `demo/sr_transfer/predictions_hold.md` : au moins ×10 et mieux que le maintien à un coefficient), tenu sur les 3 partitions.
  - Réserve : le filtre (b0, b1) amplifie l'aigu près de Nyquist à fs′, invisible dans la bande commune (réserve écrite avant le bras).
- **Relire un checkpoint ZOH en maintien du premier ordre (FOH) est exactement la correction à deux coefficients.** c′ = c·b/(β₁ + β₀/a), le terme m = 0 reporté dans D. Argument analytique (même pôle, même gain continu, même premier moment, forme (p + q·z⁻¹)/(1 − a′z⁻¹) unique) ; sur les 12 bras (44 100 Hz contre l'appareil, 88 200 Hz, 96 kHz, 192 kHz ; 3 partitions), écart relatif maximal des ESR de 9,97·10⁻⁴, sous la tolérance 1·10⁻³.
  - Preuve : F-0027, E-0025 ; `demo/sr_transfer/FIX_foh.md` (sections 1 à 3 et 6) ; `demo/sr_transfer/sr_transfer_paired_ssm_wavenet_s925_last_asfoh.json` (et s736, s688) ; test `tests/numerical/test_ssm_discretization.py`.
  - Statut : tolérance préinscrite (`demo/sr_transfer/FIX_foh.md`, section 1), tenue. Le critère d'identité à 48 kHz, lui, n'est pas tenu tel qu'écrit (section Retiré ou réfuté). Conversion actuelle (`_asfoh`), pas `_asfoh_v1`, corrigée après relecture Codex.
- **T1 : le ZOH fige une avance au premier moment, le FOH n'en a pas.** Pour un mode diagonal stable (z = λΔ, w = −c/λ), gain continu 2 Re(w) en ZOH et en FOH ; le premier moment FOH égale celui du mode continu ; celui du ZOH en diffère de 2 Re(w φ(z)), φ(z) = 1/2 + z/12 − z³/720 + O(z⁵) ; divisé par un gain continu non nul, c'est une avance du retard de groupe à fréquence nulle, qui tend vers 0,5 échantillon pour un mode lent. Mode réel (z = −s, Re c non nul) : avance 1/s − 1/(eˢ − 1) dans ]0 ; 1/2[ échantillon ; retard pris quand Δ → Δ/r : κ_mode (1 − 1/r)/2, κ_mode dans ]0 ; 1[.
  - Preuve : T1, `THEORY.md` ; `theory/T1_hold_moments.md` ; contrôles symbolique et numérique `theory/t1_check.py` (vrai `DiagonalSSMLayer`, float64).
  - Statut : D (démontré, relecture Codex en deux passes). Portée : fréquence nulle seulement, une couche linéaire (pas le réseau), noyaux infinis et arithmétique exacte, gain continu non nul pour parler de retard ; la borne κ_mode < 1 ne vaut que pour les modes réels, alors que les modes entraînés sont surtout complexes.
- **Le plancher résiduel n'est ni le rééchantillonneur ni la précision.** À 96 kHz, rééchantillonnage haute qualité : le plancher monte de 3,7 à 5,8 % au lieu de baisser de moitié ; float64 de bout en bout : écart d'au plus 0,07 %.
  - Preuve : F-0027, E-0025 ; `demo/sr_transfer/predictions_floor.md` ; `demo/sr_transfer/floor_925.json`, `demo/sr_transfer/floor_736_688.json`.
  - Statut : préinscrit (explications R et P réfutées par leurs seuils écrits), 3 partitions, relecture FOH des checkpoints ZOH.
- **T4, forme du retard : le réseau ZOH rééchelonné prend un retard κ (1 − 48 000/fs′)/2 échantillon à 48 kHz.** Rapport κ(192 kHz)/κ(96 kHz) de 0,991 à 1,031, dans [0,8 ; 1,25] pour les 4 checkpoints ; κ Big Muff 2,525 et 2,529 (s925), 2,428 et 2,419 (s736), 1,920 et 1,903 (s688) ; VS100R 1,934 et 1,993 ; retard non corrigé de 0,48 à 0,95 échantillon à 48 kHz.
  - Preuve : T4, `THEORY.md` ; `demo/sr_transfer/predictions_kappa.md` (R1) ; `demo/sr_transfer/kappa.json` ; script `demo/sr_transfer/kappa.py`.
  - Statut : préinscrit, R1 tenue ; T4 au statut M (mesuré sous préinscription, sans preuve pour le réseau).
  - Conditions : ZOH, Δ rééchelonné sans correction, 96 et 192 kHz ; modèle contre modèle ; retard par bande d'octave (500 Hz à 8 kHz, Welch 4096, cohérence au moins 1/2) ; Big Muff : 3 checkpoints ; VS100R : 1 checkpoint (`vs100r_gc_film_joint_s925_r2`, FiLM sur gain et contour, 8 prises de test).
- **T4, accord avec l'estimation tirée de l'appareil (VS100R).** κ direct 1,964 (médiane des deux fréquences) contre l'estimation indirecte 1,86, écart 0,104 sous le seuil 0,3.
  - Preuve : T4 ; `demo/sr_transfer/predictions_kappa.md` (R2) ; `demo/sr_transfer/kappa.json`.
  - Statut : lecture R2 préinscrite et tenue ; l'estimation indirecte était post hoc (courbes ESR(décalage) contre l'appareil, un seul checkpoint).
- **T4, nature du retard : pur sur Big Muff, dispersif sur le VS100R.** Écart-type du retard entre bandes de 0,022 à 0,050 échantillon (Big Muff, seuil 0,1) ; VS100R 0,175 (96 kHz) et 0,265 (192 kHz), la bande de 500 Hz retardée de 0,144 contre 0,442 à 0,549 pour les autres (96 kHz).
  - Preuve : T4 (Portée) ; `demo/sr_transfer/predictions_kappa.md` (R4) ; `demo/sr_transfer/kappa.json`.
  - Statut : lecture R4 préinscrite ; un checkpoint par appareil côté VS100R.
- **T1-P1 : un modèle entraîné en FOH ne prend qu'un petit retard au transfert, sous le seuil préinscrit.** κ primaire 0,036 (médiane de 6 κ aux bandes 500, 1 000 et 2 000 Hz), verdict « soutenue » (soutenue si la valeur absolue est au plus 0,15, réfutée à partir de 0,5) ; les 6 κ de 0,031 à 0,043, les 18 κ par bande de 0,003 à 0,059 ; lecture secondaire 0,034. Contraste : ZOH aux mêmes graines, autres entraînements, κ de 1,9 à 2,6 aux mêmes bandes.
  - Preuve : T1 (T1-P1), `THEORY.md` ; `demo/sr_transfer/kappa_foh.json` ; script `demo/sr_transfer/kappa_foh.py` ; checkpoints `paired_ssm_wavenet_foh_s925`, `paired_ssm_wavenet_foh_s736`, `paired_ssm_wavenet_foh_s688`.
  - Statut : préinscrit dans `THEORY.md` avant le script, portes tenues (FOH chargé, ESR à 48 kHz égale au record, aller-retour de 2,6·10⁻⁴ à 4,3·10⁻⁴). Bandes 4 et 8 kHz sans lecture (0,041 à 0,157). Le signe n'est pas établi (voir En attente).
- **Entraîner en FOH : libellé préinscrit « inférieur » à 48 kHz.** d = ln(ESR_ZOH / ESR_FOH), même graine, partition et drapeaux : 0,105 (925), 0,187 (736), -1,169 (688) ; moyenne -0,292, sous la marge -0,140. Le libellé tient entièrement à la partition 688 : ESR de test FOH 0,13668 contre 0,04245 (ZOH), perte de validation 0,0345 contre 0,0295, donc pas un écart propre au test. Aux deux autres partitions, le FOH est plus bas (ESR 0,03947 contre 0,04385, 0,04148 contre 0,04999) ; leur moyenne 0,146 n'est pas la quantité préinscrite.
  - Preuve : F-0030, E-0036 ; `demo/sr_transfer/predictions_foh_training.md` (verdict) ; `demo/nablafx_bench/paired_ssm_wavenet_s688.json`, `demo/nablafx_bench/paired_ssm_wavenet_foh_s688.json` (et 925, 736).
  - Statut : préinscrit avant les runs, verdict relu par Codex (lecture seule). Une graine, trois partitions ; le bruit apparié entre deux règles de discrétisation n'est pas connu. Les jumelles ZOH ont tourné sous l'ancien pilote NVIDIA (écart déclaré avant les runs).
- **Un modèle entraîné en FOH se transfère sans correction.** ESR contre son propre modèle à 48 kHz de 2,6·10⁻⁴ à 3,2·10⁻⁴ à 96 kHz (seuil 5·10⁻⁴) ; de 2,4·10⁻⁴ à 4,4·10⁻⁴ sur 88 200 Hz, 96 kHz et 192 kHz (descriptif).
  - Preuve : E-0036 ; `demo/sr_transfer/predictions_foh_training.md` ; `demo/sr_transfer/sr_transfer_paired_ssm_wavenet_foh_s925_last_plain.json` (et s736, s688).
  - Statut : préinscrit, tenu sur les 3 runs. Ce critère compare chaque modèle à lui-même et ne dit rien de sa qualité.

## Préliminaire

- **À 44 100 Hz, ne pas rééchelonner Δ coûte bien plus que prévu.** ESR `naive` / `delta` de 5,3 à 7,0, contre 1,02 à 1,5 prévu par analogie avec les LSTM (Wilczek et al., DAFx22).
  - Preuve : E-0025 ; `demo/sr_transfer/RESULTS.md` (P3) ; `demo/sr_transfer/sr_transfer_paired_ssm_wavenet_s925_last.json` (et s736, s688).
  - Statut : préinscrit, P3 partielle sur les 3 partitions (sens tenu, ampleur réfutée).
- **L'écart de montée non corrigé croît avec fs′ et se trouve dans la bande audio.** 88 200 Hz < 96 kHz < 192 kHz pour les 3 partitions ; à 96 kHz, ESR de 0,0434 à 0,0774 contre 0,005 à 0,05 prévu. Sur l'intérieur des segments, l'ESR contre `around` à fs′ et l'ESR contre le modèle à 48 kHz dans la bande commune diffèrent d'au plus 2,3·10⁻³ en relatif (88 200 Hz à 192 kHz) : l'écart ne vient pas du contenu au-dessus de 24 kHz.
  - Preuve : E-0025 ; `demo/sr_transfer/RESULTS.md` (P5) ; `demo/sr_transfer/sr_transfer_paired_ssm_wavenet_s925_last.json`, `demo/sr_transfer/sr_transfer_paired_ssm_wavenet_s925_last_posthoc.json` (et s736, s688).
  - Statut : préinscrit, P5 partielle pour 925 et 736 (croissance tenue, niveau au-dessus du prévu), tenue pour 688 ; contrôle dans la bande commune post hoc (relecture Codex).
- **Loi en τ² de l'écart non corrigé, compatible avec un retard pur.** Rapport d'ESR 192/96 de 2,08 à 2,14 et 88/96 de 0,84 à 0,84, contre 2,25 et 0,83 pour un petit retard pur proportionnel à la jauge de T4 (rapport des carrés des jauges ; l'ESR exacte d'un retard pur dépend du spectre) ; jauges 0,228, 0,250 et 0,375 échantillon à 48 kHz pour fs′ = 88 200 Hz, 96 kHz et 192 kHz.
  - Preuve : E-0025 ; `demo/sr_transfer/predictions_hold.md` ; `demo/sr_transfer/sr_transfer_paired_ssm_wavenet_s925_last_holdtest.json` (et s736, s688) ; jauge de T4 évaluée en forme fermée par `paper/p3_sample_rate/numbers.py`.
  - Statut : post hoc, relevée après les résultats de `demo/sr_transfer/RESULTS.md` ; sa première confrontation préinscrite est H4 (section Retiré ou réfuté).
  - Descriptif post hoc, 2 segments écoutés, partition 925, 192 kHz : ESR contre le modèle à 48 kHz de 0,177 à 0,181, puis de 0,0057 à 0,0069 après compensation du retard fractionnaire de `demo/sr_transfer/kappa.json` (translation circulaire, bords compris) ; source `demo/sr_transfer/brightness.json`.
- **Le maintien à un coefficient explique la majeure partie de l'écart, pas tout.** Coefficient d'entrée ZOH remplacé par la porte de 48 kHz : réduction de ×2,67 à ×5,38 à 96 et 192 kHz ; ESR de 0,012 à 0,016 à 88 200 Hz et 96 kHz, contre 0,036 à 0,077 sans correction.
  - Preuve : E-0025 ; `demo/sr_transfer/RESULTS_hold.md` (H1) ; `demo/sr_transfer/sr_transfer_paired_ssm_wavenet_s925_last_holdtest.json` (et s736, s688).
  - Statut : préinscrit, H1 partielle sur les 3 partitions (seuil ×5, réfutée sous ×2). Limite déclarée avant les chiffres : somme de Riemann, gain continu faux pour les modes rapides.
- **Faire tourner le modèle à 2 ou 4 fois 44 100 Hz dégrade la fidélité à l'appareil.** Modèle à 88 200 Hz ou 176 400 Hz (Δ rééchelonné, sans correction) puis retour à 44 100 Hz : ESR 2,10 à 3,07 fois (×2) et 3,37 à 5,25 fois (×4) celle de `around`.
  - Preuve : E-0025 ; `demo/sr_transfer/RESULTS.md` (P8) ; `demo/sr_transfer/sr_transfer_paired_ssm_wavenet_s925_last.json` (et s736, s688).
  - Statut : préinscrit sans sens prédit (P8 sans objet), descriptif. Mesuré sans correction, avant la mise en évidence de l'avance du ZOH, qui domine cette ESR (`demo/sr_transfer/predictions_brightness.md`).
- **Contraste S4-TF-L-16 : la correction par mode ne le ramène pas au plancher.** Après correction à deux coefficients, ESR de 0,070 à 0,473 contre le modèle à 48 kHz ; réduction à 96 kHz de ×1,47 à ×1,87, à 192 kHz de ×0,72 (dégradation, s925) à ×1,29. Pôles repliés à 48 kHz : 12,4 à 13,0 % des modes ; TFiLM à blocs de 128 échantillons non corrigé. P9 : gain de `naive` à `delta` à 44 100 Hz de ×3,5 à ×3,6 (plus petit que SSM-WaveNet), écart à 96 kHz 2,04 à 3,38 fois celui de SSM-WaveNet.
  - Preuve : F-0026, E-0025 ; `demo/sr_transfer/RESULTS.md` (P9), `demo/sr_transfer/RESULTS_hold.md` (H5, H9) ; `demo/sr_transfer/sr_transfer_paired_s4_tf_l_16_s925_last_twotap.json`, `demo/sr_transfer/sr_transfer_paired_s4_tf_l_16_s925_last_holdtest.json` (et s736, s688).
  - Statut : préinscrit, verdicts mêlés. H5 (maintien à un coefficient, réduction prévue entre ×1,5 et ×5) : de ×0,97 à ×1,52 à 96 et 192 kHz, tenue seulement pour s736 à 96 kHz, réfutée ailleurs. H9 (réduction sous ×1,5 à 96 kHz) tenue pour s736, réfutée de peu pour s925 et s688 ; P9 partielle pour s925 et s736 (rapport sous 3), tenue pour s688.
- **Critère à 44 100 Hz des modèles entraînés en FOH : tenu pour 2 partitions sur 3.** ESR contre l'appareil du FOH entraîné sur celle de sa jumelle ZOH relue en FOH : 0,896 (925), 0,810 (736), 3,55 (688), seuil 1,15.
  - Preuve : E-0036 ; `demo/sr_transfer/predictions_foh_training.md` ; `demo/sr_transfer/sr_transfer_paired_ssm_wavenet_foh_s688_last_plain.json`, `demo/sr_transfer/sr_transfer_paired_ssm_wavenet_s688_last_asfoh.json` (et 925, 736).
  - Statut : préinscrit ; non tenu à 688, qui hérite de l'ESR élevée du run à 48 kHz.
- **Passer le signal intermédiaire d'une chaîne synthétique par 48 kHz ne change rien de pertinent en sortie.** Signal intermédiaire décimé à 48 kHz puis remonté, contre pleine bande à 192 kHz : médiane des 16 paires -73,2 dB sur les cordes pincées synthétiques (pire paire -68,3 dB) ; pire cas toutes excitations -42,1 dB (multisinus, deux écrêteurs durs), sous le seuil de pertinence de −30 dB fixé d'avance.
  - Preuve : `demo/chain_ultrasonic/RESULTS.md` ; `demo/chain_ultrasonic/predictions.md` ; `demo/chain_ultrasonic/results.json`.
  - Statut : préinscrit ; D1 réfutée vers le bas (section Retiré ou réfuté). Correction datée du 2026-09-24 (relecture Codex) : l'intervention mesurée comprend aussi l'altération du filtrage sous 24 kHz, elle n'isole pas l'ultrasonore. Systèmes Wiener/Hammerstein synthétiques définis en échantillons, cordes pincées à 8 harmoniques ; plancher de l'instrument -55,8 dB (multisinus) et -56,2 dB (bruit coloré) ; aucun appareil réel.

## Retiré ou réfuté

- **Bras `lead` (restaurer l'avance apprise en multipliant C par exp(λε)) : réfuté.** À 44 100 Hz, `lead` / `delta` de 1,25 à 1,44 (P4 : ±2 %) ; à 96 et 192 kHz, de 1,04 à 3,35 (P6 : baisse d'au moins 30 %), le bras aggrave l'écart.
  - Preuve : E-0025 ; `demo/sr_transfer/RESULTS.md` (P4, P6, relecture Codex point 3) ; `demo/sr_transfer/sr_transfer_paired_ssm_wavenet_s925_last.json` (et s736, s688).
  - Statut : préinscrit, réfuté sur les 3 partitions. Ce bras modifie les résidus modaux sans le terme d'entrée d'une vraie avance : l'échec réfute le bras, pas l'hypothèse d'un décalage.
- **H2 : le maintien à un coefficient dégrade la descente.** À 44 100 Hz, ESR de `delta_hold` supérieure de 5,6 à 15,9 % à celle de `delta` (prévu : ±2 %).
  - Preuve : E-0025 ; `demo/sr_transfer/RESULTS_hold.md` (H2) ; `demo/sr_transfer/sr_transfer_paired_ssm_wavenet_s925_last_holdtest.json` (et s736, s688).
  - Statut : préinscrit, réfuté sur les 3 partitions (somme de Riemann, modes rapides, limite déclarée avant les chiffres).
- **H4 : la correction à un coefficient ne change pas la nature de l'erreur.** Rapport 192/96 du résidu de 1,99 à 2,26, contre 2,25 pour la loi en τ² (prévu : nettement différent).
  - Preuve : E-0025 ; `demo/sr_transfer/RESULTS_hold.md` (H4) ; `demo/sr_transfer/sr_transfer_paired_ssm_wavenet_s925_last_holdtest.json` (et s736, s688).
  - Statut : préinscrit, « réfutée » dans `demo/sr_transfer/RESULTS_hold.md` ; la prédiction était conditionnelle (« si H1 tient ») et H1 n'est que partielle : à lire comme une description du résidu, qui garde la loi en τ² (un coefficient par mode ne peut garder à la fois le gain continu et l'avance).
- **H7 : le résidu de la correction à deux coefficients n'est pas une loi en τ⁴.** Rapport 192/96 de 1,19 à 1,36, contre au moins 3,5 prévu (5,1 pour τ⁴) : un plancher presque indépendant de la fréquence cible.
  - Preuve : E-0025 ; `demo/sr_transfer/RESULTS_hold.md` (H7) ; `demo/sr_transfer/sr_transfer_paired_ssm_wavenet_s925_last_twotap.json` (et s736, s688).
  - Statut : préinscrit, réfuté sur les 3 partitions ; le plancher reste non attribué (section En attente).
- **H8, et la lecture « `delta` transfère mieux que `around` à 44 100 Hz » : retirée.** À 44 100 Hz, ESR à deux coefficients / `delta` de 1,023 à 1,125 (prévu : au plus 2 % au-dessus), ce qui reproduit `around` / `delta` sur l'intérieur des segments (1,03 à 1,13).
  - Preuve : E-0025 ; `demo/sr_transfer/RESULTS_hold.md` (H8) ; `demo/sr_transfer/sr_transfer_paired_ssm_wavenet_s925_last_twotap.json`, `demo/sr_transfer/sr_transfer_paired_ssm_wavenet_s925_last_posthoc.json` (et s736, s688).
  - Statut : H8 préinscrite, réfutée ; lecture : la correction reproduit le modèle à 48 kHz, et l'avance de `delta` sur `around` venait d'un décalage de phase favorable face à l'appareil.
- **Critère d'identité de la relecture FOH à 48 kHz : non tenu tel qu'écrit.** ESR d'identité de 1,5·10⁻¹¹ à 4,0·10⁻¹¹, au-dessus du seuil 1·10⁻¹², inatteignable en float32 (seuil fixé sans regarder la précision de travail).
  - Preuve : `demo/sr_transfer/FIX_foh.md` (sections 1 et 6) ; `demo/sr_transfer/sr_transfer_paired_ssm_wavenet_s925_last_asfoh.json` (et s736, s688) ; témoin `demo/sr_transfer/asfoh_precision.py`.
  - Statut : préinscrit, non tenu ; seuil laissé tel quel. Les chiffres du témoin float32/float64 ne sont pas régénérés ici (section En attente).
- **Lecture « le plancher est du repliement non linéaire (N plutôt que Q) » : retirée.** À −20 dB, rapport V3/V2 de 0,19 à 0,24, sous le seuil 1/2 ; mais baisser le niveau appauvrit aussi l'aigu interne, ce qui fait baisser l'erreur de discrétisation Q : critère confondu. Critère « concentration dans l'aigu » non rempli non plus : 44 à 50 % de l'erreur au-dessus de 10 kHz, contre au moins 70 %. La conséquence produit « le modèle à 96 kHz est plus fidèle » est retirée avec elle.
  - Preuve : F-0027, E-0025 (mise à jour) ; `demo/sr_transfer/predictions_floor.md` (correction écrite avant la mesure complémentaire) ; `demo/sr_transfer/floor_925.json`, `demo/sr_transfer/floor_736_688.json`.
  - Statut : lecture des seuils préinscrits retirée après relecture ; mesure complémentaire préinscrite, non tranchée (section En attente).
- **T4-H2 : κ n'est pas une constante de l'architecture.** Médiane Big Muff 2,423 contre 1,964 pour le VS100R, écart 0,460 au-delà de ±20 % (0,393) ; sur le même appareil, les trois partitions Big Muff vont de 1,90 à 2,53.
  - Preuve : T4 (H2 réfutée), `THEORY.md` ; `demo/sr_transfer/predictions_kappa.md` (R3) ; `demo/sr_transfer/kappa.json`.
  - Statut : préinscrit (R3), réfuté. Une correction de retard doit se calculer par checkpoint.
- **R5, origine de κ par bloc : lecture non interprétable.** Le bras « bloc seul » fait tourner les 7 autres blocs à fs′ avec leur Δ de 48 kHz, donc trop vite en temps physique ; il ne mesure pas la jauge d'un bloc isolé. Aucune conclusion sur l'origine de κ.
  - Preuve : T4 (Preuve) ; `demo/sr_transfer/predictions_kappa.md` (R5) ; `demo/sr_transfer/kappa.json`.
  - Statut : préinscrit, défaut du protocole vu en lisant les chiffres ; seuils non modifiés.
- **D1 (l'ultrasonore d'un premier étage compte dans la bande audio) : réfutée vers le bas, pour l'intervention « passer par 48 kHz », qui comprend aussi le filtrage sous 24 kHz.** Toutes les paires sur cordes pincées sous −60 dB (pire -68,3 dB). D0 (plancher de l'instrument sous −60 dB) réfutée pour le multisinus (-55,8 dB) et le bruit coloré (-56,2 dB) ; sur deux sons, 15 paires non linéaires sur 16 sont sous le plancher (-62,5 dB) et la pire le dépasse de 0,92 dB : cette excitation ne mesure presque rien (le document source dit « sous le plancher » pour toutes).
  - Preuve : `demo/chain_ultrasonic/RESULTS.md` ; `demo/chain_ultrasonic/results.json`.
  - Statut : préinscrit (`demo/chain_ultrasonic/predictions.md`), réfuté ; systèmes synthétiques seulement.

## En attente

- **Attribution du plancher résiduel (repliement non linéaire N ou discrétisation dépendante du contenu Q).** Balayage à −40 et −60 dB : rapport ESR(−60) / ESR(−40) de 0,35 à 0,44, entre 1/5 et 1/2 : non départagé ; la part indépendante du niveau est au plus de 1,7 à 2,4 % du plancher à 0 dB.
  - Preuve : F-0027 ; `demo/sr_transfer/predictions_floor.md` ; `demo/sr_transfer/floor_levels.json`.
  - Statut : préinscrit, non tranché ; le Big Muff n'est peut-être pas linéaire à −60 dB. Il manque une cible native à 96 kHz ou plus, qui n'existe pas pour le Big Muff.
- **Origine de κ > 1 au niveau du réseau.** T1 borne κ_mode dans ]0 ; 1[ pour un mode réel ; les κ mesurés (1,90 à 2,53) exigent une cascade de couches, des modes complexes ou des signes mêlés, non départagés.
  - Preuve : T4 (Preuve : aucune pour le réseau), `THEORY.md` ; `demo/sr_transfer/kappa.json`.
  - Statut : ouvert ; une mesure isolant un bloc demanderait une nouvelle préinscription.
- **Signe du retard des modèles entraînés en FOH.** Les 18 κ par bande sont positifs sur ces segments (0,003 à 0,059), mais aucune incertitude n'est estimée.
  - Preuve : T1 (T1-P1, non prédit), `THEORY.md` ; `demo/sr_transfer/kappa_foh.json`.
  - Statut : non établi ; l'énoncé ne dit rien du signe.
- **Cause du run FOH à la partition 688.** Post hoc : décalage pur écarté (retard optimal 0 dans 10 segments sur 12, ESR intérieure réalignée 0,135 puis 0,133, contre 0,040 pour la jumelle ZOH) ; pôles trop rapides écartés (mesure sans fichier JSON, chiffre non cité).
  - Preuve : E-0036 ; `demo/sr_transfer/predictions_foh_training.md` (examen post hoc) ; `demo/sr_transfer/foh_lag.json`.
  - Statut : cause inconnue ; une trajectoire d'entraînement divergente sans mécanisme propre au FOH est compatible, non démontrée.
- **Brillance à 96 et 192 kHz : repliement retiré ou compensation perdue.** Lectures préinscrites R1 à R3 : « non tranché ». Descriptif : à 192 kHz les ZOH perdent de l'aigu (B de -0,56 à -0,18 dB sur 4 à 16 kHz), les FOH entraînés en gagnent (0,22 à 0,27 dB), entraînements différents ; part hors grille harmonique d'un sinus à 997 Hz en baisse de 10,6 à 31,4 dB (limites de l'instrument : la case 0 est exclue, écart déclaré après coup ; établissement du réseau non vérifié ; filtre de retour atténuant près de 24 kHz sur les seuls chemins à 96 et 192 kHz). Post hoc, tiers d'octave de 16 kHz, 192 − 48 kHz : ZOH de -1,35 à -0,06 dB, FOH de 0,57 à 0,86 dB ; le modèle à 48 kHz est plus proche du réel pour 5 checkpoints sur 6.
  - Preuve : `demo/sr_transfer/predictions_brightness.md` ; `demo/sr_transfer/brightness.json`.
  - Statut : préinscrit, non tranché ; lecture par bande post hoc. Décision produit de l'utilisateur (garder 48 kHz) : décision, pas résultat. Aucune écoute à l'aveugle faite.
- **Chiffres cités dans les documents mais non régénérés ici.**
  - Coût du moteur natif C++ (FOH) face à NAM A2 : seulement dans un tableau Markdown, `cpp/SSM_WAVENET_NATIVE.md` (E-0029, F-0027), sans fichier de mesure lisible ; à remesurer avec sortie JSON s'il est cité.
  - Écarts des contrôles numériques de T1 : imprimés par `theory/t1_check.py` et recopiés dans `THEORY.md`, sans fichier de sortie ; `numbers.py` ne relance pas la couche. T1 est cité par son statut.
  - Égalité des noyaux relecture FOH et deux coefficients en float64, et témoin float32/float64 de `demo/sr_transfer/asfoh_precision.py` : chiffres seulement dans `demo/sr_transfer/FIX_foh.md`.
- **Transfert sur un second appareil et fidélité au-dessus de 48 kHz.** La correction n'est mesurée que sur le Big Muff (3 checkpoints), dont la capture n'existe pas au-dessus de 44 100 Hz ; sur le VS100R, capturé nativement à 192 kHz, seul κ est mesuré (modèle contre modèle). Aucune fidélité à l'appareil après montée n'est donc mesurée.
  - Preuve : `demo/sr_transfer/predictions_kappa.md` (Ce que cela ne dira pas) ; `demo/sr_transfer/predictions_floor.md` ; E-0037 (capture VS100R).
  - Statut : non mesuré.
