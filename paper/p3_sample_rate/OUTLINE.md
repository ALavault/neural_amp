# P3 — Plan de l'article

- Titre de travail : Transfert de fréquence d'échantillonnage des modèles audio à espace d'états : avance du maintien d'ordre zéro et correction à deux coefficients.
- Dossier sans prose : l'auteur rédige le texte ; chiffres dans `numbers.json` (clés citées entre accents graves), résultats dans `RESULTS.md`, revendications dans `CLAIMS.md`.

## Cible

- Lieu : IEEE WASPAA 2027 (Workshop on Applications of Signal Processing to Audio and Acoustics), édition 2027.
- Date limite de soumission : vers fin avril 2027, « à confirmer » sur l'appel officiel.
- Format : 4 pages de contenu + 1 page de références seulement, gabarit IEEE deux colonnes.
- Relecture : double aveugle.
- Politique IA : usage d'outils d'IA déclaré dans les remerciements, selon la politique LLM de l'IEEE SPS ; le texte des sections est écrit par l'auteur, le dossier ne contient que faits, chiffres, plan et figures.
- Anonymisation : double aveugle alors que le dépôt GitHub est public et nommé ; ne citer dans la soumission ni le nom du dépôt, ni l'auteur, ni les identifiants de commit ; code et scripts de figures via un miroir anonyme ou en annexe après acceptation (à confirmer avec l'appel).
- Droits des données : ToneTwist AFx (Big Muff S050_V100), licence CC BY-NC, aucun audio brut dans le dépôt ; protocole, découpage et code de référence NablAFx (licence du code à vérifier, à confirmer) ; capture propre du Marshall Valvestate VS100R (session du 2026-09-26), audio brut hors du dépôt.
- Statut des résultats : exploratoire, D-PRODUCT-002 ; captures scellées non lues.
- Alternative 1 : ICASSP 2028, 4 pages + 1 page de références, relecture non aveugle, soumission mi ou fin septembre 2027 (à confirmer).
- Alternative 2 : DAFx27, 8 pages références comprises, double aveugle, soumission fin mars 2027 (à confirmer) ; place pour la descente, le contraste S4-TF-L-16 et la brillance.

## Thèse

- Le maintien d'ordre zéro (ZOH) fige à la fréquence d'entraînement un écart de premier moment 2 Re(w φ(z)) : rapporté à un gain continu non nul, un écart signé du retard de groupe à fréquence nulle, avance dans ]0 ; 1/2[ échantillon pour un mode réel, de signe libre pour un mode complexe, tendant vers un demi-échantillon quand Δ → 0 à λ et c fixés ; le maintien du premier ordre (FOH) n'en a pas (T1, démontré ; `t1_phi0`).
- Rééchelonner Δ transfère SSM-WaveNet de 48 kHz à 44 100 Hz sans perte face au rééchantillonnage (P1 et P2 préinscrites, 3 partitions ; `down_delta_around_ratio_max`).
- En montée (88 200 Hz à 192 kHz), l'écart au modèle à 48 kHz est compatible avec une contribution dominante de cet écart de premier moment : il ne s'effondre pas à −20 dB (critère H3 ; `hold_m20db_ratio_min`), il croît comme le carré du retard (lecture post hoc ; `up_delta_ratio_192_96_min`) ; la part du repliement n'est pas isolée.
- Une correction causale à deux coefficients par mode, égale à la relecture du checkpoint ZOH en FOH, ramène toutes les fréquences de montée à une ESR de `twotap_esr_min` à `twotap_esr_max` du modèle à 48 kHz, sans réentraînement (H6 ; F-0026, F-0027).
- Le plancher résiduel persiste avec le rééchantillonneur haute qualité et en float64 (R et P réfutées selon les critères préinscrits) ; son origine n'est pas attribuée.
- Entraîner directement en FOH est étiqueté inférieur à 48 kHz (E-0036, une partition sur trois en cause) : la voie retenue est la relecture FOH.
- Au niveau du réseau, le retard non corrigé suit κ (1 − 48 000/fs′)/2 avec κ propre au checkpoint (T4, mesuré ; `kappa_bm_min` à `kappa_bm_max`) ; un modèle entraîné en FOH ne prend qu'un petit retard, sous le seuil préinscrit, signe non établi (T1-P1 ; `fohk_primaire`).

## Plan et budget

| Section | Contenu | Pages | Figures/tableaux |
|---|---|---|---|
| 1 Introduction | Modèles d'effets audio à espace d'états, hôtes à 44 100 Hz à 192 kHz ; problème : un modèle entraîné à une fréquence ; contributions C2 à C7 de `CLAIMS.md` | 1/2 | aucun |
| 2 Moments du maintien (T1) | Noyaux ZOH et FOH d'un mode diagonal ; gain continu commun ; écart de premier moment 2 Re(w φ(z)), avance du retard de groupe pour un gain non nul ; mode réel : avance dans ]0 ; 1/2[, κ_mode dans ]0 ; 1[ | 3/4 | `figures/fig_zoh_advance.pdf` (panneau du haut) |
| 3 Dispositif | SSM-WaveNet, Big Muff ToneTwist et protocole NablAFx, 3 partitions ; bras `delta`, `around`, `naive`, maintien à un coefficient, deux coefficients, relecture FOH ; mesures (descente contre l'appareil, montée dans la bande commune contre le modèle à 48 kHz, intérieur des segments) ; préinscriptions | 1/2 | Tableau 1 (bras et mesures) |
| 4 Descente et montée sans correction | P1, P2, P7 ; P3 et P5 partielles ; loi en τ² post hoc ; H3 | 1/2 | `figures/fig_esr_vs_rate.pdf` (courbes ZOH seules) |
| 5 Correction à deux coefficients = relecture FOH | Construction (gain continu et premier moment du mode à 48 kHz) ; équivalence avec la relecture FOH ; H6, H7 réfutée, H8 ; plancher : R et P réfutées, attribution ouverte | 3/4 | `figures/fig_esr_vs_rate.pdf` ; Tableau 2 (réductions et plancher) |
| 6 Retard du réseau (T4) et modèles entraînés en FOH | Forme κ (1 − 48 000/fs′)/2, κ par checkpoint, H2 réfutée ; T1-P1 ; non-infériorité FOH étiquetée inférieure (E-0036) | 1/2 | `figures/fig_zoh_advance.pdf` (panneau du bas) |
| 7 Limites | S4-TF-L-16 (TFiLM, pôles repliés) ; un appareil pour la correction ; cohérence, pas fidélité ; plancher non attribué | 1/4 | `figures/fig_floor_levels.pdf` (si la place le permet) |
| 8 Conclusion | Règle pratique aux fréquences testées : Δ seul à 44 100 Hz, relecture FOH à 88 200 Hz, 96 kHz et 192 kHz ; ouverts : κ > 1, plancher | 1/4 | aucun |

## Figures et tableaux

| Élément | Générateur | Clés de numbers.json | Contenu |
|---|---|---|---|
| `figures/fig_esr_vs_rate.pdf` | `figures.py`, fonction `esr_vs_rate` | `up_delta_esr_s<partition>_<fs>`, `hold_esr_s<partition>_<fs>`, `twotap_esr_s<partition>_<fs>`, `fohtrain_plain_esr_s<partition>_<fs>` (partition 925, 736, 688 ; fs 88, 96, 192), `t4_jauge_88`, `t4_jauge_96`, `t4_jauge_192` | ESR contre le modèle à 48 kHz selon la fréquence cible, échelle log : ZOH sans correction, maintien à un coefficient, deux coefficients, entraîné en FOH ; guide en jauge² ancré à 96 kHz (post hoc) |
| `figures/fig_zoh_advance.pdf` | `figures.py`, fonction `zoh_advance` | `kappa_bm_s925_96`, `kappa_bm_s925_192`, `kappa_bm_s736_96`, `kappa_bm_s736_192`, `kappa_bm_s688_96`, `kappa_bm_s688_192`, `kappa_vs100r`, `fohk_primaire` ; formes fermées de T1 calculées dans la fonction | Haut : avance ZOH 1/s − 1/(eˢ − 1) d'un mode réel et avance FOH nulle ; bas : κ_mode à r = 2 et 4, borne 1, κ mesurés du réseau (ZOH Big Muff, VS100R, FOH) |
| `figures/fig_floor_levels.pdf` | `figures.py`, fonction `floor_levels` | `floor_esr_s<partition>_m<niveau>` (niveau 0, 20, 40, 60 dB sous le niveau du test) | ESR du plancher à 96 kHz selon le niveau d'entrée, avec les deux comportements de référence (plateau, distorsion faible) ; figure optionnelle |
| Tableau 1 | à composer depuis `demo/sr_transfer/predictions.md` et `demo/sr_transfer/predictions_hold.md` | aucune | Bras, référence de comparaison, fréquences, préinscription |
| Tableau 2 | à composer | `twotap_reduction_min`, `twotap_reduction_max`, `hold_reduction_min`, `hold_reduction_max`, `twotap_ratio_192_96_min`, `twotap_ratio_192_96_max`, `asfoh_ecart_rel_max`, `floor_v1_v0_hausse_pct_max`, `floor_v2_v1_ecart_pct_max`, `floor_m60_m40_ratio_min`, `floor_m60_m40_ratio_max`, `s4_twotap_esr_min`, `s4_twotap_esr_max` | Réductions par correction, rapport 192/96, équivalence relecture FOH, contrôles du plancher, contraste S4-TF-L-16 |

## Théorie

| Énoncé | Statut | Rôle dans l'article |
|---|---|---|
| T1 | D (2026-10-02) | Section 2 : dit quelle quantité la correction doit annuler (l'avance au premier moment) et pourquoi la relecture FOH l'annule à fréquence nulle ; borne κ_mode < 1 d'un mode réel, contraste avec les κ mesurés |
| T4 | M (2026-10-02) pour la forme | Section 6 : forme du retard du réseau mesurée (R1), κ par checkpoint (H2 réfutée) ; aucune preuve pour le réseau, présenté comme mesure |

- T5 (conditionnement des pôles) non retenu : hors du sujet du transfert de fréquence.
- À développer : l'équivalence « relecture FOH = correction à deux coefficients » (argument analytique de `demo/sr_transfer/FIX_foh.md`, section 3, test `tests/numerical/test_ssm_discretization.py`) n'est pas inscrite dans `THEORY.md` ; l'inscrire, avec relecture indépendante, avant de la présenter comme proposition.
- À développer : κ > 1 au niveau du réseau ; T1 borne un mode réel sous 1, l'excès exige une cascade de couches, des modes complexes ou des signes mêlés, non départagés.
- À développer : plancher résiduel non attribué (repliement non linéaire N contre discrétisation dépendante du contenu Q).
- À développer : T1 ne porte que sur la fréquence nulle ; les retards par bande d'octave mesurés (500 Hz à 8 kHz) n'en découlent pas directement.
- À développer : une preuve, même partielle, de la forme (1 − T′/T) pour un réseau de couches linéaires en cascade ferait passer T4 de M à E.

## Travaux liés

- Voir `RELATED.md` du même dossier (rédigé séparément, avec `refs.bib`) : indépendance à la fréquence d'échantillonnage des modèles neuronaux d'effets, discrétisation des modèles à espace d'états (ZOH, bilinéaire, FOH), rééchantillonnage autour du modèle.

## Manques avant soumission

- Comparaison avec les méthodes publiées d'indépendance à la fréquence (voir `RELATED.md`) : au minimum discussion bibliographique (pas de nouvelle expérience) ; idéalement une méthode de référence rejouée sur les mêmes checkpoints, nouvelle expérience CPU préinscrite, quelques jours.
- Second appareil pour la correction : relecture FOH et montée du checkpoint VS100R existant, comparées à la capture native à 192 kHz ; nouvelle expérience CPU préinscrite, environ un jour ; transformerait la cohérence en fidélité, sous réserve de l'avance d'environ un échantillon du modèle VS100R (Q-0013).
- Fidélité au-dessus de 48 kHz sur le Big Muff : impossible sans cible native ; à déclarer comme limite.
- Variabilité selon la graine : les 3 checkpoints sont 3 partitions à graine fixe ; plusieurs graines par partition demanderaient de nouveaux entraînements GPU (file GPU occupée).
- Entraînement en FOH : davantage de partitions ou de graines pour trancher la non-infériorité et la cause du run à 688 ; nouvelle expérience GPU préinscrite.
- Incertitude sur κ (rééchantillonnage par segments ou par prises) : nouvelle mesure CPU préinscrite, faible coût ; nécessaire pour parler du signe du κ FOH.
- Origine de κ > 1 : mesure isolant un bloc (réseau à 48 kHz, un seul bloc à fs′ avec rééchantillonnage autour), nouvelle préinscription, CPU ; ou décomposition théorique (T4 vers E).
- Équivalence relecture FOH : inscription dans `THEORY.md` et relecture indépendante, travail théorique sans expérience.
- Écoute à l'aveugle 48 kHz contre montée corrigée : session d'écoute de l'utilisateur ; sans elle, aucune formulation perceptive.
- Coût du moteur natif, si cité : remesure avec sortie JSON (banc existant `cpp/benchmarks/ssm_wavenet_benchmark.cpp`), CPU, une heure.
- Anonymisation : miroir anonyme du code des figures et de `numbers.py`, sans nom de dépôt ; quelques heures.
- Témoins de mécanisme (relecture Codex) : correction du seul retard global appliquée à tout le test, pour situer l'apport propre de la correction par mode ; contrôle linéarisé ou synthétique séparant erreur de maintien, repliement des pôles et repliement des non-linéarités ; pour S4-TF-L-16, interventions séparées sur la durée physique du TFiLM et sur les modes repliés ; nouvelles expériences CPU préinscrites.
- Unités statistiques : le test Big Muff est fait de 12 segments de 5 s d'un seul fichier, pas de prises indépendantes ; les plages min–max décrivent 3 checkpoints, ce ne sont pas des intervalles de confiance ; l'étiquette FOH « inférieur » est une règle préinscrite, pas un test d'infériorité ; à dire dans le texte.
- Reproductibilité pour un tiers : accès documenté aux poids des checkpoints, aux données et à l'environnement, séquence de reproduction de chaque bras ; le miroir anonyme doit couvrir ces dépendances, pas seulement les figures.
- `THEORY.md`, T1 : la formulation « en avance de 2 Re(w φ(z)) » est signée ; préciser que le signe et la borne ne valent que pour un mode réel (travail sur le registre, hors de ce dossier).
- Vérifier sur les appels officiels la date limite, le format et le modèle de relecture de WASPAA 2027 (à confirmer).
