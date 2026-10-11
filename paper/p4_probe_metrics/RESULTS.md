<!-- Généré par paper/render.py depuis RESULTS.src.md. -->

# P4 : sondes d'identification comme métriques de fidélité (résultats)

<!-- Dossier sans prose. Chaque nombre décimal est une clé de numbers.json entre doubles accolades, régénérée par numbers.py depuis diagnosis/probe_metrics/ (fichiers suivis par git). Rendu : uv run python paper/render.py paper/p4_probe_metrics. -->
<!-- Classement (D-PRODUCT-002, étiquette de publication « exploratoire ») : Établi = règle de lecture ou attente commitée avant les chiffres (diagnosis/probe_metrics/NOTE.md), mesurée sur le test ou la séance réelle ; Préliminaire = lectures post hoc (« lectures ajoutées », gap_checks.py, baselines_extra.py, AUC), seuils choisis après les données, lectures réelles sans indépendance A/B établie ; Retiré ou réfuté = attente préinscrite non tenue, outil écarté, lecture retirée ; En attente = non mesuré ou non régénérable. -->
<!-- Vocabulaire (D-0010) : « score d'écart » ou « pseudo-distance », jamais « distance » ; les sondes sont des instruments d'analyse, jamais des émulateurs. -->

## Établi

- Protocole gelé avant toute lecture d'un défaut, choisi sur la seule capacité d'extraction de la référence statique (aucune détection lue).
  - FIR et Hammerstein 3 et 5 : 256 taps ; ridge 1·10⁻² (FIR), 1·10⁻⁴ (Hammerstein). Volterra–Laguerre 2 et 3 : 8 filtres de Laguerre gelés, pôles 0,80 et 0,90, ridge 1·10⁻⁴.
  - Capacité r_ref / RMS(y_ref) sur la lecture B du développement : FIR 0,094, H3 0,062, H5 0,062, VL2 0,220, VL3 0,119.
  - Limite consignée : taps au bord de la grille préinscrite, non étendue après coup (FIR 0,131 à 128 taps).
  - Preuve : `diagnosis/probe_metrics/NOTE.md` (Protocole gelé v1 et v2), `diagnosis/probe_metrics/dev_select.json`, `diagnosis/probe_metrics/dev_select_v2.json` ; commits 26a1904 et b800268 ; E-0048.
  - Statut : préinscrit (règles de choix écrites avant les chiffres).
  - Conditions : 48 kHz, référence DSP statique, bruit coloré et sinus glissant synchronisé, 3 niveaux, graines 100 à 102, lecture sur la graine + 50.
- Borne |d_audio − d_probe| ≤ r_ref + r_eval : 0 violation sur 9560 lignes d'étalonnage (de 120 lignes pour WH à 1920 pour FIR et Hammerstein), 0 sur 138 lignes de sonde VS100R, 0 sur 150 lignes JVM410H.
  - Preuve : `diagnosis/probe_metrics/report_numbers.json` (bound), `diagnosis/probe_metrics/real_pairs_vs100r.csv`, `diagnosis/probe_metrics/real_pairs_jvm.csv`.
  - Statut : préinscrit (NOTE, attendu 0). Inégalité triangulaire sur B, vraie sur ces données ; elle ne fait pas du score une distance.
- Équivalence linéaire en paramètres fit(y_eval) − fit(y_ref) = fit(y_eval − y_ref) et égalité d_probe² = Δθᵀ G_B Δθ (G_B : Gram des caractéristiques de lecture), vérifiées numériquement ; contre-exemple F = x + a·x³, G = x + 2a·x³ : erreurs de fit égales, score d'écart porté par la seule branche d'ordre 3.
  - Preuve : `tests/unit/test_probe_metrics.py` (test_linear_equivalence_and_gram_norm, test_same_fit_error_but_different_systems_are_separated).
  - Statut : préinscrit (NOTE, contrôles mathématiques). Hypothèse : normalisation et régularisation fonctions de x_A seulement. Aucun chiffre de résultat archivé (tolérances du test seulement).
- Emboîtement Hammerstein 5 / Volterra–Laguerre 3 absent dans les deux sens : part d'une colonne xᵏ retardée hors de l'espace Volterra–Laguerre 3 de 0,13 à 0,96 ; part médiane d'une colonne Volterra hors de l'espace Hammerstein 5 0,66 (max 0,94).
  - Conséquence : un écart de score entre ces deux sondes ne se lit pas comme un écart de capacité.
  - Preuve : `diagnosis/probe_metrics/dev_select_v2.json` (nesting) ; commit b800268.
  - Statut : préinscrit (règle v2 : mesuré à part, jamais supposé).
  - Conditions : bruit coloré, graine 100, 32 768 échantillons, moindres carrés sans régularisation.
- Détection à sévérité maximale et niveau médian (une cellule = un défaut × une famille ; « détecté » si R = score / plancher ≥ 10, « non détecté » si R < 3) : FIR 76 détectés, 2 limites, 0 non détecté ; H3 76 / 1 / 1 ; H5 76 / 1 / 1 ; VL3 77 / 1 / 0 ; ESR 72 / 1 / 5 ; MR-STFT 8 / 7 / 63 ; IMD (deux tons) 13 / 0 / 0 ; signature harmonique totale (sinus glissant) 12 / 1 / 0.
  - Lecture du rapport : les sondes ne représentent pas le bruit blanc, leur plancher est très bas et elles « détectent » presque tout ; le seuil seul ne les départage pas.
  - Preuve : `diagnosis/probe_metrics/report_numbers.json` (readings_strongest_middle), `diagnosis/probe_metrics/table_test.csv`.
  - Statut : règle préinscrite (NOTE, lecture fixée avant les chiffres de test) ; seuil d'instrument, pas d'audibilité.
  - Conditions : 13 défauts DSP connus × 3 sévérités, 6 familles d'excitation × 3 niveaux × 3 graines (200 à 202 ; DI : une réalisation), plancher = même chaîne mesurée deux fois avec un bruit indépendant à −80 dB.
- Sensibilité à la plus faible sévérité (niveau médian, familles détectées sur 6) : cubique préfiltré vu par FIR 6/6, H3 6/6, H5 5/6, VL2 6/6, VL3 6/6, contre ESR 0/6 ; quadratique : ESR 3/6, sondes 6/6 ; terme u⁷ (au-delà du degré) vu par presque aucune métrique (FIR 2/6, H3 1/6, ESR 0/6).
  - Mécanisme avancé par le rapport (non testé à part) : la sonde moyenne sur toute l'estimation et rejette ce qui sort de son espace, comme une détection synchrone.
  - Preuve : `diagnosis/probe_metrics/report_numbers.json` (weakest_detected).
  - Statut : règle préinscrite.
- Signature harmonique par ordre, spécifique : à sévérité maximale sur le sinus glissant synchronisé, ordres détectés : quadratique 2, cubique 1 et 3, terme croisé u[n]·u[n−5] 2 ; quasi déterministe d'une réalisation à l'autre (90e centile de l'écart-type du log du total 1·10⁻⁴).
  - Preuve : `diagnosis/probe_metrics/table_test.csv`, `diagnosis/probe_metrics/report_numbers.json` (repeatability).
  - Statut : métrique et règle préinscrites.
  - Réserve : sinus glissants de test identiques bit à bit à ceux du développement (durée tirée de graine % 3) ; exige un sinus glissant.
- Répétabilité entre graines (écart-type du log du score, médiane sur les cellules) : de 0,0016 (ESR) à 0,0059 pour ESR, MR-STFT et les sondes FIR, H3, H5, VL2, VL3 ; IMD 0,025 ; WH 0,034. 90e centiles : 0,07 à 0,22 pour ces métriques, IMD 0,31, WH 0,77, ordres harmoniques jusqu'à 0,47.
  - Preuve : `diagnosis/probe_metrics/report_numbers.json` (repeatability).
  - Statut : lecture préinscrite (NOTE). Bruit additif exclu du calcul. Magnitude, phase et enveloppe, ajoutées post hoc, sont rapportées en préliminaire.
  - Réserve : sur les sinus glissants, la stabilité ne vaut pas généralisation à de nouvelles réalisations.
- Détecteur ad hoc de dynamique d'enveloppe E2 : les quatre critères préinscrits sont remplis aux 3 niveaux sur 3 (−12, −9, −6 dBFS).
  - Monotonie : relâchement 60 → 100 → 200 ms : 0,12 → 0,54 → 1,28 dB à −12 dBFS ; 0,20 → 0,88 → 2,06 dB à −6 dBFS.
  - Sensibilité : relâchement 60 ms bien au-dessus de 3 × le plancher (5,4·10⁻⁵ à 7,6·10⁻⁵ dB).
  - Spécificité : pire autre défaut, co-répondants exclus par construction : coloration (0,05 dB), 25 à 40 fois sous le relâchement 200 ms ; gain et polynômes quasi nuls.
  - Contrôle interne : indice d'attaque du défaut de relâchement = 0,06 à 0,09 de son indice de relâchement, à toutes les sévérités (d'abord codé à 200 ms seul, étendu comme préinscrit après relecture Codex).
  - Co-répondants déclarés d'avance et rapportés : écho 0,66 à 0,81 dB, préfiltre 0,23 à 0,27 dB ; l'indice n'est pas exclusif d'un écho.
  - Preuve : `diagnosis/probe_metrics/dynamics_verdict.json`, `src/fssr_nam/product/dynamics.py`, `diagnosis/probe_metrics/NOTE.md` (préinscription E2) ; commits 20af272, 4555f06, ab8ce38 ; E-0049, F-0036.
  - Statut : préinscrit avant implémentation.
  - Conditions : chaîne dynamique synthétique, 13 défauts appliqués à cette chaîne, sauts de niveau sur bruit coloré (22 sauts fort → faible, 21 faible → fort), graines 200 à 202 ; plancher trivial (bruit à −80 dB moyenné sur les sauts) ; aucune validation sur matériel.
- VS100R, contrôles négatifs (basses, médiums, aigus réglés avant l'étage de tonalité) sous 3 planchers réels pour toutes les sondes : H3 médiane 0,13 (max 1,23) ; ESR 0,26 (max 3,73, seul dépassement, sur une section à glissement d'entrée) ; MR-STFT 0,57 (max 1,32).
  - Preuve : `diagnosis/probe_metrics/report_numbers.json` (real_vs100r), `diagnosis/probe_metrics/real_pairs_vs100r.csv`.
  - Statut : attente préinscrite (NOTE, paires réelles), tenue pour les sondes.
  - Conditions : envoi FX VS100R, séance unique, sans modèle neuronal, `di_test` (sources indépendantes de `di_train`), plancher réel = médiane des trois répétitions du centre (prises 1, 20, 39), ESR lue en amplitude.
- Coût d'une paire (4 s d'estimation + 4 s de lecture, 48 kHz) : ESR 8,5·10⁻⁴ s, MR-STFT 0,080 s, FIR 0,11 s, IMD 0,16 s, H3 0,39 s, H5 0,68 s, signature harmonique 0,81 s, VL2 1,4 s, VL3 6,1 s ; WH 32 s pour 1 s d'estimation.
  - Preuve : `diagnosis/probe_metrics/cost.json`.
  - Statut : mesure préinscrite (NOTE : coût mesuré à part, condition fixe).
  - Conditions : machine partagée, charge 15 à 18 sur 32 cœurs, CPU.

## Préliminaire

- Part captée d_probe / d_audio et part non représentée décrivent la couverture des défauts connus par chaque sonde et suggèrent des hypothèses diagnostiques ; aucun critère de classification ni taux de confusion n'est établi (sévérité maximale, niveau médian, médiane sur les 6 familles) : défauts linéaires et statiques presque entièrement captés par Hammerstein (gain 1,00, retard 1,00, passe-tout 1,00) ; FIR ne voit les non-linéarités que par la réponse linéaire équivalente (cubique 0,79, quadratique 0,64, croisé 0,45) ; bruit additif à peine capté par aucune sonde (0,01 à 0,08) ; écho à 30 ms hors mémoire non représenté (0,99 à 1,00) ; terme u⁷ part non représentée 0,52 à 0,95.
  - Attentes descriptives écrites avant les mesures (NOTE), sans seuil chiffré : bruit additif peu capté, soutenue ; défauts hors capacité faiblement scorés, nuancée, Hammerstein 5 capte 0,98 du terme u⁷ (degré 7) ; « Hammerstein voit peu les constantes de temps », non tenue sur le relâchement (parts captées 0,97 et 0,97 pour H3 et H5) ; relecture Codex du 2026-10-11.
  - Robustesse au bruit de mesure qui en découle : le bruit additif n'entre presque pas dans les scores de sonde, alors que l'ESR le compte entièrement.
  - Part captée et part non représentée ne s'additionnent pas (aucune orthogonalité hors échantillon).
  - Preuve : `diagnosis/probe_metrics/report_numbers.json` (captured_strongest_middle), `diagnosis/probe_metrics/table_test.csv` ; F-0033 ; figure `figures/captured_share.pdf`.
  - Statut : post hoc. La part captée est une lecture ajoutée (REPORT) ; la part non représentée figure parmi les lectures descriptives de NOTE mais REPORT la liste aussi comme lecture ajoutée.
- Volterra–Laguerre 3 représente la non-linéarité préfiltrée et les termes croisés que Hammerstein laisse en partie hors de son espace : part non représentée VL3 0,10 (cubique), 0,11 (quadratique), 0,09 (croisé u[n]·u[n−5]) contre H3 0,30, 0,42, 0,58 ; Hammerstein (256 taps) garde mieux la mémoire linéaire longue (retard : H3 0,06, VL3 0,31 ; passe-tout : H3 0,06, VL3 0,28).
  - Sur excitation corrélée (bruit coloré, plucks), Hammerstein 3 attribue le terme croisé à sa branche d'ordre 2 (0,96 à 0,98 du score, contre 0,87 à 0,97 pour le quadratique) : sa décomposition par branche ne distingue pas « carré » et « produit retardé ».
  - Preuve : `diagnosis/probe_metrics/report_numbers.json`, `diagnosis/probe_metrics/decomposition_noise.json`, `diagnosis/probe_metrics/decomposition_plucks.json` ; F-0033.
  - Statut : post hoc ; emboîtement absent (voir Établi) : complémentarité, pas classement de capacité.
- Dépendance au contenu de l'excitation (sévérité maximale, niveau médian, plage sur les 5 sondes) : écho capté de 0,28 à 0,51 sur plucks, 0,14 à 0,25 sur DI, 0,06 à 0,10 sur bruit coloré, au moins 0,99 sur deux tons ; relâchement d'enveloppe capté de 0,56 à 0,69 sur DI contre 0,95 à 0,97 sur bruit stationnaire : ce manque n'apparaît qu'avec une excitation réaliste.
  - Preuve : `diagnosis/probe_metrics/table_test.csv`.
  - Statut : post hoc. REPORT cite pour la DI une plage plus étroite, sans préciser sondes ni niveau : à réconcilier.
- Score de sonde au-dessus de d_audio sur la DI pour les termes pairs (H3 et H5, quadratique et croisé) : part captée 1,09 à 1,14 ; hors échantillon, une branche peut extrapoler ; écart surestimé, cas de refus de conclure.
  - Preuve : `diagnosis/probe_metrics/table_test.csv`.
  - Statut : post hoc.
- Signal de non-identifiabilité : somme des contributions par branche de Hammerstein 5 rapportée au score de 1,2 à 1,8 pour les défauts linéaires et statiques, de 3,7 à 8,7 pour le bruit et l'écho ; seuil 2,5 au-delà duquel les branches « se compensent ».
  - Preuve : `diagnosis/probe_metrics/decomposition_noise.json`, `diagnosis/probe_metrics/decomposition_plucks.json`, `src/fssr_nam/product/fidelity.py` (CANCELLING).
  - Statut : exploratoire ; seuil choisi après l'étalonnage, à valider sur d'autres données.
  - Conditions : bruit coloré −20 dBFS et plucks −10 dBFS, graine 200 (lecture 250), sévérité maximale.
- Métriques existantes du dépôt au banc : magnitude aveugle au passe-tout et au retard (R 1,6 et 5,5) ; phase exactement nulle pour un gain (score au plus 2,1·10⁻¹⁵) et aveugle au cubique (R 1,5) ; ensemble, elles séparent ce que la MR-STFT confond.
  - Répétabilité entre graines de ces trois métriques : médianes de 0,0035 à 0,011 (enveloppe), 90e centiles de 0,10 à 0,26 (phase, nulle pour un gain pur).
  - Preuve : `diagnosis/probe_metrics/table_test.csv`, `diagnosis/probe_metrics/baselines_extra.py`, `diagnosis/probe_metrics/report_numbers.json` (repeatability).
  - Statut : post hoc (mêmes règles de lecture que le banc). Médianes sur les familles, sévérité maximale, niveau médian.
- VS100R, séparation des changements réels (gain 0 exclu) et des contrôles négatifs, AUC avec p exact unilatéral, sur `di_test` : toutes paires, H3 0,98 (p = 8,0·10⁻⁵, 7 changements contre 12 contrôles), ESR 0,89 (p = 1,9·10⁻³), MR-STFT 0,93 ; entrée commune seulement, H3 0,98 (p = 6,7·10⁻⁴, 5 contre 10), ESR 0,98 (p = 6,7·10⁻⁴), MR-STFT 0,92 (p = 4,0·10⁻³) ; bruit rose, ESR 0,99, H3 0,98.
  - Lecture : à entrée commune, ESR et sondes séparent pareillement ; l'avantage apparent des sondes sur toutes les paires vient des sections à glissement d'acquisition.
  - Preuve : `diagnosis/probe_metrics/report_numbers.json` (real_vs100r), `diagnosis/probe_metrics/real_pairs_vs100r.csv` ; F-0034 ; figure `figures/vs100r_separation.pdf`.
  - Statut : post hoc (AUC non préinscrites ; comparaisons directes restreintes à l'entrée commune après relecture Codex).
  - Conditions : 7 changements réels hors gain 0 et 12 contrôles négatifs ; une seule répétition à entrée commune sur `di_test`.
- Robustesse aux défauts d'acquisition : sur le contrôle négatif « aigus 10 », dont la section `di_test` contient un glissement, ESR à 3,73 planchers, MR-STFT 0,51, H3 0,11 ; chaque sonde est ajustée sur l'entrée enregistrée de sa propre prise.
  - Preuve : `diagnosis/probe_metrics/real_pairs_vs100r.csv`, `tests/unit/test_probe_metrics.py` (test_own_input_fit_removes_an_input_mismatch).
  - Statut : post hoc ; un seul cas. La procédure d'ajustement sur l'entrée propre a été fixée avant tout score (écart de procédure daté dans NOTE).
- Plancher réel dominé par la dérive au cours de la séance : les deux répétitions avec la prise 1 valent 4,3 à 5,1 fois la répétition 20–39 (H3, `di_test`).
  - Preuve : `diagnosis/probe_metrics/real_pairs_vs100r.csv` ; F-0034.
  - Statut : lecture post hoc ; conséquence : un plancher fait de répétitions éloignées surestime le bruit d'une comparaison rapprochée.
- Métriques existantes sur le réel, entrée commune : `di_test` (5 changements, 10 contrôles) magnitude 0,98 (p = 6,7·10⁻⁴), phase 0,94 (p = 2,3·10⁻³), enveloppe 0,94 (p = 2,3·10⁻³) ; bruit rose (7 contre 12) magnitude 0,98 (p = 7,9·10⁻⁵), phase 1,00 (p = 2,0·10⁻⁵), enveloppe 0,90 (p = 1,3·10⁻³).
  - Preuve : `diagnosis/probe_metrics/real_extra_vs100r.csv`.
  - Statut : post hoc.
- Sinus glissant −3 dBFS : la signature harmonique totale ne sépare plus (AUC 0,63, p = 0,19, contre 0,99 à −10 dBFS, p = 4,0·10⁻⁵) ; un glissement d'acquisition dans le sinus −3 dBFS de la prise 20, référence de la plupart des paires, en est un facteur établi : écart d'ordre 3 entre répétitions 0,007 à −10 dBFS, 0,21 à −3 dBFS.
  - Contribution de la saturation non exclue : l'écart entre réglages (gain 7,5 contre centre, ordres impairs 3 à 9) baisse aussi, de 0,12 à 0,13 vers 0,07 à 0,10.
  - Sans la section touchée : AUC 0,88 (p = 0,13, 4 changements contre 2 contrôles seulement).
  - Preuve : `diagnosis/probe_metrics/gap_checks_sweep.json`, `diagnosis/probe_metrics/real_pairs_vs100r.csv`, `diagnosis/probe_metrics/report_numbers.json`.
  - Statut : post hoc (`diagnosis/probe_metrics/gap_checks.py`).
- Garde d'intégrité des sinus glissants (alignement global, retards par fenêtres comparés au profil médian de la séance, rejet dès que l'étendue max − min de ces écarts atteint 12 échantillons, une seule fenêtre aberrante suffit ; `diagnosis/probe_metrics/real_pairs.py`) : 2 sections signalées sur la séance (prise 20 à −3 dBFS, prise 38 à −20 dBFS) ; sur une grille de 13 glissements de 24 échantillons insérés à −10 dBFS, les glissements non détectés coûtent au plus 0,39 du plancher de chaque ordre en début de balayage et 0,64 en fin, les glissements détectés jusqu'à 100 planchers.
  - Preuve : `diagnosis/probe_metrics/gap_checks_slip_harm.json`, `diagnosis/probe_metrics/sweep_integrity.json`, `diagnosis/probe_metrics/real_pairs.py` ; commits 9d56ed6, c50cc50, 5a7df99.
  - Statut : outil post hoc, réécrit après deux relectures Codex ; pas une borne générale (positions testées seulement).
- JVM410H OD1 (ToneTwist), monotonie observée : sur 40 échelles par sortie (4 boutons × 2 côtés × 5 métriques), le score croît strictement avec la distance au centre (ρ de Spearman minimal 1) ; magnitude, phase et enveloppe aussi (48 échelles, ρ minimal 1).
  - Preuve : `diagnosis/probe_metrics/report_numbers.json` (real_jvm), `diagnosis/probe_metrics/real_extra_jvm.csv`.
  - Statut : lecture préinscrite (monotonie observée, non imposée), rangée en préliminaire : A (0 à 150 s) et B (215 à 300 s) viennent du même fichier, indépendance non établie ; aucune répétition, donc aucun plancher réel. Lecture des métriques existantes post hoc.
- JVM410H, couverture partielle : Hammerstein 5 capte 0,63 (préampli) et 0,62 (haut-parleur) de l'écart, part non représentée 0,75 et 0,76 ; les scores de sonde y sont des signatures partielles.
  - Preuve : `diagnosis/probe_metrics/report_numbers.json` (real_jvm).
  - Statut : post hoc ; causes possibles non départagées (capacité gelée dépassée, A et B non indépendants, niveaux variables de la DI).
- JVM410H, limite de capacité à fort gain : sur la sortie de référence, résidu de lecture de Hammerstein 5 0,71 au gain 5 (ESR 0,50) contre 0,06 au gain 0 ; aucune classe essayée ne descend sous 0,59 en lecture (ESR 0,34 ; c'est le même fait, les résidus sont des rapports RMS), ni sous 0,59 en échantillon pour les sondes linéaires en paramètres dont la valeur en échantillon est archivée (Hammerstein 5 et sonde parallèle à préfiltres de Laguerre ; F-0035 inclut aussi 1 024 taps, base tanh et Volterra–Laguerre 3, sans valeur en échantillon archivée).
  - Classes essayées : 1 024 taps 0,72 ; base saturante tanh 0,67 (gain 5) et 0,59 (gain 10) ; Wiener–Hammerstein parallèle à préfiltres de Laguerre fixes 0,59 à 0,66 en échantillon, 0,68 à 0,83 en lecture ; Wiener–Hammerstein ajusté sur 1 s 0,39 en échantillon mais 1,08 en lecture (surapprentissage) ; Volterra–Laguerre 3 comparable à Hammerstein 5 sur les paires (part non représentée 0,68 à 0,97 contre 0,68 à 0,97). Hammerstein 5 en échantillon : 0,74 sur 150 s, 0,65 sur 30 s.
  - Contrôles n'appuyant pas les explications suivantes, dans les conditions testées (ils ne les réfutent pas en général) : bruit (mesuré pendant les silences d'entrée, ce qui ne borne pas un bruit dépendant du signal ; sortie à −28 dB du RMS au gain 5, −26 dB au gain 0) ; dynamique sur une demi-seconde (deux groupes d'une seule tranche de niveau ; résidu 0,67 après un passage plus fort, 0,66 sans) ; dérive d'horloge (fenêtres espacées, retards entiers ; retard optimal à ± 1 échantillon près sur 300 s) ; la séparation A/B ne suffit pas à expliquer l'échec (même résultat en échantillon).
  - Preuve : `diagnosis/probe_metrics/gap_checks_jvm.json`, `diagnosis/probe_metrics/gap_checks_jvm_causes.json`, `diagnosis/probe_metrics/gap_checks_jvm_parallel.json`, `diagnosis/probe_metrics/gap_checks_jvm_saturating.json` ; F-0035 ; figure `figures/jvm_residual.pdf`.
  - Statut : post hoc (`diagnosis/probe_metrics/gap_checks.py`) ; cause restante en attente (Q-0014).
- Coût en fonction de la durée : exposant de la loi temps ∝ T^k proche de 1 (ESR 1,2, FIR 1,0, H5 1,0, VL3 0,8, WH 1,1) ; extrapolation linéaire à une minute d'audio depuis la plus longue durée mesurée : ESR 0,026 s, MR-STFT 2,2 s, FIR 1,8 s, H3 4,9 s, H5 10 s, VL2 22 s, VL3 117 s, WH 38 min.
  - Preuve : `diagnosis/probe_metrics/cost_scaling.json`.
  - Statut : mesure post hoc (demandée après le protocole, D-0010), machine partagée.
- Ensemble minimal de métriques complémentaires proposé : ESR (mesure directement l'écart global, y compris les composantes mal représentées par les sondes, dont la capture dépend de l'excitation : écho capté jusqu'à 0,99 sur deux tons) ; magnitude et phase de `src/fssr_nam/metrics/spectral.py` ; FIR ; Hammerstein 3 (H5 redondant ici) ; Volterra–Laguerre 3 ; signature harmonique par ordre si un sinus glissant intact existe ; E2 si des sauts de niveau existent ; part non représentée et avertissement de capacité (seuil 0,3).
  - Preuve : `diagnosis/probe_metrics/REPORT.md` (ensemble minimal, usages et coût), `src/fssr_nam/product/fidelity.py`.
  - Statut : synthèse post hoc, non testée comme ensemble.

## Retiré ou réfuté

- Attente préinscrite VS100R « changements réels au-dessus de 10 planchers réels » : non tenue par aucune métrique sur l'ensemble des changements. Changements hors gain 0 : H3 0,6 à 3,2, ESR 0,9 à 5,2, MR-STFT 0,7 à 6,2 ; maximum sur sondes, ESR et MR-STFT, `di_test` et bruit rose : 7,3 (0 rapport ≥ 10) ; signature harmonique totale sur les sinus glissants : au plus 6,0 ; signature par ordre (ordres 1 à 5, sections signalées par la garde exclues) : 3 rapports ≥ 10, tous sur 1 changement (ordres 2, 4 et 5, `sweep_-20dBFS`, au plus 61,4). Gain 0 (faible signal, rapporté à part, hors AUC) : H3 7,7, ESR 7,2, MR-STFT 15,6 (seul rapport ≥ 10, sur ce seul changement).
  - Preuve : `diagnosis/probe_metrics/report_numbers.json` (real_vs100r), `diagnosis/probe_metrics/real_pairs_vs100r.csv`.
  - Statut : préinscrit, réfuté. Explication post hoc : plancher dominé par la dérive (voir Préliminaire).
- Wiener–Hammerstein retiré de l'ensemble : peu fiable dans 35 cellules sur 39 (règle préinscrite : dispersion entre départs > la moitié du score) ; dispersion d'environ 0,087 du RMS de la sortie quel que soit le défaut, sous l'écart mesuré seulement pour les plus gros (redistribution et préfiltre à leurs deux plus fortes sévérités) ; au développement, dispersion 0,092 juste sous le seuil 0,093 (capacité 0,186).
  - Part captée d'apparence sage (gain 0,98, cubique 0,88) : effet du jumelage des ajustements référence / évaluation partis des mêmes initialisations, qui masque la non-unicité.
  - Coût 32 s pour 1 s d'estimation ; répétabilité la pire du banc (médiane 0,034, 90e centile 0,77).
  - Preuve : `diagnosis/probe_metrics/report_numbers.json` (wiener_hammerstein, dev_v2), `diagnosis/probe_metrics/calibration_test_wh_noise.csv`, `diagnosis/probe_metrics/cost.json` ; F-0033 ; correctifs 7064c38.
  - Statut : règle préinscrite ; retrait de l'ensemble minimal post hoc. Exclu aussi de la lecture Nolly.
  - Conditions : bruit coloré −20 dBFS, 1 s d'estimation et 1 s de lecture, 3 départs, graines 200 à 202.
- MR-STFT de `src/fssr_nam/metrics/spectral.py` écartée : 8 détectés, 7 limites, 63 non détectés à sévérité maximale ; aveugle au passe-tout et au retard à la plus faible sévérité (0/6 et 0/6), magnitude seule, dominée par un bruit à −80 dB dans les bandes où la référence est faible.
  - Preuve : `diagnosis/probe_metrics/report_numbers.json` ; F-0033.
  - Statut : règle préinscrite ; retrait post hoc ; constat propre à cette implémentation, non généralisable aux MR-STFT.
- `envelope_error` (`src/fssr_nam/metrics/transient.py`) écartée comme mesure de dynamique : non spécifique (R 1,4·10⁴ pour un gain, 9,5·10³ pour le relâchement).
  - Preuve : `diagnosis/probe_metrics/table_test.csv`.
  - Statut : post hoc.
- Lecture retirée : corrélation de profil entre ESR et sondes ; lue en amplitude, l'ESR est identique à d_audio et la corrélation n'était que du bruit d'arrondi.
  - Preuve : `diagnosis/probe_metrics/NOTE.md` (relecture Codex) ; commit 7064c38.
- Résultats Wiener–Hammerstein antérieurs au correctif (renormalisation de h_in qui annulait le pas de Gauss–Newton, h_out ajusté sur une autre région) : remplacés ; seuls les chiffres recalculés sont cités.
  - Preuve : `diagnosis/probe_metrics/NOTE.md` ; commit 7064c38 ; les anciens restent dans l'historique git.
- Garde d'intégrité par quarts de sinus glissant retirée : elle pouvait manquer un glissement et en inventer un ; formulation « au pire environ 60 % du plancher » corrigée par la mesure sur grille.
  - Preuve : `diagnosis/probe_metrics/NOTE.md` (deuxième et troisième relectures Codex) ; commits 9d56ed6, c50cc50, 5a7df99.
- Essai court à deux tons hors domaine (fréquences d'estimation et de lecture différentes, scores Hammerstein au-dessus de d_audio) : exclu des résultats, correction avant la passe de test.
  - Preuve : `diagnosis/probe_metrics/calibration_smoke_two_tone_sweep.csv`, `diagnosis/probe_metrics/NOTE.md` (écart au protocole).
- Première procédure des paires réelles (alignement après décimation, glissement jugé par quarts, 21 paires sur 23 exclues) : remplacée avant tout score par l'ajustement de chaque sonde sur sa propre entrée.
  - Preuve : `diagnosis/probe_metrics/NOTE.md` (écart de procédure, paires réelles).

## En attente

- Lecture des rendus Neural DSP Archetype: Nolly, ampli Rhythm (E-0051) : règles figées avant tout rendu (`diagnosis/probe_metrics/NOLLY_READING.md`, `diagnosis/probe_metrics/nolly_reading.py`, commit eb4438c) ; `diagnosis/probe_metrics/nolly_reading.json` n'existe pas : aucun chiffre.
  - H1, monotonie le long des échelles : 5 boutons × 2 côtés × 5 métriques = 50 comparaisons ; « confirmée » à 45 signes ou plus, « non confirmée » si plus de 5 comparaisons mesurées échouent.
  - H2, capacité des sondes (lien Q-0014) : r/RMS de Hammerstein 5 et Volterra–Laguerre 3 aux cinq réglages de l'échelle de gain ; règle de refus au-dessus du seuil gelé d'avertissement de capacité ; croissance « oui » si au moins 3 des 4 pas croissent pour les deux sondes.
  - H3, linéarité du baffle : résidu relatif d'un filtre estimé baffle coupé → baffle actif, seuil gelé à environ trois fois le plancher de la méthode mesuré sur le substitut ; au-delà, « linéarité non établie », jamais « non-linéaire ».
  - H4, spécificité de E2 face au baffle : 9 comparaisons requises ; attendu préinscrit « co-réponse au baffle ».
  - Essai à blanc sur substitut synthétique : JSON non commité, chiffres non citables.
- Cause de l'irreprésentabilité de la sortie JVM410H à fort gain (Q-0014) : non testables avec ces données, le déterminisme de l'enregistrement (aucune répétition) et une cascade d'étages non linéaires ; demande des prises répétées ou une sonde à plusieurs étages.
- Lien perceptif : aucune donnée d'écoute ; le seuil 10 est un seuil d'instrument ; portée limitée à la détection.
- E2 sur matériel : aucun saut de niveau daté dans les captures (sauts de gain ToneTwist non datés, `bursts_recovery` sur une seule prise micro aux réglages connus) ; pas de nouvelle capture matérielle possible pour l'instant (D-0013).
- Plancher réel apparié dans le temps et répétabilité d'un jour à l'autre sur la VS100R : demandent des répétitions du centre intercalées entre les prises (nouvelles captures).
- Non mesurable avec ces données : BLA au sens strict sur le réel, phase des réponses harmoniques au sens du sinus synchronisé sur la VS100R, niveaux hors des trois sinus glissants, contenu au-dessus de 20 kHz (sinus) ou 24 kHz (DI), sortie VS100R après l'étage de tonalité.
  - Preuve : `diagnosis/probe_metrics/manifest.json`, `diagnosis/probe_metrics/NOTE.md`.
- Contrôles Codex cités dans REPORT sans fichier archivé : point de glissement non détecté ajouté en fin de balayage, absence de faux positif de la garde sous bruit ajouté ; non régénérables, non cités.
- Généralisation des lectures sur sinus glissant à de nouvelles réalisations : demande des sinus de test distincts de ceux du développement.
- Validation du seuil de compensation des branches sur d'autres données.
- Énoncés théoriques (équivalence linéaire, norme de Gram, borne triangulaire) : absents de `THEORY.md`, à formaliser.
