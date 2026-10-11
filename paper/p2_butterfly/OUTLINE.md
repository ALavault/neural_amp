# P2 — Plan : effet papillon dans l'entraînement des modèles neuronaux d'effets audio

<!-- Dossier sans prose : titres, puces, tableaux. Les valeurs sont dans RESULTS.md et
CLAIMS.md ; ici seules les clés de numbers.json sont nommées. -->

- Titre de travail : Effet papillon dans l'entraînement des modèles neuronaux d'effets audio : graine, tirage des données et décisions pilotées par la validation.
- Rédaction : par l'auteur seul ; ce dossier fournit faits, nombres, affirmations permises, plan et figures.

## Cible

- Venue : DAFx27 (International Conference on Digital Audio Effects), Crémone, 24 au 27 août 2027.
- Échéance de soumission : vers fin mars 2027, à confirmer (DAFx26 : soumission le 30 mars, notification le 15 mai, version finale le 3 juillet).
- Format : 8 pages au plus, références comprises (oral et poster) ; pages supplémentaires payantes : non trouvé ; gabarit LaTeX DAFx de l'édition (DAFx26 : `dafx26v3.sty`, option `blind`), version 2027 à confirmer.
- Relecture : double aveugle, procédure « 1+1/2 » (relecture rapide des articles limites, étape de réfutation) d'après DAFx26 ; à confirmer pour DAFx27 ; la soumission doit déjà avoir la qualité d'une version finale.
- Politique IA : non trouvée pour DAFx26 ; à vérifier dans l'appel DAFx27 ; texte écrit par l'auteur, aucune section produite par un LLM (règle du projet).
- Prépublication : déconseillée ; si postée, mention « non relue, non acceptée ». Un auteur doit s'inscrire et présenter.
- Anonymisation : le dépôt GitHub du projet est public et nommé ; version de relecture avec dépôt miroir anonymisé (code, scripts d'analyse, fiches de runs), page d'écoute anonymisée, métadonnées PDF et audio nettoyées, autocitations à la troisième personne, aucun nom ni ORCID ni URL du dépôt nommé.
- Droits des données : ToneTwist AFx Big Muff, Zenodo 10891515, licence CC BY-NC 4.0 (attribution, usage non commercial) ; pas d'audio brut dans le dépôt ; extraits d'écoute éventuels à diffuser avec attribution et mention NC.
- Alternative : IEEE TASLP, version longue (13 pages à la soumission, relecture à simple anonymat), qui laisserait la place aux résultats préliminaires et aux expériences manquantes.

## Thèse

- Ce que les fiches soutiennent, avec le statut de chaque élément (clés de `numbers.json`) :
  1. La dispersion observée diminue lorsque la chaîne de données (partition et ordre des lots, fixés ensemble) est fixée : seulement Préliminaire, comparaison brute descriptive (`pilot_e_sd`, `bench_raw_sd`) ; le verdict préinscrit du pilote E a été calculé sur une autre quantité que l'effet run préinscrit (`rank_seed_run_effect`) et reste suspendu ; sans attribution à la partition seule ni rapport de variances causal, avec la correction du 2026-10-02 (runs non déterministes, initialisation non exclue : `pilot_e_ci_hi`) et « fixer n'est pas choisir » (`pilot_e_gmean`).
  2. À la fourche 100, après une perturbation d'un pas float32 de chaque poids, l'écart-type du log de l'ESR final est bien plus grand quand le run prend ses décisions de validation que quand il rejoue le calendrier et la durée du témoin : Établi selon les critères préinscrits (`pilot_a_s_decide`, `pilot_a_s_replay`, `pilot_a_delta_ratio`), une graine ; « tard dans l'entraînement » désigne l'époque 100 de ce protocole, pas une phase tardive comparable à d'autres travaux.
  3. « De façon dépendante de la phase » : seulement Préliminaire (`fork5_containment`, `fork25_containment`, `fork100_containment`, trois points dont un confondu) ; à écrire comme piste, pas comme résultat. Écart entre la thèse proposée et les fiches.
  4. Le détecteur de plateau tourne sur du bruit : Établi, mesure post hoc corrigée (`plateau_local_ratio_min`, `plateau_local_ratio_max`, `plateau_margin_max_main`) ; T3 (a) : faible sensibilité au seuil d'une statistique agrégée simulée, mesurée sous préinscription (`t3_shift_up`, `t3_shift_down`), pas l'indépendance des dates individuelles (`t3_counter_k3_proto`, `t3_counter_k3_up`).
  5. Le régler n'est pas un correctif utile : critère de qualité préinscrit manqué pour le seuil entraîné (`pilot_f_gmean`, `pilot_f_g_threshold`, `pilot_f_quality_factor`), verdict de dispersion suspendu (`pilot_f_sd`, quantité) ; parmi les seuils rejoués, aucun ne rend la division reproductible (`plateau_sim_sd_min`) ; le retirer par un calendrier fixe commun ne réduit pas la dispersion hors échantillon : réfutation préinscrite (`pilot_cx_r_early`, `pilot_cx_r_late`).
  6. Fixer la partition rend les comparaisons d'architectures appariées : soutenu par la règle préinscrite (`paired_r5`), mais r₅ est petit surtout parce que chaque bras disperse peu (`paired_removed_share`, `paired_corr`).
  7. Le classement complet par moyenne géométrique sur 12 segments de test est fragile, entre runs d'une même architecture (`rank_seed_reordered`, `rank_child_kept`, `rank_val_children_kept`), post hoc ; le choix du meilleur l'est bien moins (`rank_seed_best_kept`, `rank_val_best_kept`) et les inversions portent sur des quasi-égalités (`rank_seed_flip_gap`, `rank_val_flip_gap`) ; rien sur l'ordre entre architectures.
- Contribution, à présenter comme une réplication contrôlée sur un modèle d'effet audio et non comme la découverte de l'effet papillon : contraste décide/rejoue après une perturbation d'un pas sur chaque poids, entraînement déterministe bit à bit, diagnostic des marges de déclenchement, tentatives de correction réfutées (seuil, calendrier fixe), seuils préinscrits, tous les runs rapportés.
- Recommandation pratique permise : fixer et publier la partition, rapporter plusieurs graines, ne pas classer des modèles sur un run et 12 segments ; ne pas présenter un calendrier fixe ou un seuil relevé comme remède.

## Plan et budget

| Section | Contenu | Pages | Figures/tableaux |
|---|---|---|---|
| 1. Introduction | Comparaisons publiées à un run (NablAFx) ; question : d'où vient la dispersion entre runs et que font les décisions de validation ; contributions (thèse 1, 2, 4 à 7) | 0,75 | — |
| 2. Dispositif | Big Muff, protocole NablAFx, deux pièges (weight decay, polarité ; T2), garde de polarité, déterminisme bit à bit, conventions (log de l'ESR, moyennes géométriques, tous les runs) | 1 | Tab. 1 : conditions, runs, préinscription |
| 3. Structure de la dispersion | Facteur largement commun, plan emboîté, S4-TF-L-16 inverse, pilote E et sa correction, pilote G | 1,25 | `figures/dispersion.pdf` |
| 4. Effet papillon | Pilote A (fourche 100), mécanisme : marges de plateau en fluctuation locale, T3 (a) ; lecture du sort (decided_when), connectivité ; fourches 5 et 25 signalées comme préliminaires | 1,5 | `figures/fork_spread.pdf` |
| 5. Corriger l'ordonnanceur ? | Seuil relevé (pilote F), simulation des seuils, quantité surveillée ; calendrier fixe (pilote C) et extension réfutée ; canal post hoc non reproduit | 1 | `figures/pilot_c_pairs.pdf` |
| 6. Conséquences pour l'évaluation | Partition fixée et protocole apparié (r₅ et sa correction), fragilité des classements, `best` contre `last` | 0,75 | Tab. 2 : décomposition à deux facteurs et bootstrap |
| 7. Limites | Un appareil, une architecture principale, pas d'écoute, seuils de qualité pilote, sources de courbes non suivies | 0,5 | — |
| Références | Voir `RELATED.md` | 1 | — |
| Total | Marge de 0,25 page | 7,75 | 3 figures, 2 tableaux |

## Figures et tableaux

| Fichier | Générateur | Clés de numbers.json | Section |
|---|---|---|---|
| `figures/fork_spread.pdf` | `figures.py:fork_spread` (lit `diagnosis/butterfly/pilot_A_results.json`) | `pilot_a_s_decide`, `pilot_a_s_replay`, `pilot_a_k1_delta`, `fork5_s_decide`, `fork5_s_replay`, `fork5_replay_k1_delta`, `fork25_s_decide`, `fork25_s_replay` | 4 |
| `figures/dispersion.pdf` | `figures.py:dispersion` (écart-type brut de ln ESR par condition ; IC χ² seulement pour des graines indépendantes, aucun pour les plans emboîtés 3 × 2 du banc et du pilote F ni pour les bras de fourche ; le banc y figure par `bench_raw_sd`, pas par l'effet run `rank_seed_run_effect` ; lit les fiches de `demo/nablafx_bench`, `pilot_C_results.json`, `pilot_C_extension_results.json`, `pilot_A_results.json`, `global_factor.json`) | `bench_raw_sd`, `pilot_e_sd`, `pilot_e_ci_lo`, `pilot_e_ci_hi`, `pilot_f_sd`, `pilot_c_sd_decide`, `pilot_c_sd_fixe`, `pilot_cx_sd_decide`, `pilot_cx_sd_fixe`, `paired_sd_ssm`, `paired_sd_s4`, `pilot_a_s_decide`, `pilot_a_s_replay`, `nested_within_sd` | 3 |
| `figures/pilot_c_pairs.pdf` | `figures.py:pilot_c_pairs` (lit `pilot_C_results.json`, `pilot_C_extension_results.json`) | `pilot_c_ratio`, `pilot_c_esr_48`, `pilot_c_esr_49`, `pilot_cx_worst_decide`, `pilot_cx_r_eight`, `pilot_cx_r_sixteen`, `pilot_cx_r_wo` | 5 |
| Tab. 1 (à composer par l'auteur) | Fiches de runs, `RESULTS.md` | `pilot_e_runs`, `pilot_a_runs_per_arm`, `paired_splits`, `proto_params_ssm`, `proto_params_s4` | 2 |
| Tab. 2 (à composer par l'auteur) | `numbers.py:evaluation` | `rank_seed_run_effect`, `rank_seed_segment_effect`, `rank_seed_interaction`, `rank_seed_kept`, `rank_child_run_effect`, `rank_child_kept`, `rank_seed_run_effect_best`, `rank_child_run_effect_best` | 6 |

## Théorie

| Énoncé | Statut | Rôle dans l'article |
|---|---|---|
| T3 | M pour l'indépendance au seuil, S pour la loi de premier passage (2026-10-02) | Section 4 : (a) faible sensibilité de la statistique agrégée simulée (médiane des médianes prédites) aux seuils testés, mesurée (M), l'indépendance des résidus restant une hypothèse de la règle préinscrite ; (b) loi de premier passage, supposée (S), comme modèle candidat seulement |
| T2 | D (2026-10-02) | Section 2 : condition du piège de polarité le long du gain ; motive le contrôle de polarité, la garde reste un choix de protocole ; sans probabilité d'inversion |

- Pour porter T3 au statut D (preuve complète, contrôle symbolique, contrôle numérique sur le code réel, relecture indépendante) :
  1. (a) Resserrer d'abord l'énoncé (voir « Manques ») puis démontrer la faible sensibilité au seuil : borne sur la probabilité qu'un record dépende de θ quand θ est très petit devant la fluctuation relative σ (l'ensemble des valeurs qui battent le meilleur à θ près mais pas à θ = 0 a une probabilité d'ordre θ/σ par époque), sommée sur la fenêtre de patience de 21 époques et sur la durée jusqu'à la première division.
  2. Contrôle symbolique de la borne (sympy) et contrôle numérique sur le vrai `torch.optim.lr_scheduler.ReduceLROnPlateau`, pas seulement sur `diagnosis/seeds/plateau_simulation.py:halvings` (dont l'équivalence au rappel doit elle-même être vérifiée, porte de rejeu décalée d'une époque).
  3. (b) Corriger le modèle avant toute preuve : les `t3_misses_late` écarts vont tous vers une division plus tardive ; tester une tendance mieux extrapolée ou un bruit à mémoire (au-delà de VR(10)) ; dériver la loi de premier passage dans le cadre de la statistique des records (tendance plus bruit indépendant) ; préinscrire la mesure sur de nouvelles runs (réajuster les 21 runs serait post hoc).
  4. Étendre la portée déclarée : autre architecture (S4-TF-L-16), autre appareil, autre patience.
  5. Deux passes de relecture indépendante (Codex gpt-6-astra, xhigh puis high), comme pour T2.
- Théorie à développer en lien avec l'article, sans statut aujourd'hui : formaliser la chaîne « perturbation → bruit de validation → date de division → trajectoire » (marges de `plateau_margin.md`) comme prédiction quantitative de s(décide) à partir des marges ; expliquer pourquoi la pente log ESR contre log L1 dépasse 2 (`decided_slope_children`).

## Travaux liés

- Voir `RELATED.md` (préparé en parallèle avec `refs.bib`, entrées marquées [V] ou [NV]).

## Manques avant soumission

- Pertinence perceptive : séance ABX de `demo/listening/butterfly.html` jamais faite (T-0005) ; coût : une séance d'écoute de l'auteur, 3 extraits × 3 paires × 10 essais, sans GPU ; nouvelle expérience (humaine) ; sans elle, seule « aucune évaluation perceptive n'étaye ces écarts d'ESR » est permise.
- Un seul appareil (Big Muff) : rejouer le pilote A (fourche 100, deux bras × 5 runs) sur un second appareil ToneTwist ; coût : environ 15 runs déterministes d'environ 2 h 30 chacun ; nouvelle expérience ; attention au retard de 4 échantillons des cibles F-Fuzz (F-0027).
- Une architecture principale : S4-TF-L-16 a la structure de variance inverse (F-0028) ; pilote A sur S4-TF-L-16 en mode déterministe ; coût : environ 15 runs ; nouvelle expérience.
- Pilote E non déterministe : le refaire en mode déterministe pour séparer initialisation et non-déterminisme ; coût : 6 runs, environ 12 h GPU ; nouvelle expérience.
- Partition et ordre des lots inséparés : modifier le module de données de nablafx, puis 6 runs ou plus ; nouvelle expérience.
- Fourches précoces et dépendance à la phase : n = 5 par bras, seuils de qualité pilote, fourche 5 confondue par la garde ; plus d'enfants par fourche (au moins 10) aux fourches 25 et 100 ; nouvelle expérience (environ 20 à 40 runs).
- Direction de la perturbation (Q-0010) : d'autres directions sur le même parent ; nouvelle expérience.
- T3 (b) : nouvelles runs préinscrites ou réutilisation de runs futures ; peut se greffer sur les expériences ci-dessus.
- Puissance : 3 à 8 graines par condition, intervalles larges (`pilot_e_ci_hi`, `nested_within_ci_hi`) ; plus de graines si une affirmation chiffrée doit tenir.
- Sources non suivies par git : `demo/runs/*/logs/metrics.csv` alimente `plateau_local_*`, `plateau_k0_first_margin`, `mq_*`, `decided_val_*`, `decided_test_over_val_*`, `decided_beyond_two_*` ; décider de leur archivage (dépôt anonymisé ou Zenodo) avant soumission ; pas d'expérience.
- Écarts d'arrondi entre fiches et recalcul, à corriger dans les fiches ou à citer depuis `numbers.json` : `paired_r3`, `decided_val_dilation`, `lmc_barrier_ratio`, `nested_within_ci_hi`, `mq_l1_same`, `rank_best_last_max_gap`, `det_s4_cost` (détail dans le compte rendu du dossier) ; pas d'expérience.
- Bibliographie : `RELATED.md` et `refs.bib` (autre agent) ; vérifier la nouveauté de « décisions de validation comme amplificateurs » (recherche bornée de E-0008, nouveauté non établie) ; pas d'expérience.
- Appel DAFx27 : dates, politique IA, gabarit et règles de pages à confirmer à sa parution ; pas d'expérience.
- Anonymisation : miroir anonyme du code et des fiches, page d'écoute sans nom ; pas d'expérience.
- Énoncé de T3 (a) dans `THEORY.md` à resserrer (« l'époque de la première division ne dépend pas du seuil » dit plus que la mesure : statistique agrégée simulée, contre-exemple individuel `t3_counter_k3_proto` / `t3_counter_k3_up`, indépendance des résidus non établie `t3_ar1_vr10`) ; travail de registre hors du dossier, non fait ici ; pas d'expérience.
- Intervalles χ² : ceux des bras de fourche mêlent quatre perturbations et un témoin non perturbé (pas des tirages i.i.d.), et la normalité des log ESR est douteuse à ces effectifs ; la figure `figures/dispersion.pdf` ne donne plus d'intervalle pour les bras de fourche ni pour les plans emboîtés ; décider d'un intervalle par bootstrap ou n'en donner aucun ; pas d'expérience.
- Contraste décide/rejoue : le bras rejoue impose aussi la durée d'entraînement du témoin ; à la fourche 5 la garde de polarité reste active ; l'ordre des lots est partagé entre enfants ; séparer l'ordonnanceur du dispositif décide/rejoue (calendrier rejoué avec arrêt propre, ou calendrier propre avec arrêt imposé) ; nouvelle expérience (deux bras de plus × 5 runs à la fourche 100).
- « Époque 100 » n'établit pas une « phase tardive » comparable à celle d'autres travaux, ni un chaos au sens des systèmes dynamiques (aucun exposant de Lyapunov mesuré) ; à formuler comme époque de ce protocole ; pas d'expérience, ou mesure de divergence en fonction du temps si l'on veut le terme.
- Présentation de la contribution : réplication contrôlée sur un modèle d'effet audio (contraste décide/rejoue après une perturbation d'un pas de chaque poids, diagnostic des marges, tentatives de correction réfutées), pas découverte de l'effet papillon ; cohérence avec `RELATED.md` ; pas d'expérience.
- Ordre SSM-WaveNet/S4-TF-L-16 et tirage des segments de test : jamais mesuré (le bootstrap existant ne porte que sur des runs SSM-WaveNet) ; demande les ESR par segment des runs S4-TF-L-16, par inférence CPU sur les points de contrôle existants ; pas d'entraînement.
- Pilote E dans la même quantité que l'effet run du banc : demande les ESR par segment des six runs à chaîne commune, par inférence CPU ; pas d'entraînement.
- Pilotes E et F : les verdicts préinscrits de dispersion ont été calculés sur l'écart-type brut du log de l'ESR moyen (`pilot_e_sd`, `pilot_f_sd`) alors que les préinscriptions définissent l'écart-type de l'effet run (`rank_seed_run_effect`) ; relevé en relecture le 2026-10-11 ; calculer l'effet run de ces douze runs (ESR par segment par inférence CPU sur les points de contrôle existants, pas d'entraînement), puis appliquer les règles telles qu'écrites ; d'ici là, la thèse 1 reste préliminaire.
- Classement complet contre choix du meilleur : rapporter les deux (`rank_val_children_kept` contre `rank_val_best_kept`, `rank_seed_kept` contre `rank_seed_best_kept`) et la taille des inversions (`rank_seed_discordant_median`, `rank_seed_flip_gap`, `rank_child_flip_gap`, `rank_val_flip_gap`) ; à reprendre dans les limites de la section 6 ; pas d'expérience.
- Tirage des segments de validation et décisions de plateau : jamais mesuré sur les trajectoires (le bootstrap existant porte sur les points de contrôle terminaux) ; rejouer le détecteur sur des courbes recalculées par segment demanderait les pertes de validation par segment à chaque époque (inférence CPU sur des points de contrôle intermédiaires, que le protocole ne conserve pas) ; nouvelle expérience.
- Rapport L1/ESR : écrire 2 comme référence sous changement d'échelle de l'erreur à forme fixe, pas comme borne ; garder les pentes empiriques (`decided_slope_fixe`, `decided_slope_children`) ; pas d'expérience.
