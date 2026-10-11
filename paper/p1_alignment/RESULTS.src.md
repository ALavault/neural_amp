# P1 — Résultats : pièges d'alignement et de polarité dans l'évaluation des modèles neuronaux d'effets audio

<!-- Dossier sans prose (D-PRODUCT-002, research-memory D-0016). Chaque nombre décimal est une clé de numbers.json, régénérée par numbers.py depuis les fiches existantes. Voie produit : étiquette « exploratoire ». F-, E-, D-, Q-, T- : research-memory projects/neural_amp. -->

## Établi

- **Retard relatif cible–sortie de 4 échantillons sur F-Fuzz, commun à deux architectures et non appris.**
  Interprétation : latence de rendu des données, non isolée d'une erreur de phase commune aux modèles ; le retard entrée–cible mesuré sur l'audio seul varie de −11 à +4 selon le fichier (`DIAGNOSIS_ffuzz.md`), ce n'est pas une mesure propre de latence (relecture Codex du 2026-10-11).
  Retard optimal sortie → cible de +4 échantillons dans {{ffuzz_lag4_ssm}} segments de test sur {{ffuzz_n_segments}} (SSM-WaveNet-FiLM) et {{ffuzz_lag4_s4}} (S4-param ; +3 dans les {{ffuzz_lag3_s4}} autres) ; +4 dans {{ffuzz_train_lag4_ssm}} et {{ffuzz_train_lag4_s4}} segments d'entraînement sur {{ffuzz_train_n_segments}}.
  ESR de test brute {{ffuzz_esr_raw_ssm}} et {{ffuzz_esr_raw_s4}} (rapport {{ffuzz_esr_raw_ratio}}) alors que la MR-STFT de test diffère d'un facteur {{ffuzz_mrstft_raw_ratio}} ({{ffuzz_mrstft_raw_ssm}} contre {{ffuzz_mrstft_raw_s4}}) : plancher d'ESR commun.
  ESR intérieure {{ffuzz_esr_interior_lag0_ssm}} et {{ffuzz_esr_interior_lag0_s4}} au retard 0, {{ffuzz_esr_realigned_ssm}} et {{ffuzz_esr_realigned_s4}} réalignée (division par {{ffuzz_realign_factor_ssm}} et {{ffuzz_realign_factor_s4}}) ; au retard 0, l'ESR croît avec le fuzz, de {{ffuzz_esr_lag0_f000_ssm}} (F000) à {{ffuzz_esr_lag0_f100_ssm}} (F100) pour SSM.
  Preuve : F-0027, E-0027 ; `DIAGNOSIS_ffuzz.md` (sections 3 et 4), `diagnosis/ffuzz/segments.json`, `diagnosis/ffuzz/train_lag.json`, `demo/nablafx_bench/param_ssm_wavenet_ffuzz_r2.json`, `demo/nablafx_bench/param_s4_ffuzz_r2.json`.
  Statut : hypothèse H1 et sa prédiction (|d| ≥ 1, même retard pour les deux modèles, ESR réalignée au plus la moitié) écrites avant toute mesure ; classement des hypothèses non aveugle (test de fumée vu avant le classement, section 3b) ; réalignement a posteriori (retard choisi après avoir vu la sortie), donc seuil de comparaison et non gain atteignable.
  Conditions : ToneTwist F-Fuzz natif à 48 kHz, {{ffuzz_n_settings}} réglages dont {{ffuzz_n_held_out}} retirés de l'entraînement, une graine ({{ffuzz_seed}}) et une partition ({{ffuzz_split}}) par architecture, pertes L1 {{ffuzz_w_l1}} et MR-STFT {{ffuzz_w_mrstft}}, inférence CPU sur checkpoints existants.

- **Réentraînement préinscrit à cibles avancées de 4 échantillons : tous les critères tenus.**
  ESR de test {{ffuzz_adv4_esr_ssm}} (SSM-WaveNet-FiLM) et {{ffuzz_adv4_esr_s4}} (S4-param), seuil {{ffuzz_thr_esr}} ; écart entre architectures revenu, S4/SSM = {{ffuzz_adv4_ratio}} pour un seuil de {{ffuzz_thr_gap}} (avant : {{ffuzz_esr_raw_ratio}}) ; F100/F000 de SSM {{ffuzz_adv4_f100_f000_ssm}} pour un seuil de {{ffuzz_thr_f100}}.
  Retard optimal 0 dans {{ffuzz_adv4_lag0_ssm}} et {{ffuzz_adv4_lag0_s4}} segments sur {{ffuzz_n_segments}} ; ESR intérieure au retard 0 {{ffuzz_adv4_interior_ssm}} et {{ffuzz_adv4_interior_s4}}, un peu sous le réalignement a posteriori ({{ffuzz_esr_realigned_ssm}} et {{ffuzz_esr_realigned_s4}}).
  Preuve : F-0030, E-0035 ; `FIX_ffuzz.md` (sections 1 à 4 avant tout changement de code, rapport en section 6), `demo/nablafx_bench/param_ssm_wavenet_ffuzz_adv4_r2.json`, `demo/nablafx_bench/param_s4_ffuzz_adv4.json`, `diagnosis/ffuzz/segments_adv4.json`, test `tests/integration/test_bench_target_advance.py`.
  Statut : préinscrit ; rapport relu par Codex en lecture seule.
  Conditions : avance de {{ffuzz_target_advance}} échantillons à l'entraînement, la validation et le test ; une graine, une partition par architecture, aucune barre d'erreur sur le rapport ; {{ffuzz_adv4_steps_ssm}} et {{ffuzz_adv4_steps_s4}} pas (arrêt anticipé de S4) ; relance de SSM après un arrêt externe, écart déclaré avant relance ; changement de pilote NVIDIA couvert par une porte au bit pour SSM-FiLM, pas pour le FiLM statique de S4-param.

- **Les témoins Big Muff et Marshall ne présentent pas le retard entier uniforme de +4 de F-Fuzz (même instrument).**
  Retards non nuls sur une partie des segments, de plusieurs échantillons sur certains réglages Marshall ; instrument au pas entier, un résidu fractionnaire n'est pas exclu.
  Big Muff : retard 0 dans {{ctrl_bigmuff_lag0_ssm}} segments de test sur {{ctrl_bigmuff_n}} pour SSM-WaveNet et {{ctrl_bigmuff_lag0_s4}} pour S4-TF ; ESR {{ctrl_bigmuff_esr_lag0}} → {{ctrl_bigmuff_esr_best}} au réalignement.
  Marshall : retard 0 dans {{ctrl_marshall_lag0_ssm}} (SSM) et {{ctrl_marshall_lag0_s4}} (S4) segments sur {{ctrl_marshall_n}} ; réalignement par segment {{ctrl_marshall_esr_lag0}} → {{ctrl_marshall_esr_best}} (gain de {{ctrl_marshall_gain_pct}} %, optimiste car le retard est choisi sur le segment).
  Preuve : F-0027, E-0027 ; `diagnosis/ffuzz/other_devices_lag.json`, `diagnosis/ffuzz/other_devices_lag.py`, `DIAGNOSIS_ffuzz.md` (section 6).
  Statut : lecture écrite avant la mesure (section 6 du diagnostic).
  Conditions : un run par appareil et par architecture (partition 925), cibles ToneTwist.

- **VS100R à 48 kHz contre 192 kHz : lecture préinscrite M1 sur l'ESR brute.**
  M1 « {{vs100r_m1_label}} » tenue telle qu'écrite : d = {{vs100r_m1_d}} (moyenne de ln ESR, B192 moins A48), au-delà des deux seuils écrits ({{vs100r_m1_thr_ln}} et {{vs100r_m1_bound}}).
  Preuve : E-0040 ; `demo/capture/valvestate/predictions_192k.md`, `demo/capture/valvestate/eval_192k.json`.
  Statut : préinscrit ; l'ESR brute de cette lecture est sensible à l'alignement (voir Préliminaire), M1 ne dit rien du timbre.
  Conditions : Marshall Valvestate VS100R, capture exploratoire de l'auteur (envoi FX, gain et contour seuls), {{vs100r_n_test_takes}} prises de test, 3 graines par bras, lecture à 48 kHz.

- **Longueur d'entraînement par prise, lecture P préinscrite (ESR brute).**
  L10 (30 s par prise) « {{trainlen_p_l10_label}} » que C (150 s) : d = {{trainlen_p_l10_d}}, d_i = {{trainlen_p_l10_di}} ; acceptable : {{trainlen_p_l10_acceptable}}.
  L20 (60 s) « {{trainlen_p_l20_label}} » : d = {{trainlen_p_l20_d}}, d_i = {{trainlen_p_l20_di}} ; acceptable : {{trainlen_p_l20_acceptable}}.
  Seuils écrits d'avance : « meilleur » si d ≤ −{{trainlen_thr_verdict}} et les trois d_i < 0 ; acceptable si d ≤ +{{trainlen_thr_accept}} et chaque d_i ≤ +{{trainlen_thr_accept_seed}}.
  Preuve : F-0037, E-0050 ; `demo/train_length/predictions.md`, `demo/train_length/eval_train_length.json`.
  Statut : préinscrit ; la règle de décision appliquée à P donne 30 s ; l'utilisateur a gardé 150 s hors règle (D-0012).
  Conditions : VS100R, SSM-WaveNet ZOH FiLM joint, 3 graines appariées par bras ({{trainlen_n_runs}} runs), {{trainlen_n_takes}} prises, {{trainlen_n_pairs}} couples gain–contour vus, checkpoint `last`.

- **Même comparaison, co-lecture à retard commun compensé : l'étiquette et l'acceptabilité s'inversent.**
  Un retard commun τ par run, estimé sur `di_validation` (grille de −2 à +3 échantillons), appliqué tel quel à `di_test`, sans oracle par prise.
  L10 « {{trainlen_a_l10_label}} » : d = {{trainlen_a_l10_d}}, d_i = {{trainlen_a_l10_di}} (ESR plus haute d'un facteur {{trainlen_a_l10_ratio}} en moyenne géométrique) ; acceptable : {{trainlen_a_l10_acceptable}}.
  L20 « {{trainlen_a_l20_label}} » : d = {{trainlen_a_l20_d}}, d_i = {{trainlen_a_l20_di}} ; acceptable : {{trainlen_a_l20_acceptable}} (d₄₂ = {{trainlen_a_l20_d42}} pour une borne de {{trainlen_thr_accept_seed}}).
  Retard selon le bras : τ de {{trainlen_tau_l10_min}} à {{trainlen_tau_l10_max}} échantillon pour L10, {{trainlen_tau_c_min}} à {{trainlen_tau_c_max}} pour C, {{trainlen_tau_l20_min}} à {{trainlen_tau_l20_max}} pour L20 ; le calage divise l'ESR par environ {{trainlen_div_c}} (C), {{trainlen_div_l10}} (L10) et {{trainlen_div_l20}} (L20).
  Lecture fixée d'avance : en cas de divergence, l'écart est attribué au calage sous ce protocole, sans autre cause ; la règle de décision reste celle de P.
  Preuve : F-0037, E-0050 ; `demo/train_length/predictions.md` (section « Co-lecture à retard compensé »), `demo/train_length/coreading_delay.py`, `demo/train_length/eval_train_length_aligned.json`, `demo/train_length/parity_gpu_cpu.json`.
  Statut : « {{trainlen_rule}} » ; étiquettes calculées à titre descriptif ; les ESR de test du banc existaient sur disque quand la règle a été écrite (non affichées par cette session, non exclu pour une session antérieure).
  Conditions : inférence GPU (écart d'instrument déclaré), parité GPU–CPU en ln ESR au plus {{trainlen_parity}} ; effet de bord du retard fractionnaire {{trainlen_edge}} pour une borne de {{trainlen_edge_bound}} ; {{trainlen_grid_edge_runs}} run au bord de grille ; retard par rampe de phase FFT.

- **Piège de polarité de L1 + MR-STFT le long du gain de sortie (théorème T2).**
  Sous les hypothèses de T2 (perte auraloss spécifiée, poids positifs, cible non nulle, aucune puissance STFT de la cible exactement au plancher), avec L = w₁ L1 + w₂ MRSTFT et ŷ = (−1 + ε) y, ŷ = −y est un minimum local strict du chemin de gain si w₁ E|y| < w₂ m, et n'en est pas un si w₁ E|y| > w₂ m (m : moyenne sur les résolutions de ρ_r + f_r).
  Preuve : `THEORY.md` (T2), `theory/T2_polarity_trap.md`, `theory/t2_check.py`.
  Statut : énoncé théorique, D (2026-10-02) au registre `THEORY.md` (preuve, contrôles symbolique et numérique, relecture Codex en deux passes) ; une direction, une solution exacte, un lot ; rien sur la dynamique ni sur la probabilité d'inversion ; corollaire non établi pour la tête tanh du banc.
  Conditions : perte du banc `scripts/product_nablafx_bench.py` (auraloss par défaut) ; marge numérique sur Big Muff en attente (voir En attente).

- **Une run SSM-WaveNet a convergé vers une sortie inversée.**
  ESR de test {{polar_inv_esr}} au pas {{polar_inv_step}}, {{polar_inv_esr_negated}} une fois la sortie négée ; MR-STFT identique dans les deux cas ({{polar_inv_mrstft}} et {{polar_inv_mrstft_negated}}) ; L1 {{polar_inv_l1}} contre {{polar_inv_l1_negated}} négée, proche de 2E|y| = {{polar_two_mean_abs_y}} (E|y| = {{polar_mean_abs_y}} sur la cible de test).
  Preuve : F-0005, E-0005 ; `paper/icassp2027/data/polarity_ssmzoh_seed42.json`, `paper/RESULTS_SSM_SEEDS.md` (section 1).
  Statut : constat post hoc, n = 1 ; compatible avec T2 (rétrodiction), non expliqué par lui.
  Conditions : ToneTwist Big Muff, protocole NablAFx, poids L1 {{polar_w_l1}} et MR-STFT {{polar_w_mrstft}}, discrétisation ZOH, exemption de weight decay seule, sans garde de polarité, graine 42.

- **Garde de polarité (`PolarityGuard`) : retournements tôt dans l'entraînement.**
  {{polar_guard_flips}} retournements sur {{polar_guard_runs}} runs gardés du banc, jamais après l'époque {{polar_guard_last_epoch}} ; de {{polar_guard_zoh_flips_min}} à {{polar_guard_zoh_flips_max}} par run SSM-WaveNet ZOH.
  Run gardée de même graine : ESR {{polar_guard42_esr}}, {{polar_guard42_esr_negated}} si l'on nie sa sortie.
  Preuve : F-0005, E-0005 ; `scripts/product_nablafx_bench.py` (classe `PolarityGuard`), `demo/nablafx_bench/ssmzoh_guard_seed42.json` et les huit autres fiches `*_guard_*`, `paper/icassp2027/data/polarity_ssmzoh_guard_seed42.json`.
  Statut : descriptif, post hoc ; un retournement du garde n'est pas une inversion finale.
  Conditions : 6 runs SSM-WaveNet ZOH (graines 42 à 44, répétitions) et 3 runs S4-TF-L-16, Big Muff.

- **Facultatif : sur DI calme, les modèles amplifient le résidu de quantification de l'entrée ; sa contribution à l'ESR globale est très faible.**
  Excès médian de la sortie des modèles : {{quiet_excess_db}} dB au-dessus du plancher de la pédale (L médian {{quiet_l_median}}, de {{quiet_l_min}} à {{quiet_l_max}}) : {{quiet_model_dbfs}} contre {{quiet_target_dbfs}} dBFS ; aucune cohérence détectable avec la cible ({{quiet_coherence}}, au niveau du plancher de l'estimateur) (passages forts {{quiet_coherence_loud}}) ; DI calme annulé : puissance relative {{quiet_zero_di_db}} dB en médiane (10 log₁₀ P₀/P) ; 32 checkpoints ne sont pas 32 réplications indépendantes ; amplification d'environ {{quiet_gain_db}} dB d'un résidu à {{quiet_di_floor_dbfs}} dBFS.
  Preuve : F-0029, E-0031 ; `diagnosis/noise/predictions_quiet.md`, `diagnosis/noise/quiet_frames.json`, `diagnosis/noise/control_zero_di.json`.
  Statut : préinscrit, contrôle déclaré avant de tourner ; issue du tableau préinscrit : « {{quiet_verdict}} ».
  Conditions : {{quiet_n_models}} checkpoints SSM-WaveNet Big Muff du pilote C, {{quiet_n_quiet_windows}} fenêtres calmes, inférence CPU ; cause (MR-STFT ou extrapolation) non attribuable ; on ignore si la pédale a reçu ce résidu.

## Préliminaire

- **VS100R : un retard oracle sous-échantillon renverse le classement entre bras.**
  Retard oracle médian (hors prise 16) de {{vs100r_delay_a48_min}} à {{vs100r_delay_a48_max}} échantillon pour A48, de {{vs100r_delay_b192_min}} à {{vs100r_delay_b192_max}} pour B192, selon la graine.
  Compensé, l'ESR de A48 est divisée par au moins 2 sur {{vs100r_n_takes_ge2}} prises sur {{vs100r_n_test_takes}} à chaque graine : de {{vs100r_ratio_s42_min}} à {{vs100r_ratio_s42_max}} à la graine 42, de {{vs100r_ratio_all_min}} à {{vs100r_ratio_all_max}} sur les trois graines ; au plus {{vs100r_ratio_small_max}} sur les deux autres prises (2 et 16).
  En ESR brute, B192 est meilleur sur {{vs100r_b192_better_raw_s42}}, {{vs100r_b192_better_raw_s43}} et {{vs100r_b192_better_raw_s44}} prises ; après compensation des deux bras, A48 l'est sur {{vs100r_a48_better_aligned_s42}}, {{vs100r_a48_better_aligned_s43}} et {{vs100r_a48_better_aligned_s44}}.
  Preuve : F-0032, E-0040 (note de confusion reprise par E-0044) ; `DIAGNOSIS_vs100r_192k.md` (H6 et mises à jour des graines 43 et 44), `diagnosis/vs100r_192k/measures.json`, `diagnosis/vs100r_192k/measures_s43.json`, `diagnosis/vs100r_192k/measures_s44.json`.
  Statut : post hoc, descriptif ; retard choisi sur la cible de test de chaque prise (±2 échantillons) : il montre la sensibilité du classement, ce n'est pas une correction utilisable ; un oracle peut absorber une erreur de phase du modèle ; origine non établie (Q-0013).
  Conditions : une capture, {{vs100r_n_test_takes}} prises de test, 3 graines par bras.

- **VS100R : le modèle à 48 kHz est en avance d'environ un échantillon, à peu près constant en fréquence.**
  Un décalage commun de {{vs100r_shift_common}} échantillon (environ {{vs100r_shift_us}} µs) fait passer l'ESR médiane de {{vs100r_shift_esr0}} à {{vs100r_shift_esr1}} sur {{vs100r_shift_n}} prises ; retard modèle → appareil de {{vs100r_gd_min}} à {{vs100r_gd_max}} échantillon par bande d'octave de 500 Hz à 8 kHz, alors que le modèle rend le retard de groupe dispersif de l'appareil.
  Même instrument sur d'autres modèles : décalage optimal médian {{vs100r_ctrl_bigmuff_shift}} (Big Muff) et {{vs100r_ctrl_marshall_shift}} (Marshall ToneTwist), contre {{vs100r_ctrl_training_shift}} sur le matériau d'entraînement de la VS100R.
  Preuve : E-0037 ; `demo/capture/valvestate/timing_shift.py`, `demo/capture/valvestate/timing_shift.json`, `demo/capture/valvestate/group_delay.json`, `demo/capture/valvestate/timing_controls.json`.
  Statut : post hoc (écrit après l'évaluation, non déclaré dans `demo/capture/valvestate/predictions_model.md`) ; décalage commun choisi sur les prises notées.
  Conditions : un run (`vs100r_gc_film_joint_s925_r2`, graine 42), section de test, 48 kHz.

- **Longueur d'entraînement : contrôle post hoc avec τ estimé sur `di_test`.**
  L10 encore « {{trainlen_h_l10_label}} » : d = {{trainlen_h_l10_d}}, d_i = {{trainlen_h_l10_di}} (facteur {{trainlen_h_l10_ratio}}) ; L20 « {{trainlen_h_l20_label}} » : d = {{trainlen_h_l20_d}}.
  τ de {{trainlen_htau_l10_min}} à {{trainlen_htau_l10_max}} pour L10, {{trainlen_htau_c_min}} à {{trainlen_htau_c_max}} pour C, {{trainlen_htau_l20_min}} à {{trainlen_htau_l20_max}} pour L20.
  Preuve : F-0037, E-0050 ; `demo/train_length/posthoc_test_delay.py`, `demo/train_length/eval_train_length_posthoc_test_tau.json`.
  Statut : post hoc, hors règle, non indépendant (`di_test` sert à ajuster τ puis à noter) ; ne change aucune étiquette ; aucune attribution causale.
  Conditions : celles de la co-lecture.

- **Trois modèles publiés sur Big Muff ont un L1 proche de 2E|y| : compatible avec une sortie inversée, non confirmé.**
  {{polar_pub_names}} : ESR de test de {{polar_pub_esr_min}} à {{polar_pub_esr_max}} (une sortie exactement inversée donne 4) et L1 de {{polar_pub_l1_min}} à {{polar_pub_l1_max}}, soit {{polar_pub_l1_ratio_min}} à {{polar_pub_l1_ratio_max}} fois 2E|y| ; les autres modèles publiés de la table ont une ESR d'au plus {{polar_pub_others_esr_max}}.
  Preuve : F-0005, E-0005 ; `paper/icassp2027/data/published_bigmuff.json` (annexe de Comunità, Steinmetz et Reiss 2025, tables 20 et 24), `paper/icassp2027/data/polarity_ssmzoh_seed42.json` (E|y|).
  Statut : post hoc ; non confirmé faute de checkpoints publiés ; E|y| mesuré sur notre fichier de test, supposé identique à leur jeu de test.
  Conditions : valeurs publiées, une run par modèle.

- **Taux d'inversion sans garde : non estimable.**
  En ZOH, la seule run sans garde est inversée ; en discrétisation `free`, {{polar_free_inverted}} sur {{polar_free_runs}} (ESR d'au plus {{polar_free_esr_max}}).
  Preuve : F-0005 ; `THEORY.md` (T2, prédiction T2-P1), `demo/nablafx_bench/ssmwavenet_seed42.json`, `demo/nablafx_bench/ssmwavenet_seed43.json`, `demo/nablafx_bench/ssmwavenet_seed44.json`.
  Statut : descriptif, n = 1 en ZOH ; « 0 inversion sur 5 » s'observerait environ une fois sur quatre sans aucun effet (`THEORY.md`).
  Conditions : Big Muff, poids L1 {{polar_w_l1}} et MR-STFT {{polar_w_mrstft}}.

## Retiré ou réfuté

- **Retiré le 2026-10-02 : « la MR-STFT, aveugle à la phase, ne voit pas un retard pur ».**
  Correction : la MR-STFT voit un retard de 4 échantillons sur les cibles F-Fuzz, linéairement en d, mais faiblement devant la MR-STFT de test des modèles ({{ffuzz_mrstft_raw_ssm}} et {{ffuzz_mrstft_raw_s4}}) ; le retard porte en revanche toute l'ESR. Valeur du coût MR-STFT : en attente (aucune fiche JSON).
  Preuve : E-0039 ; `DIAGNOSIS_ffuzz.md` (correction datée en section 5), `FIX_ffuzz.md` (correction d'une prémisse de la section 1) ; l'invariant préinscrit du correctif et son issue ne changent pas.

- **Réfutées : les autres causes du plancher F-Fuzz, par leurs seuils écrits d'avance.**
  Agrégation (H3) : médiane {{ffuzz_alt_median}} contre moyenne {{ffuzz_alt_mean}}, les 10 % pires segments portent {{ffuzz_alt_worst10_share}} du total (seuil 50 %).
  Aigu hors d'atteinte (H4) : part de l'erreur au-dessus de 5 kHz {{ffuzz_alt_hf_share}} en moyenne par segment (seuil 60 %), ESR après passe-bas à 5 kHz {{ffuzz_alt_lowpass_ratio}} fois l'ESR (seuil 50 %).
  Marqueurs (H5) : les {{ffuzz_alt_marker_n}} segments de bord portent {{ffuzz_alt_marker_pct}} % du total (seuil 30 %).
  Contexte manquant (C8) : ESR intérieure {{ffuzz_alt_interior}} contre ESR complète {{ffuzz_alt_full}}.
  Bruit de la cible (H2) : résidu après réalignement de l'ordre de l'ESR réalignée ({{ffuzz_esr_realigned_ssm}}), loin des 11 à 19 % qu'il faudrait.
  Preuve : E-0027 ; `DIAGNOSIS_ffuzz.md` (section 4), `diagnosis/ffuzz/segments.json` ; SSM-WaveNet-FiLM, runs `_r2`.

- **Réfutée pour la prise 16 : « le retard explique l'écart de la prise 16 de la VS100R » (H6).**
  Retards oracles {{vs100r_t16_delay_a48}} (A48) et {{vs100r_t16_delay_b192}} (B192) ; après compensation ln(B/A) vaut encore {{vs100r_t16_ln_ratio_aligned}}. L'écart de cette prise dépend du run (H2 soutenue sur trois graines), pas de l'alignement.
  Preuve : E-0040 ; `DIAGNOSIS_vs100r_192k.md` (sections 4 et mises à jour), `diagnosis/vs100r_192k/measures.json`.
  Statut : hypothèse écrite avant la mesure (section 3 du diagnostic).

- **Retirées le 2026-10-03 après relecture Codex : « l'avantage apparent de B192 est un avantage d'alignement » et « invente un effet du contour ».**
  Elles dépassaient les mesures ; remplacées par « le classement entre les bras dépend de l'alignement temporel (descriptif) ».
  Preuve : `DIAGNOSIS_vs100r_192k.md` (section « Corrections du 2026-10-03 »).

- **Réfutée : « tout modèle ZOH entraîné à 48 kHz est en avance ».**
  Retard oracle médian des {{vs100r_bigmuff_lead_runs}} modèles Big Muff ZOH et FOH de {{vs100r_bigmuff_lead_min}} à {{vs100r_bigmuff_lead_max}} échantillon. Ne réfute pas une contribution du ZOH sur la VS100R ; instrument différent (découpe puis décalage circulaire).
  Preuve : `DIAGNOSIS_vs100r_192k.md` (section 5, élimination du ZOH jugée trop large le 2026-10-03), `diagnosis/vs100r_192k/bigmuff_lead.json`.
  Statut : post hoc, mesure née pendant le diagnostic.

- **Attente non tenue (sans seuil) : τ du bras C proche de l'oracle par prise.**
  Attendu en valeur absolue proche du retard oracle de `DIAGNOSIS_vs100r_192k.md` ; obtenu {{trainlen_tau_c_min}} à {{trainlen_tau_c_max}} contre {{vs100r_delay_a48_min}} à {{vs100r_delay_a48_max}} : un retard commun par run estimé sur `di_validation` n'est pas l'oracle par prise de test.
  Preuve : E-0050 ; `demo/train_length/predictions.md` (contrôles de l'instrument), `demo/train_length/eval_train_length_aligned.json`.

## En attente

- **Capture logicielle Nolly comme contrôle sans acquisition.** Modèles C, GREY, L, F3 préinscrits (D-0015, E-0051) ; co-lecture à retard compensé avec la règle de `demo/train_length/coreading_delay.py` inchangée, après une avance commune entière fixée aux marqueurs ; rendus non livrés (T-0013).
  Preuve attendue : `demo/nolly_models/predictions.md`, `demo/nolly_models/evaluate.py`.
  Ce qu'elle trancherait : si un retard fractionnaire commun reste sur une cible déterministe sans décalage d'entrées.

- **Origine de l'avance d'environ un échantillon de la VS100R (Q-0013).** Décalage entre les deux entrées de l'interface soupçonné ; test à trois prises prêt (`demo/capture/valvestate/INTER_INPUT_TEST.md`, lecture fixée avant les prises) mais bloqué (T-0011) : aucune capture matérielle pour l'instant (D-0013) ; les prises micro du 2026-10-08 ne contenaient pas ce test (E-0046).

- **Confirmation de l'inversion des trois modèles publiés.** Exige leurs checkpoints ou leurs sorties (calcul de l'ESR de la sortie négée) ; non disponibles dans le dépôt.

- **Marge numérique de T2 sur Big Muff** (m, rapport w₂ m / w₁ E|y|). Calculée par `theory/t2_check.py` sur l'audio et la vraie perte ; journal hors dépôt, aucune fiche JSON : non régénérable ici.

- **Coût MR-STFT d'un retard pur de 4 échantillons sur les cibles F-Fuzz et identité ESR = 2(1 − ρ_cible(4)).** Mesurés à la relecture E-0039, sans fiche JSON dans le dépôt : à régénérer (CPU, audio ToneTwist requis).

- **Prédiction T2-P1** (pas d'inversion finale quand w₁ E|y| est assez grand devant w₂ m). Non lancée ; plan départageant : deux bras ZOH sans garde, environ 12 graines par bras (`THEORY.md`).

- **Origine des 4 échantillons F-Fuzz** (latence du plugin, suréchantillonnage interne ou chaîne de rendu) : non vérifiée (Q-0011, partie ouverte) ; présence du même plancher dans les résultats F-Fuzz publiés de NablAFx : non vérifiée.

- **Réentraînement VS100R à cibles avancées** (prédiction de `DIAGNOSIS_vs100r_192k.md`, section 5) : non lancé ; aucun nouvel entraînement neuronal sur la VS100R sous D-0015 ; effet sur la comparaison des longueurs inconnu (`demo/train_length/predictions.md`).
