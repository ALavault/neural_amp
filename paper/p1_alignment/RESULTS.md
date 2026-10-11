<!-- Généré par paper/render.py depuis RESULTS.src.md. -->

# P1 — Résultats : pièges d'alignement et de polarité dans l'évaluation des modèles neuronaux d'effets audio

<!-- Dossier sans prose (D-PRODUCT-002, research-memory D-0016). Chaque nombre décimal est une clé de numbers.json, régénérée par numbers.py depuis les fiches existantes. Voie produit : étiquette « exploratoire ». F-, E-, D-, Q-, T- : research-memory projects/neural_amp. -->

## Établi

- **Latence de 4 échantillons des cibles F-Fuzz, non apprise par deux architectures.**
  Retard optimal sortie → cible de +4 échantillons dans 1463 segments de test sur 1463 (SSM-WaveNet-FiLM) et 1459 (S4-param ; +3 dans les 4 autres) ; +4 dans 100 et 100 segments d'entraînement sur 100.
  ESR de test brute 0,1366 et 0,1377 (rapport 1,01) alors que la MR-STFT de test diffère d'un facteur 1,6 (0,18 contre 0,29) : plancher d'ESR commun.
  ESR intérieure 0,136 et 0,137 au retard 0, 0,0013 et 0,0032 réalignée (division par 103 et 43) ; au retard 0, l'ESR croît avec le fuzz, de 0,113 (F000) à 0,182 (F100) pour SSM.
  Preuve : F-0027, E-0027 ; `DIAGNOSIS_ffuzz.md` (sections 3 et 4), `diagnosis/ffuzz/segments.json`, `diagnosis/ffuzz/train_lag.json`, `demo/nablafx_bench/param_ssm_wavenet_ffuzz_r2.json`, `demo/nablafx_bench/param_s4_ffuzz_r2.json`.
  Statut : hypothèse H1 et sa prédiction (|d| ≥ 1, même retard pour les deux modèles, ESR réalignée au plus la moitié) écrites avant toute mesure ; classement des hypothèses non aveugle (test de fumée vu avant le classement, section 3b) ; réalignement a posteriori (retard choisi après avoir vu la sortie), donc seuil de comparaison et non gain atteignable.
  Conditions : ToneTwist F-Fuzz natif à 48 kHz, 11 réglages dont 2 retirés de l'entraînement, une graine (42) et une partition (925) par architecture, pertes L1 0,5 et MR-STFT 0,5, inférence CPU sur checkpoints existants.

- **Réentraînement préinscrit à cibles avancées de 4 échantillons : tous les critères tenus.**
  ESR de test 0,00164 (SSM-WaveNet-FiLM) et 0,00344 (S4-param), seuil 0,01 ; écart entre architectures revenu, S4/SSM = 2,10 pour un seuil de 1,5 (avant : 1,01) ; F100/F000 de SSM 5,49 pour un seuil de 10.
  Retard optimal 0 dans 1463 et 1463 segments sur 1463 ; ESR intérieure au retard 0 0,00109 et 0,00284, un peu sous le réalignement a posteriori (0,0013 et 0,0032).
  Preuve : F-0030, E-0035 ; `FIX_ffuzz.md` (sections 1 à 4 avant tout changement de code, rapport en section 6), `demo/nablafx_bench/param_ssm_wavenet_ffuzz_adv4_r2.json`, `demo/nablafx_bench/param_s4_ffuzz_adv4.json`, `diagnosis/ffuzz/segments_adv4.json`, test `tests/integration/test_bench_target_advance.py`.
  Statut : préinscrit ; rapport relu par Codex en lecture seule.
  Conditions : avance de 4 échantillons à l'entraînement, la validation et le test ; une graine, une partition par architecture, aucune barre d'erreur sur le rapport ; 15000 et 14944 pas (arrêt anticipé de S4) ; relance de SSM après un arrêt externe, écart déclaré avant relance ; changement de pilote NVIDIA couvert par une porte au bit pour SSM-FiLM, pas pour le FiLM statique de S4-param.

- **La latence est propre aux rendus F-Fuzz (contrôles Big Muff et Marshall, même instrument).**
  Big Muff : retard 0 dans 11 segments de test sur 12 pour SSM-WaveNet et 11 pour S4-TF ; ESR 0,0415 → 0,0413 au réalignement.
  Marshall : retard 0 dans 121 (SSM) et 96 (S4) segments sur 153 ; réalignement par segment 0,101 → 0,088 (gain de 13 %, optimiste car le retard est choisi sur le segment).
  Preuve : F-0027, E-0027 ; `diagnosis/ffuzz/other_devices_lag.json`, `diagnosis/ffuzz/other_devices_lag.py`, `DIAGNOSIS_ffuzz.md` (section 6).
  Statut : lecture écrite avant la mesure (section 6 du diagnostic).
  Conditions : un run par appareil et par architecture (partition 925), cibles ToneTwist.

- **VS100R à 48 kHz contre 192 kHz : lecture préinscrite M1 sur l'ESR brute.**
  M1 « B192 plus proche » tenue telle qu'écrite : d = −0,422 (moyenne de ln ESR, B192 moins A48), au-delà des deux seuils écrits (−0,405 et −0,225).
  Preuve : E-0040 ; `demo/capture/valvestate/predictions_192k.md`, `demo/capture/valvestate/eval_192k.json`.
  Statut : préinscrit ; l'ESR brute de cette lecture est sensible à l'alignement (voir Préliminaire), M1 ne dit rien du timbre.
  Conditions : Marshall Valvestate VS100R, capture exploratoire de l'auteur (envoi FX, gain et contour seuls), 8 prises de test, 3 graines par bras, lecture à 48 kHz.

- **Longueur d'entraînement par prise, lecture P préinscrite (ESR brute).**
  L10 (30 s par prise) « meilleur » que C (150 s) : d = −0,118, d_i = −0,091 ; −0,038 ; −0,225 ; acceptable : oui.
  L20 (60 s) « équivalent » : d = +0,039, d_i = +0,104 ; −0,011 ; +0,024 ; acceptable : oui.
  Seuils écrits d'avance : « meilleur » si d ≤ −0,1 et les trois d_i < 0 ; acceptable si d ≤ +0,10 et chaque d_i ≤ +0,25.
  Preuve : F-0037, E-0050 ; `demo/train_length/predictions.md`, `demo/train_length/eval_train_length.json`.
  Statut : préinscrit ; la règle de décision appliquée à P donne 30 s ; l'utilisateur a gardé 150 s hors règle (D-0012).
  Conditions : VS100R, SSM-WaveNet ZOH FiLM joint, 3 graines appariées par bras (9 runs), 35 prises, 12 couples gain–contour vus, checkpoint `last`.

- **Même comparaison, co-lecture à retard commun compensé : l'étiquette et l'acceptabilité s'inversent.**
  Un retard commun τ par run, estimé sur `di_validation` (grille de −2 à +3 échantillons), appliqué tel quel à `di_test`, sans oracle par prise.
  L10 « moins bon » : d = +0,348, d_i = +0,359 ; +0,622 ; +0,064 (ESR plus haute d'un facteur 1,42 en moyenne géométrique) ; acceptable : non.
  L20 « non tranché » : d = −0,074, d_i = +0,254 ; −0,222 ; −0,252 ; acceptable : non (d₄₂ = +0,254 pour une borne de 0,25).
  Retard selon le bras : τ de 0,80 à 0,85 échantillon pour L10, 0,95 à 1,05 pour C, 1,00 à 1,05 pour L20 ; le calage divise l'ESR par environ 4,9 (C), 3,0 (L10) et 5,4 (L20).
  Lecture fixée d'avance : en cas de divergence, l'écart est attribué au calage sous ce protocole, sans autre cause ; la règle de décision reste celle de P.
  Preuve : F-0037, E-0050 ; `demo/train_length/predictions.md` (section « Co-lecture à retard compensé »), `demo/train_length/coreading_delay.py`, `demo/train_length/eval_train_length_aligned.json`, `demo/train_length/parity_gpu_cpu.json`.
  Statut : « predictions.md (6f62341) : règle écrite avant P, exécutée après P » ; étiquettes calculées à titre descriptif ; les ESR de test du banc existaient sur disque quand la règle a été écrite (non affichées par cette session, non exclu pour une session antérieure).
  Conditions : inférence GPU (écart d'instrument déclaré), parité GPU–CPU en ln ESR au plus 1,9·10⁻⁵ ; effet de bord du retard fractionnaire 8,6·10⁻¹³ pour une borne de 1,0·10⁻⁵ ; 0 run au bord de grille ; retard par rampe de phase FFT.

- **Piège de polarité de L1 + MR-STFT le long du gain de sortie (théorème T2).**
  Avec L = w₁ L1 + w₂ MRSTFT et ŷ = (−1 + ε) y, ŷ = −y est un minimum local strict du chemin de gain si w₁ E|y| < w₂ m, et n'en est pas un si w₁ E|y| > w₂ m (m : moyenne sur les résolutions de ρ_r + f_r).
  Preuve : `THEORY.md` (T2), `theory/T2_polarity_trap.md`, `theory/t2_check.py`.
  Statut : énoncé théorique, D (2026-10-02) au registre `THEORY.md` (preuve, contrôles symbolique et numérique, relecture Codex en deux passes) ; une direction, une solution exacte, un lot ; rien sur la dynamique ni sur la probabilité d'inversion ; corollaire non établi pour la tête tanh du banc.
  Conditions : perte du banc `scripts/product_nablafx_bench.py` (auraloss par défaut) ; marge numérique sur Big Muff en attente (voir En attente).

- **Une run SSM-WaveNet a convergé vers une sortie inversée.**
  ESR de test 3,95 au pas 5100, 0,052 une fois la sortie négée ; MR-STFT identique dans les deux cas (0,396 et 0,396) ; L1 0,0286 contre 0,0035 négée, proche de 2E|y| = 0,0299 (E|y| = 0,0149 sur la cible de test).
  Preuve : F-0005, E-0005 ; `paper/icassp2027/data/polarity_ssmzoh_seed42.json`, `paper/RESULTS_SSM_SEEDS.md` (section 1).
  Statut : constat post hoc, n = 1 ; compatible avec T2 (rétrodiction), non expliqué par lui.
  Conditions : ToneTwist Big Muff, protocole NablAFx, poids L1 1,0 et MR-STFT 0,1, discrétisation ZOH, exemption de weight decay seule, sans garde de polarité, graine 42.

- **Garde de polarité (`PolarityGuard`) : retournements tôt dans l'entraînement.**
  21 retournements sur 9 runs gardés du banc, jamais après l'époque 20 ; de 0 à 5 par run SSM-WaveNet ZOH.
  Run gardée de même graine : ESR 0,0358, 3,98 si l'on nie sa sortie.
  Preuve : F-0005, E-0005 ; `scripts/product_nablafx_bench.py` (classe `PolarityGuard`), `demo/nablafx_bench/ssmzoh_guard_seed42.json` et les huit autres fiches `*_guard_*`, `paper/icassp2027/data/polarity_ssmzoh_guard_seed42.json`.
  Statut : descriptif, post hoc ; un retournement du garde n'est pas une inversion finale.
  Conditions : 6 runs SSM-WaveNet ZOH (graines 42 à 44, répétitions) et 3 runs S4-TF-L-16, Big Muff.

- **Facultatif : sur DI calme, les modèles amplifient le résidu de quantification de l'entrée ; l'ESR ne le voit pas.**
  Sortie des modèles +7,2 dB au-dessus du plancher de la pédale (L médian 5,3, de 2,7 à 8,9) : −67,0 contre −74,2 dBFS ; cohérence avec la cible 0,0046 (passages forts 0,96) ; DI calme mis à zéro : −64,4 dB ; amplification d'environ 33 dB d'un résidu à −99,9 dBFS.
  Preuve : F-0029, E-0031 ; `diagnosis/noise/predictions_quiet.md`, `diagnosis/noise/quiet_frames.json`, `diagnosis/noise/control_zero_di.json`.
  Statut : préinscrit, contrôle déclaré avant de tourner ; issue du tableau préinscrit : « soutenue : exces propre aux passages calmes ».
  Conditions : 32 checkpoints SSM-WaveNet Big Muff du pilote C, 225 fenêtres calmes, inférence CPU ; cause (MR-STFT ou extrapolation) non attribuable ; on ignore si la pédale a reçu ce résidu.

## Préliminaire

- **VS100R : un retard oracle sous-échantillon renverse le classement entre bras.**
  Retard oracle médian (hors prise 16) de 0,80 à 0,85 échantillon pour A48, de 0,05 à 0,15 pour B192, selon la graine.
  Compensé, l'ESR de A48 est divisée par au moins 2 sur 6 prises sur 8 à chaque graine : de 2,2 à 9,4 à la graine 42, de 2,2 à 12,4 sur les trois graines ; au plus 1,21 sur les deux autres prises (2 et 16).
  En ESR brute, B192 est meilleur sur 5, 8 et 6 prises ; après compensation des deux bras, A48 l'est sur 7, 6 et 7.
  Preuve : F-0032, E-0040 (note de confusion reprise par E-0044) ; `DIAGNOSIS_vs100r_192k.md` (H6 et mises à jour des graines 43 et 44), `diagnosis/vs100r_192k/measures.json`, `diagnosis/vs100r_192k/measures_s43.json`, `diagnosis/vs100r_192k/measures_s44.json`.
  Statut : post hoc, descriptif ; retard choisi sur la cible de test de chaque prise (±2 échantillons) : il montre la sensibilité du classement, ce n'est pas une correction utilisable ; un oracle peut absorber une erreur de phase du modèle ; origine non établie (Q-0013).
  Conditions : une capture, 8 prises de test, 3 graines par bras.

- **VS100R : le modèle à 48 kHz est en avance d'environ un échantillon, à peu près constant en fréquence.**
  Un décalage commun de +1,0 échantillon (environ 21 µs) fait passer l'ESR médiane de 0,024 à 0,0024 sur 35 prises ; retard modèle → appareil de 0,97 à 1,26 échantillon par bande d'octave de 500 Hz à 8 kHz, alors que le modèle rend le retard de groupe dispersif de l'appareil.
  Même instrument sur d'autres modèles : décalage optimal médian −0,1 (Big Muff) et −0,1 (Marshall ToneTwist), contre +0,95 sur le matériau d'entraînement de la VS100R.
  Preuve : E-0037 ; `demo/capture/valvestate/timing_shift.py`, `demo/capture/valvestate/timing_shift.json`, `demo/capture/valvestate/group_delay.json`, `demo/capture/valvestate/timing_controls.json`.
  Statut : post hoc (écrit après l'évaluation, non déclaré dans `demo/capture/valvestate/predictions_model.md`) ; décalage commun choisi sur les prises notées.
  Conditions : un run (`vs100r_gc_film_joint_s925_r2`, graine 42), section de test, 48 kHz.

- **Longueur d'entraînement : contrôle post hoc avec τ estimé sur `di_test`.**
  L10 encore « moins bon » : d = +0,358, d_i = +0,359 ; +0,617 ; +0,097 (facteur 1,43) ; L20 « non tranché » : d = −0,031.
  τ de 0,75 à 0,85 pour L10, 1,00 à 1,05 pour C, 0,90 à 1,10 pour L20.
  Preuve : F-0037, E-0050 ; `demo/train_length/posthoc_test_delay.py`, `demo/train_length/eval_train_length_posthoc_test_tau.json`.
  Statut : post hoc, hors règle, non indépendant (`di_test` sert à ajuster τ puis à noter) ; ne change aucune étiquette ; aucune attribution causale.
  Conditions : celles de la co-lecture.

- **Trois modèles publiés sur Big Muff ont un L1 proche de 2E|y| : compatible avec une sortie inversée, non confirmé.**
  GCN-TF-2500-L-16, LSTM-32, TCN-2500-S-16 : ESR de test de 3,39 à 3,61 (une sortie exactement inversée donne 4) et L1 de 0,0293 à 0,0295, soit 0,98 à 0,99 fois 2E|y| ; les autres modèles publiés de la table ont une ESR d'au plus 0,70.
  Preuve : F-0005, E-0005 ; `paper/icassp2027/data/published_bigmuff.json` (annexe de Comunità, Steinmetz et Reiss 2025, tables 20 et 24), `paper/icassp2027/data/polarity_ssmzoh_seed42.json` (E|y|).
  Statut : post hoc ; non confirmé faute de checkpoints publiés ; E|y| mesuré sur notre fichier de test, supposé identique à leur jeu de test.
  Conditions : valeurs publiées, une run par modèle.

- **Taux d'inversion sans garde : non estimable.**
  En ZOH, la seule run sans garde est inversée ; en discrétisation `free`, 0 sur 3 (ESR d'au plus 0,236).
  Preuve : F-0005 ; `THEORY.md` (T2, prédiction T2-P1), `demo/nablafx_bench/ssmwavenet_seed42.json`, `demo/nablafx_bench/ssmwavenet_seed43.json`, `demo/nablafx_bench/ssmwavenet_seed44.json`.
  Statut : descriptif, n = 1 en ZOH ; « 0 inversion sur 5 » s'observerait environ une fois sur quatre sans aucun effet (`THEORY.md`).
  Conditions : Big Muff, poids L1 1,0 et MR-STFT 0,1.

## Retiré ou réfuté

- **Retiré le 2026-10-02 : « la MR-STFT, aveugle à la phase, ne voit pas un retard pur ».**
  Correction : la MR-STFT voit un retard de 4 échantillons sur les cibles F-Fuzz, linéairement en d, mais faiblement devant la MR-STFT de test des modèles (0,18 et 0,29) ; le retard porte en revanche toute l'ESR. Valeur du coût MR-STFT : en attente (aucune fiche JSON).
  Preuve : E-0039 ; `DIAGNOSIS_ffuzz.md` (correction datée en section 5), `FIX_ffuzz.md` (correction d'une prémisse de la section 1) ; l'invariant préinscrit du correctif et son issue ne changent pas.

- **Réfutées : les autres causes du plancher F-Fuzz, par leurs seuils écrits d'avance.**
  Agrégation (H3) : médiane 0,135 contre moyenne 0,137, les 10 % pires segments portent 0,20 du total (seuil 50 %).
  Aigu hors d'atteinte (H4) : part de l'erreur au-dessus de 5 kHz 0,17 en moyenne par segment (seuil 60 %), ESR après passe-bas à 5 kHz 0,81 fois l'ESR (seuil 50 %).
  Marqueurs (H5) : les 44 segments de bord portent 3,5 % du total (seuil 30 %).
  Contexte manquant (C8) : ESR intérieure 0,1364 contre ESR complète 0,1367.
  Bruit de la cible (H2) : résidu après réalignement de l'ordre de l'ESR réalignée (0,0013), loin des 11 à 19 % qu'il faudrait.
  Preuve : E-0027 ; `DIAGNOSIS_ffuzz.md` (section 4), `diagnosis/ffuzz/segments.json` ; SSM-WaveNet-FiLM, runs `_r2`.

- **Réfutée pour la prise 16 : « le retard explique l'écart de la prise 16 de la VS100R » (H6).**
  Retards oracles 1,45 (A48) et 0,85 (B192) ; après compensation ln(B/A) vaut encore 1,32. L'écart de cette prise dépend du run (H2 soutenue sur trois graines), pas de l'alignement.
  Preuve : E-0040 ; `DIAGNOSIS_vs100r_192k.md` (sections 4 et mises à jour), `diagnosis/vs100r_192k/measures.json`.
  Statut : hypothèse écrite avant la mesure (section 3 du diagnostic).

- **Retirées le 2026-10-03 après relecture Codex : « l'avantage apparent de B192 est un avantage d'alignement » et « invente un effet du contour ».**
  Elles dépassaient les mesures ; remplacées par « le classement entre les bras dépend de l'alignement temporel (descriptif) ».
  Preuve : `DIAGNOSIS_vs100r_192k.md` (section « Corrections du 2026-10-03 »).

- **Réfutée : « tout modèle ZOH entraîné à 48 kHz est en avance ».**
  Retard oracle médian des 6 modèles Big Muff ZOH et FOH de −0,175 à −0,075 échantillon. Ne réfute pas une contribution du ZOH sur la VS100R ; instrument différent (découpe puis décalage circulaire).
  Preuve : `DIAGNOSIS_vs100r_192k.md` (section 5, élimination du ZOH jugée trop large le 2026-10-03), `diagnosis/vs100r_192k/bigmuff_lead.json`.
  Statut : post hoc, mesure née pendant le diagnostic.

- **Attente non tenue (sans seuil) : τ du bras C proche de l'oracle par prise.**
  Attendu en valeur absolue proche du retard oracle de `DIAGNOSIS_vs100r_192k.md` ; obtenu 0,95 à 1,05 contre 0,80 à 0,85 : un retard commun par run estimé sur `di_validation` n'est pas l'oracle par prise de test.
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
