<!-- Workflow audit-amp-autocast (run wf_1dad7b94-624), 2026-10-04, lecture seule, sans GPU : 4 auditeurs, 4 vérificateurs sceptiques, 1 synthèse. Scripts de mesure des agents : /fastdata/lavaulta/tmp/ampaudit (hors dépôt). -->

# Audit de torch.autocast pour neural_amp

Périmètre : torch 2.13.0+cu130, Lightning 2.6.1, RTX PRO 4000 Blackwell (sm_120). Ces versions ont été relevées par la commande `.venv/bin/python -c "import torch, lightning…"` et par `nvidia-smi`. La doc consultée est la 2.12 (https://docs.pytorch.org/docs/2.12/amp.html), plus ancienne que la torch installée. Quand elles diffèrent, l'en-tête installé fait foi : `.venv/.../torch/include/ATen/autocast_mode.h`.

Aucune exécution n'a eu lieu sur CUDA. Les preuves viennent du device meta, du CPU, de la table du dispatcher et de la lecture du code.

## 1. Verdict

**Oui, sous conditions, et seulement pour SSM-WaveNet.**
- Conditions : forcer la convolution FFT du SSM en float32, utiliser `16-mixed`, mesurer contre la référence TF32 actuelle. Le gain attendu est modeste (estimation : 5 à 16 % de temps, environ 4 Gio de pic en moins).
- Le S4 de nablafx est à exclure.

## 2. Configuration recommandée

1. **Précision : float16 (`Trainer(precision="16-mixed")`), avec le GradScaler que Lightning crée de lui-même.**
   - Sources du réglage Lightning : `lightning/pytorch/plugins/precision/amp.py:53-58`, et `:114-123` pour l'autocast avec `cache_enabled=False`.
   - Ce qui départage float16 et bfloat16 : les Conv1d de la référence tournent déjà en TF32. C'est le cas par défaut (sortie de `torch.backends.cudnn.allow_tf32` → True), et les matmul y passent via `set_float32_matmul_precision("high")` (`scripts/product_nablafx_bench.py:736`). Le TF32 a une mantisse de 10 bits, comme le float16 ; le bfloat16 n'en a que 7 (`torch.finfo`).
   - Mesure CPU de l'audit `ssm` (modèles entraînés en fp32, puis évalués sous autocast) :
     - bf16 avec frontière : ESR 0,0113 → 0,0167 (film) et 0,0259 → 0,0339 (pôles), soit +31 à +48 % ;
     - fp16 : ESR quasi inchangée, 0,0114 et 0,0260.
   - L'audit `perf` mesure indépendamment une erreur de sortie de 2,6e-4 en bf16 contre 4,2e-6 en fp16.
   - **Repli bf16-mixed** : seulement si le fp16 produit des pas sautés fréquents. Lightning n'utilise pas de scaler en bf16 (`amp.py:55-56, 75`).

2. **Désactiver l'autocast autour de tout le corps de `DiagonalSSMLayer.forward`** (`src/fssr_nam/models/ssm_wavenet.py:196-207`) :
   - `with torch.autocast(device_type=x.device.type, enabled=False):`, avec `x = x.float()` et, si `shift` n'est pas None, `shift` converti en float ;
   - c'est le mécanisme documenté : `torch/amp/autocast_mode.py:180-200`.

   Cette frontière est obligatoire **dans les deux précisions** :
   - **bf16** : `torch.fft.rfft` refuse le bfloat16 avec « Unsupported dtype BFloat16 » (`SpectralOps.cpp:73-91` en v2.13.0 ; `torch/_refs/fft.py:58-79` ; test sur le device meta).
   - **fp16** : le terme DC de l'entrée du SSM dépasse le maximum du float16 (65504).
     - Mesures CPU sur des checkpoints réels : |Σx| jusqu'à 181 839 sur `nablafx_vs100r48_film_s43` (blocs 2, 3, 6 et 7 au-dessus de 65504), jusqu'à 184 846 sur `g2_poles_joint_s925`.
     - En émulation complex32, cela donne des inf, puis des NaN après `irfft`.
   - L'affirmation d'un des vérificateurs, selon laquelle le fp16 marcherait sans toucher au modèle, est donc réfutée.

3. **Ne rien ajouter d'autre en fp32 en fp16.** Mettre aussi en fp32 le résiduel, les skips et `pole_shifts` ne change rien de mesurable en fp16 (0,0261 → 0,0260). Cela ne sert qu'en bf16. Sources : `ssm_wavenet.py:611-628` ; décalages de pôles arrondis à 1,1e-2 en bf16 contre 1,1e-3 en fp16.

4. **Évaluation en fp32.**
   - Ne passer `precision` qu'au Trainer de fit (`product_nablafx_bench.py:790`) : ni au `tester` (`:854`), ni d'autocast dans `metrics_by_setting` (`:663-694`).
   - Validation sous autocast : Lightning applique aussi l'autocast à `validation_step` (`lightning/pytorch/strategies/strategy.py:409-412`, `plugins/precision/precision.py:175-185`). `loss/val/tot`, qui pilote best.ckpt, EarlyStopping et ReduceLROnPlateau, est donc calculée en précision mixte.
   - Pour comparer à fp32, rapporter `last.ckpt` évalué par le tester fp32.

5. **Ajouter `precision` au record.** Aujourd'hui, le record ne permet pas de distinguer un run AMP (`product_nablafx_bench.py:877-953`).

6. **Modes à ne pas utiliser :**
   - `16-true` ni `bf16-true` : ils convertissent tout le module, y compris les paramètres du SSM (`lightning/pytorch/plugins/precision/half.py:43-44`).
   - Un autocast manuel dans `common_step` : il n'aurait ni GradScaler ni déscalage avant l'écrêtage.

7. **S4 de nablafx : hors AMP.**
   - En fp16, la longueur FFT `n = 2*l = 288000` n'est pas une puissance de 2, ce que cuFFT refuse en half (chaîne présente dans `libtorch_cuda.so` ; `third_party/nablafx/nablafx/processors/blocks.py:350-352`).
   - En bf16, la FFT est refusée de toute façon.
   - Isoler la FFT ne suffit pas : `out = torch.zeros_like(u)` reste en basse précision, et `out[idxs, :, :l] = y` lève l'erreur « Index put requires the source and destination dtypes match » (reproduite sur CPU ; `blocks.py:344`).
   - Il faudrait donc forcer tout le forward en fp32, et l'AMP n'apporterait presque plus rien.

**Gouvernance.** La frontière modifie `src/fssr_nam/models/ssm_wavenet.py`, qui est hors de la voie produit (`AGENTS.md` : `demo/`, `src/fssr_nam/product/`, `scripts/product_*.py`). Il faut donc passer la porte A du §5 avant tout usage.

## 3. Risques classés

| # | Risque | Gravité | Source |
|---|---|---|---|
| 1 | Sans frontière, en fp16 : débordement du DC de la rfft complex32 → inf/NaN → pas sautés en série | élevé (mesuré sur CPU, conséquence GPU extrapolée) | mesures `dc_check.py` / `amp_audit.py` ; `ssm_wavenet.py:204` ; https://docs.pytorch.org/docs/2.12/amp.html (« Ops not listed below… run in the type defined by their inputs ») ; aucune fft dans `autocast_mode.h:820-957` |
| 2 | Sans frontière, en bf16 : RuntimeError dès le premier pas | élevé | `SpectralOps.cpp:73-91` ; test meta ; conv1d en basse précision à `autocast_mode.h:823` |
| 3 | Échec silencieux en fp16 : un pas sauté incrémente quand même `global_step`. `FixedStepSchedule` et `max_steps` comptent alors des pas Lightning, pas des mises à jour AdamW. Avec `--no-early-stopping`, rien n'arrête un run dont la perte n'est plus finie | moyen à élevé | `lightning/pytorch/loops/optimization/automatic.py:270-280` ; `training_epoch_loop.py:113-117` ; `product_nablafx_bench.py:351-361, 811` ; `callbacks/early_stopping.py:122, 249` |
| 4 | Sélection biaisée : `loss/val/tot` calculée sous autocast, alors que le plancher de bruit a été mesuré en fp32 (4 à 9e-3) | moyen | `precision.py:175-185` ; `product_nablafx_bench.py:258-261, 279-295` |
| 5 | Les portes de reproductibilité au bit ne s'appliquent plus. `repro_gate.py` réécrit `repro_gate.json` sans garde et n'enregistre jamais une nouvelle porte : sa garde porte sur le préfixe `"repro_gate_` | moyen | `diagnosis/seeds/repro_gate.py:27-31, 133-136` |
| 6 | Déterminisme de cuFFT en half, de cuDNN fp16 et du GradScaler sous `use_deterministic_algorithms` : non documenté | moyen | docstring `torch/__init__.py:1388-1560` (aucune mention) ; amp.md 2.12 (aucune occurrence de « determin ») |
| 7 | Checkpoints : l'état du scaler est stocké sous la clé `MixedPrecision`. Un `--resume` d'un checkpoint écrit dans une autre précision perd le scaler ou le réinitialise à 65536 | moyen | `lightning/pytorch/trainer/connectors/checkpoint_connector.py:311-320, 475-479` |
| 8 | Accumulation en basse précision du résiduel et de `skip_sum` sur 8 blocs (fp16 : effet mesuré négligeable en évaluation ; en entraînement, non mesuré) | faible en fp16 | `ssm_wavenet.py:472, 611-615` |
| 9 | Écrêtage par valeur : correct en fp16, car `unscale_` a lieu avant le clip. En bf16, un inf devient ±clip sans être signalé (rare, vu la plage du bf16) | faible | `amp.py:84-97` ; `precision.py:84-94, 158-161` ; test CPU `clip_grad_value_` |

Points non problématiques, vérifiés :
- La construction des noyaux n'est pas touchée. Les complexes et le float64 ne sont pas éligibles (`autocast_mode.h:143-152, 315-320`), et les noyaux sont identiques au bit près avec et sans autocast (`amp_audit.py`, CPU).
- La MR-STFT d'auraloss sort en complex64 : la fenêtre est un attribut float32, pas un buffer (`third_party/auraloss/auraloss/freq.py:120, 226`). Vérifié sur meta et CPU. Le gradient renvoyé au modèle repasse toutefois en fp16.
- La L1 est forcée en fp32 (`autocast_mode.h:888`).
- Le checkpoint non réentrant rejoue l'état d'autocast pendant le recalcul (`torch/utils/checkpoint.py:1629, 1657-1685`).

## 4. Gain attendu (estimé, comptage sur le device meta, non mesuré sur GPU)

La référence est mesurée dans `demo/nablafx_bench/vs100r192_film_s42.json:674-675` : 19,43 Gio de pic, 1,16 s/pas.

Le pas est limité par la bande passante : 115 GFLOP par pas, soit environ 3 ms à 40 TFLOPS, pour environ 235 Go lus et écrits (`meta_count.py`). Les tensor cores n'apportent donc rien. Le trafic n'explique que 30 à 50 % du temps : les fourchettes de vitesse ci-dessous suivent la loi d'Amdahl et ne sont pas des prédictions.

| Levier | Trafic | Temps | Pic estimé | Risque numérique |
|---|---|---|---|---|
| AMP avec FFT en fp32 | −16 % | −5 à −16 % | ≈ 15,4 Gio | ESR à valider |
| FFT de 288 000 points en fp32 (sans AMP) | −22 % | −7 à −22 % (si cuFFT reste aussi efficace sur 2^8·3^2·5^3 : non vérifié) | ≈ 15,6 Gio | aucun en dynamique ; les bits de référence changent |
| AMP + FFT de 288 000 points | −39 % | −13 à −39 % | ≈ 11,6 Gio | cumul des deux |
| Checkpointing par bloc (sans AMP) | +32 % | +10 à +32 % | ≈ 5-6 Gio | aucun (levier mémoire seulement) |

Conclusion : le passage à une FFT de 288 000 points rapporte au moins autant que l'AMP, sans risque de dynamique. 288000 ≥ 2L−1 = 287999 : la convolution reste linéaire.

Facteur de confusion mesuré dans les records : le temps par pas varie de 1,10 à 1,54 s d'un run à l'autre en mode déterministe, et vaut 0,89-0,92 s en non déterministe sans FiLM. Ces deux écarts dépassent le gain attendu de l'AMP.

## 5. Plan de test minimal (à lancer plus tard ; le GPU avait un processus actif lors de l'audit)

0. **Contrôle CUDA en un pas, avant tout run.** Vérifier :
   - le dtype de sortie de conv1d sous autocast ;
   - max|x_f| < 65504 sans frontière, pour confirmer le diagnostic ;
   - la sortie complex64 avec la frontière ;
   - la sortie complex64 de `stft` ;
   - l'absence d'erreur sous `use_deterministic_algorithms(True)`.
1. **Porte A, non-régression fp32.** Après l'ajout de la frontière et de `--precision` (défaut `32-true`), relancer les trois PAIRS de `repro_gate.py` : résultat attendu IDENTIQUE. C'est obligatoire, puisque `src/` est modifié.
2. **Run court.** Même configuration que `vs100r_probe_leaky_g0_s42` : SSM-WaveNet, 600 pas, `--deterministic`, référence mesurée de 19,42 Gio et 11,58 min (`vs100r_probe_leaky_g0_s42.json:183, 201, 204-205`).
   - Deux bras : `32-true` et `16-mixed` avec frontière.
   - Au moins 2 graines par bras, même mode déterministe, même session GPU.
   - Journaliser `trainer.precision_plugin.scaler.get_scale()` et le nombre de pas sautés ; comparer à nombre égal de mises à jour.
   - Enregistrer les runs dans `demo/RUNS.jsonl`, avec la précision.
   - Écrire la porte B (AMP contre AMP au bit) dans un fichier séparé, pas dans `repro_gate.json`.
3. **Critères proposés.** Ce sont des seuils de décision à fixer par l'utilisateur, pas des valeurs tirées des données.
   - **ESR** : `test_last` évalué par le tester fp32. L'écart à fp32 doit rester dans la dispersion inter-graines déjà mesurée.
   - **NaN/inf** : aucun. Pas sautés : < 1 % des pas.
   - **Pic mémoire** (`peak_gpu_gib`) : ≤ 16 Gio, attendu ≈ 15,4.
   - **Temps par pas** : gain ≥ 10 %, au-delà de la dispersion entre graines du même bras. Sinon, ne pas adopter, et essayer d'abord la FFT de 288 000 points.

## 6. Ce qui reste invérifié

- Aucune exécution CUDA :
  - comportement réel de cuFFT en half (y compris l'avertissement « ComplexHalf support is experimental ») ;
  - fréquence des pas sautés ;
  - déterminisme sous AMP.
- Aucun entraînement sous AMP. Les ESR viennent de modèles entraînés en fp32 et évalués sur CPU : une seule prise de 3 s à un seul réglage (audit `ssm`), un extrait de 144 000 échantillons (audit `perf`). La croissance des activations pendant l'entraînement n'est pas mesurée.
- Le débordement DC a été mesuré sur des checkpoints à 48 kHz, pas sur le modèle à 192 kHz.
- Gains de vitesse et de mémoire : projections meta. Environ 0,6 à 0,8 s par pas reste inexpliqué ; seul un `torch.profiler`, en mode déterministe et non déterministe, peut trancher.
- L'efficacité de cuFFT sur 288 000 points n'est pas vérifiée.
- L'écart entre la doc 2.12 et la torch 2.13 installée : par exemple, `einsum` figure dans la liste basse précision de l'en-tête (`autocast_mode.h:836`) mais pas dans la doc. C'est sans effet ici, les einsum portant sur des complexes.
- L'effet sur l'inférence (`step()`, `cpp/`) de poids entraînés sous AMP puis exécutés en fp32 n'est pas mesuré.
- Affirmations des rapports réfutées ou corrigées, donc écartées de ce rapport :
  - la conversion silencieuse dans le S4 (en réalité une erreur `index_put`) ;
  - « +48 à 61 % » (en réalité +31 à +48 % avec frontière, +50 à +61 % sans) ;
  - le scénario d'une fenêtre STFT castée en fp16 ;
  - des listes fp16 et bf16 « partagées sur CUDA » d'après la doc ;
  - un checkpoint qui ne couvrirait « que le noyau » (il couvre toute la couche SSM) ;
  - un « même nombre de trames à 192 kHz » (il y en a 4 fois moins) ;
  - la citation `checkpoint.py:310-313` ;
  - « 1 à 38 raies » (en réalité 0 à 38) ;
  - les 72 987 NaN, invérifiables ;
  - « `--deterministic` reste applicable sous AMP », invérifiable.
