# Protocole d’exécution FSSR-NAM

## Mission normative

Conduire une campagne reproductible afin de déterminer si un modèle **Fast–Slow Structured Residual NAM (FSSR-NAM)** surpasse NAM A2 Full à coût CPU comparable (H1), atteint une fidélité non inférieure avec au moins 25 % de gain CPU (H2), ou réduit d’au moins 3 dB les composantes spectrales parasites par antialiasing local sans supprimer les harmoniques utiles (H3).

Le mandat détaillé fourni le 27 août 2026 gouverne cette campagne. Ce fichier en fixe les contrôles opérationnels dans le dépôt ; les seuils et interdictions ci-dessous sont normatifs.

## Contraintes

- Audio mono, causal, 48 kHz, `float32`, sans anticipation ni latence cachée.
- Un modèle par dispositif ; entraînement sur un GPU de 24 Go.
- Comparaison finale CPU par blocs contre NAM A2 Full, avec médiane, p95, facteur temps réel, état, latence et taille.
- Données séparées par fichier source et, si possible, interprète, instrument, session et réglage ; jamais par fenêtres voisines aléatoires.
- Phases strictes : `CORE → MATURATION → EXPLORATION → RESERVE`.

## Contrôle scientifique

Chaque affirmation doit pointer vers un artefact. Les runs, y compris les échecs, sont immuables et inscrits dans `.codex_campaign/RUN_LEDGER.jsonl`. Aucune seed défavorable ne peut être retirée sans règle préenregistrée. Les coûts théoriques complètent mais ne remplacent jamais les mesures CPU.

Les niveaux de données sont `SYNTHETIC`, `INTERNAL_DEV`, `INTERNAL_VALIDATION` et `EXTERNAL_REPORT_ONLY`. Les résultats externes restent inaccessibles tant que l’architecture, les pertes, l’entraînement, les hyperparamètres, les seeds, les métriques, les exclusions et le code d’évaluation ne sont pas gelés, et que `external_retest_authorized` n’est pas vrai dans le gel versionné.

## Gates

1. **M0** — dépôt, environnement, données inventoriées, NAM figé, tests et registre opérationnels ; identité synthétique générée et relue.
2. **M1** — systèmes synthétiques, alignement, décimation et métriques validés contre les perturbations prescrites.
3. **M2** — A2 Full/Lite entraînés, exportés, validés en blocs et benchmarkés sur au moins deux seeds.
4. **M3** — S0 à S4 (et S5 seulement après validation ADAA) causalement et numériquement validés ; apprentissages synthétiques réussis.
5. **M4** — 24 entraînements smoke : deux dispositifs, B0/B2/S3/S4, seeds 0–2 ; gate qualité, efficacité ou antialiasing.
6. **M5** — recherche bornée, ablations, protocole gelé et campagne interne finale multi-dispositifs.
7. **M6** — analyse hiérarchique, parité Python/C++, figures, tables, audio, audit adversarial et dossier de publication.

En cas d’échec M4/M5, l’autopsie examine d’abord alignement, calibration, fuite, contexte, états, clipping, énergie résiduelle, apprentissage, allocation du budget, métriques et coût réel. Aucun élargissement architectural ne précède cette autopsie.

## Architecture et validation minimales

Le modèle additionne un cœur FIR causal préfiltre–spline–postfiltre, un petit état lent GRU décimé (départ `R=64`, état 8 ou 16), un TCN résiduel causal court et une voie directe optionnelle. L’antialiasing est étudié dans l’ordre : naïf, suréchantillonnage local ×2, ×4 si prometteur, puis ADAA validée. Le résidu est régularisé et son ratio d’énergie est rapporté.

Les métriques couvrent erreur temporelle, MR-STFT, harmoniques complexes, intermodulation, énergie inharmonique, enveloppe/transitoires et efficacité. Avant comparaison, elles doivent réagir de façon documentée au gain, délais entier/fractionnaire, polarité, DC, passe-bas, bruit, suppression harmonique, composantes inharmoniques, aliasing synthétique, ringing et changement de phase.

## Provenance obligatoire

Chaque entraînement ou benchmark crée `experiments/runs/<run_id>/` avec configuration résolue, commande, commit, environnement, manifestes de données et split, seed, journaux, métriques, timings, checkpoints, prédictions, figures et statut. Aucun run n’est remplacé.

Après chaque étape importante, mettre à jour `STATE.md`, `MATURITY.json`, `DECISIONS.md`, `FAILURES.md`, `CLAIMS.md` et `HANDOFF.md`, puis produire un commit local descriptif. Ne jamais pousser vers un dépôt distant sans autorisation.

## Verdict

Le classement final est `GO-A` (H1), `GO-B` (H2), `GO-C` (H3), `GO-D` (gain robuste de la branche lente), ou `NO-GO`. Un résultat négatif reste un résultat et doit produire le même registre, les limites et l’audit final sans reformulation opportuniste des critères.
