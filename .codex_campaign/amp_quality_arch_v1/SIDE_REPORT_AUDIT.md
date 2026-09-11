# Audit des rapports issus des side threads

## Artefacts

- `reports/steering/PHYSICS_FEATURE_CONDITIONING.md` : proposition d'un bus
  causal d'états physiques multi-échelles.
- `.codex_campaign/PROSPECTIVE_CAPTURE_SUPER_RESOLUTION.md` : proposition de
  reconstruction 48→192 kHz.
- `reports/steering/report-source.md` : autopsie et steering historiques sur
  loss, horizon, TFiLM et cascade.

## Décision

La super-résolution est rejetée de cette lignée : sans mesures physiques 192 kHz,
la bande supprimée et sa phase ne sont pas identifiables et ne peuvent constituer
une vérité terrain. Elle reste une idée de lignée future distincte.

Le bus physique est retenu sous forme d'hypothèse falsifiable, avec séparation
stricte des informations : seules l'entrée dry et les features causales qui en
dérivent entrent dans le modèle. Les descripteurs wet peuvent être des cibles
auxiliaires pendant l'entraînement, jamais des entrées d'inférence.

## Shortlist intégrée

1. `selective_s6_x2` — contrôle S6 pur compact.
2. `micro_tcn_x2` — contrôle convolutionnel local compact.
3. `phys_s6_tcn_x2` — observateur S6 lent, bus physique et micro-TCN x2;
   candidat principal conceptuel, sans promotion garantie.
4. `phys_det_tcn_x2` — bus déterministe et micro-TCN x2; ablation du caractère
   appris de l'observateur.
5. `rf2047_tfilm_x2` — horizon intermédiaire et modulation couche par couche.
6. `cascade_rf2047_tfilm_x2` — deux non-linéarités séparées et même contrôle
   RF/TFiLM, autorisé physiquement seulement après gate deux-clippers.

Les versions larges des trois premières familles peuvent servir de teachers sans
consommer une nouvelle famille. NAM A2, Wright LSTM, NablAFx TCN/S4 et S6 publié
restent des comparateurs séparés de cette shortlist.

## Test discriminant

Sur fixtures synthétiques, comparer à capacité déclarée le cœur seul, l'état lent
latent, le bus déterministe et l'observateur supervisé. Le bus physique doit
améliorer d'au moins 10 % la métrique agrégée transitions/sag/compression sans
régression statique supérieure à 5 %. Une amélioration qui disparaît à coût égal
réfute le mécanisme.

Le risque runtime domine : l'observateur doit rester hors de l'îlot x2, fonctionner
à cadence 64 et alimenter des modulations causalement alignées. Chaque squelette
déployable doit passer le benchmark C++ avant tout entraînement physique.
