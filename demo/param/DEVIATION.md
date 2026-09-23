# Écart au protocole : entrées sèches de F-Fuzz

Écrit le 2026-09-23 à 22 h, après les deux runs F-Fuzz et avant leur relance.

**Constat.** `param_ssm_wavenet_ffuzz` et `param_s4_ffuzz` ont un ESR de 1,5 à 1,6 sur
tous les réglages, vus comme non vus, en test comme en validation : pire qu'un silence,
pour deux architectures différentes.

**Cause (mesurée).** J'avais téléchargé l'enregistrement « Dry » (Zenodo 10901426) alors
que la configuration nablafx de F-Fuzz utilise `DIGITAL-PARAMETRIC/DRY-with-markers`
(Zenodo 10455730). Les cibles F-Fuzz ont été rendues depuis les entrées avec marqueurs de
calage : elles sont plus longues de 2 à 6 s que le DRY sans marqueurs, et décalées de
2,000 s (5 fichiers) ou 2,999 s (1 fichier) par intercorrélation d'enveloppe ; seul
`nam` était aligné. Avec `DRY-with-markers`, les sept paires ont la même longueur au
sample près et un décalage d'enveloppe de −29 à +165 échantillons.

**Conséquence.** Les deux runs F-Fuzz sont invalides pour une raison de données, pas de
modèle. Ils restent dans `demo/RUNS.jsonl` et `demo/nablafx_bench/` et sont rapportés
comme échecs. Relance avec les entrées corrigées, **mêmes réglages et mêmes prédictions**
(rien d'autre ne change), sous de nouveaux run_id : `param_ssm_wavenet_ffuzz_r2`,
`param_s4_ffuzz_r2`. Les runs Marshall ne sont pas concernés (leur DRY est dans l'archive
Marshall).
