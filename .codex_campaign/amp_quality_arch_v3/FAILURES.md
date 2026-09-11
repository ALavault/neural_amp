# Échecs

Aucun run scientifique v3 n'a été lancé. Un défaut technique pré-run a été détecté :
les états persistants du streaming conservaient le graphe PyTorch entre chunks TBPTT,
provoquant un second backward invalide. La frontière détache désormais explicitement
les états. Un second biais pré-run donnait le poids d'une update entière au reliquat
de 256 échantillons en fin d'épisode ; chaque update couvre désormais exactement 8192
échantillons et traverse la frontière avec reset/pré-roll si nécessaire. Les trois
familles passent deux chunks backward consécutifs et une mise à jour CUDA déterministe.
Les échecs v2 restent historiques et ne servent que
d'hypothèses prospectives de plage de sortie et de mémoire.

## ARCH3-F-003 — Interruption externe du round 1

Le job `job-mte87wc2-6e549f1d` a disparu pendant `arch_v3_round_1_two_clippers_primary__slow_state_micro_tcn_x2_seed0_v1` après 15 trajectoires complètes. Aucun résultat de la trajectoire interrompue n'existe. La round entière est `INVALID`; reprise et retry restent interdits.
