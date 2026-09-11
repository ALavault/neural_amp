# Audit technique AMP-COMPETENCE-ARCH-v2

- Verdict : `NO-GO-COMPETENCE-v2`.
- Runs de compétence complets : 9/9.
- Runs de comparaison complets : 0/0.
- Reprises ou retunings v1 : 0.
- Échantillons audio physiques lus : 0.
- Blackstar, UA1176 et EXTERNAL_REPORT_ONLY sont restés verrouillés.
- Les neuf IDs attendus sont les seules entrées v2 du registre global.
- Tous les checkpoints attendus se chargent et leurs tenseurs sont finis.
- Les snapshots de source des neuf runs sont identiques octet par octet.
- La gate de compétence est recalculée à l'identique depuis les trajectoires.
- `make test` et `make lint` passent dans l'audit de clôture.

## Findings adversariaux

- `ARCH2-AF-001` — Le poids auxiliaire `0.01` de `phys_s6_tcn_x2` provient de la sélection v1, contrairement à la déclaration d'indépendance de sélection v2.
- `ARCH2-AF-002` — `phys_s6_tcn_x2` et `rf2047_tfilm_x2` instancient la même topologie observer-conditioned ; seule la supervision auxiliaire diffère. La comparaison aurait donc confondu architecture et loss.
- Ces deux findings n'affectent pas le verdict de compétence : le contrôle utilise un poids auxiliaire nul et aucun candidat n'a été lancé. Ils interdisent en revanche de réutiliser le contrat de comparaison v2.

## Limites techniques restantes

- Les snapshots prouvent l'identité interne entre runs, pas une référence historique externe.
- Le mode CUDA déterministe et l'identité exacte de l'accélérateur ne sont pas imposés par le code d'entraînement.
- Le chemin d'exception d'un run est moins fail-closed que les gates scientifiques.
