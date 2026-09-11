# FSSR-QUALITY-AA-v2

Cette lignée reprend sans changement scientifique le protocole
`FSSR-QUALITY-AA-v1`. Le seul amendement est le typage du transport : toute clé
de mapping doit être une chaîne et la clé YAML ambiguë `off` est normalisée en
chaîne avant sérialisation. Les références observées par v1 ne sont pas
réutilisées et aucune sortie x2/x4 n'a été observée avant ce gel.

Toutes les métriques, fixtures, routes, latences, filtres, seuils et règles de
sélection restent identiques à v1.
