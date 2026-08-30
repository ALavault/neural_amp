# Protocole AMP-SOTA-PROTOTYPE-v1.2

Cette lignée teste d'abord la compétence du prototype complet
`slow_long_tcn_x2`, puis la valeur causale de son chemin lent contre le même
graphe avec modulation forcée à zéro. Elle ne rejoue ni le gate antialiasing x2,
ni les approximants v1.1.

Le contrat normatif est `PROTOCOL_LOCK.yaml`. Toute observation scientifique
éligible avant le passage du préflight rend la lignée invalide. Les tests peuvent
matérialiser des fixtures inéligibles. Si les trois gates de compétence passent,
la valeur lente est mesurée sur `SLOW_VALUE_EVAL`, jamais sur l'`INTERNAL_DEV`
ayant servi aux sentinelles de compétence.
