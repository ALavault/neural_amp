# Échecs

Aucun run v1.2 n'a été lancé. Les échecs des lignées parentes restent dans leurs
répertoires immuables et ne sont pas réinterprétés ici.

## Préflight, tentative 001

La première invocation a passé `make data-audit`, puis `make test` a échoué
pendant la collecte avec quatre `ModuleNotFoundError` historiques. Aucun test
n'a été exécuté, `make lint` n'a pas été lancé, aucun gate n'a été évalué et
aucune sortie synthétique éligible n'a été produite. L'enregistrement exact est
`PREFLIGHT_ATTEMPT_001_INVALID.json`. La cause est la fermeture incomplète du
snapshot Git, pas le modèle ni le protocole scientifique.
