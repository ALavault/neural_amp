# Matrice de clôture — AMP-QUALITY-TEACHER-v9

Mise à jour initiale : 2026-09-05. Cette matrice est fail-closed : `supporté`
signifie qu'un artefact autoritatif existe; `en attente` ne constitue jamais une
preuve de réussite.

| Exigence du Goal | Preuve autoritative requise | État actuel |
|---|---|---|
| Cadrage Deep Research | `report-source.md`, corpus, journal de recherche, registre claim→source, PDF vérifié | Supporté |
| Lacunes de validité identifiées | Audit traçant asymétrie d'historique, pondération validation, latence comparateur et portée SOTA | Supporté par rapport et registre C09–C12 |
| Protocole prospectif distinct | `PROTOCOL_LOCK.yaml`, `ACTIVE_CAMPAIGN`, lignée v8 terminale immuable, zéro réutilisation de checkpoints | Supporté dans le commit v9 préenregistré |
| Watchdog 48 h par trajectoire | Code qualifié, test/frontière et `resolved_config.yaml` de chaque run | Code supporté; comportement long en attente |
| Plafond global 300 GPU-h | Ledger monotone, refus avant dépassement, budget final cohérent avec les runs | Code supporté; aucune consommation v9 encore mesurée |
| Qualification avant run | Tests complets, Ruff, catalogue, artefact canonique lié au snapshot | Supporté : 456 tests et qualification v9 |
| Préflight synthétique GPU | Trois comparateurs : forward 48k, loss source-coordinate, gradients finis, mémoire <90 %, zéro audio physique | En attente d'admission GPU |
| Développement exact | 11/11 nouveaux run IDs v9 terminaux et complets, checkpoints choisis par validation commune | 0/11; non commencé |
| Failure mining réel→généré | Fenêtres validation appariées, facteurs dry-only, résidus multi-métriques, `selection_eligible=false` | Code supporté; sorties réelles absentes |
| Sélection comparateur global | Neuf runs comparateurs complets; ESR également pondéré par dispositif; tie-break gelé | Manquante |
| Locks candidat/comparateur/recettes | Artefacts append-only antérieurs à toute résolution confirmation/test | Manquants |
| Disponibilité confirmation | Zenodo, checksum, licence, capacité de stockage | Identité/checksum/CC BY NC supportés; version 4.0 à vérifier à la gate |
| Audit Rodent/Fuzzy Logic | Seulement train/validation; marqueurs, mono, 48 kHz, longueur, finitude, énergie, clipping; promotion atomique | Non autorisé avant locks |
| Confirmation exacte | 2 dispositifs × 2 familles × seeds 0–2 = 12 runs complets | 0/12 |
| Locks checkpoints confirmation | 12 sélections par perte totale validation uniquement | Manquants |
| Ouverture test | Les cinq tests ouverts simultanément après tous les locks; compteur préalable nul | Tests encore scellés; compteur déclaré 0 |
| Verdict objectif | Seuils gelés, Rodent/Fuzzy seuls décisionnels, bootstrap hiérarchique déterministe 10 000 | Manquant |
| Clôture scientifique | État, maturité, décisions, échecs, revendications, handoff, ledgers et verdict cohérents | Manquante |

## Conditions d'arrêt explicites

- Budget v9 épuisé avant matrice complète : arrêt, exécution incomplète, pas de
  revendication SOTA.
- Archive ou checksum absent : arrêt avant lecture test.
- Licence exacte incompatible ou non vérifiable à la gate : `INVALID`.
- Besoin de changer topologie, métrique, seuil, contexte, split ou accès après
  préenregistrement : arrêt et steering; aucune correction in-place.
- OOM, instrumentation incohérente ou matrice incomplète : conserver les
  artefacts et ledger; ne jamais compléter silencieusement par un ancien run.

## Critère de Goal terminé

Le Goal n'est complet que lorsqu'un artefact final contient exactement
`GO-OBJECTIVE-SOTA` ou `NO-GO-OBJECTIVE-SOTA`, que son bootstrap compte 10 000
réplications, que les 12 runs de confirmation et tous leurs checkpoints sont
présents, et que l'audit final reproduit le verdict depuis les artefacts sans
accès anticipé aux tests.
