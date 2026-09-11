# Échecs

## `NO-GO-COMPETENCE-v2` valide

Les neuf trajectoires du contrôle sont complètes. Le plateau médian entre 10 000 et 15 000 updates est de 4,114 %, donc dans l'intervalle gelé [0 %, 5 %]. Les gardes restent toutefois en échec aux deux checkpoints : à 15 000, le gain minimal vaut -0,251051 pour un seuil strict > -0,2 et la corrélation minimale 0,888893 pour un seuil strict > 0,9. Aucun budget confirmatoire n'est sélectionnable.

Le système statique converge près de la cible. En revanche, les trois seeds dynamiques et les trois seeds deux-clippers restent sous-gainés ; le seed dynamique 0 échoue aussi en corrélation. Cette structure exclut une panne globale d'entraînement et indique une faiblesse systématique du contrôle sur la dynamique/multi-non-linéarité sous ce protocole. Elle ne permet pas de trancher entre capacité, paramétrisation et objectif d'optimisation.

Aucun run candidat n'a été lancé et aucun run v1 n'a été repris ou retuné. Une nouvelle hypothèse exige une nouvelle lignée prospective.
