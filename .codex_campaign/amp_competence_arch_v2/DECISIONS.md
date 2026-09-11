# Décisions

- `ARCH2-D-001` — Ouvrir une lignée v2 sans modifier ni réutiliser les runs v1.
- `ARCH2-D-002` — Qualifier le seul contrôle avant tout run candidat.
- `ARCH2-D-003` — Fixer le budget par passage des gardes au checkpoint courant et suivant, puis par plateau ESR médian entre 0 et 5 %.
- `ARCH2-D-004` — Utiliser un split de comparaison disjoint et réentraîner aussi le contrôle pour préserver l'appariement.
- `ARCH2-D-005` — Geler quatre candidats sans remplacement dépendant du résultat.
- `ARCH2-D-006` — Exclure tout audio physique, 192 kHz et FM9 de cette étape.
- `ARCH2-D-007` — Enregistrer `NO-GO-COMPETENCE-v2` : aucun checkpoint confirmé ne passe toutes les gardes.
- `ARCH2-D-008` — Lancer zéro run candidat conformément à la gate fail-closed.
- `ARCH2-D-009` — Ne pas réutiliser le contrat de comparaison : il est contaminé par une sélection v1 et confond topologie et supervision.
- `ARCH2-D-010` — Limiter le durcissement post-run à la vérification et à la documentation ; ne modifier ni protocole, ni runs, ni métriques, ni verdict.
