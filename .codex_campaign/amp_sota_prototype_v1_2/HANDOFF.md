# Handoff

Prochaine action autorisée : restaurer dans un commit séparé la fermeture du
snapshot historique requise par les tests suivis, puis réinvoquer le préflight
inchangé depuis un worktree propre. S'il passe, exécuter exactement le triplet
`dynamic_primary`, sans ouvrir les autres systèmes avant sa gate. Ne lire aucun
audio physique et ne générer aucune source v1.2 éligible à l'entraînement ou à
la sélection avant le préflight. Les seules fixtures de test permises restent
explicitement inéligibles.
