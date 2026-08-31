# Protocole AMP-QUALITY-TEACHER-v1

Cette lignée qualité seule demande si un teacher mono, causal et `float32` à
48 kHz dépasse le meilleur de trois comparateurs ouverts, reproductibles et
réentraînés localement sur un benchmark ToneTwist public gelé. Le contrat
normatif est `PROTOCOL_LOCK.yaml`; le fichier de configuration n'est qu'un
pointeur vers ce verrou.

Le contrôle `wavenet_x2_teacher_fast_only` partage le graphe et l'ordre
d'initialisation du candidat, avec FiLM S4 exactement nul. Ampeg seed 0 décide
prospectivement lequel continue. Trois dispositifs de développement choisissent
ensuite un comparateur global unique. Rodent et Fuzzy Logic restent scellés
jusqu'au gel du candidat, du comparateur, des recettes et des checkpoints.

La seule revendication positive possible est `GO-OBJECTIVE-SOTA`, limitée à la
fidélité objective sur ce benchmark public. CPU, C++, RTF, compression,
distillation, écoute et supériorité commerciale ou perceptuelle sont hors cycle.
