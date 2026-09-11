# Décisions

- `full_island_x2` et `teacher_x4` passent le gate scientifique.
- Les deux routes atteignent le plancher de résidu connu et sont dans la bande d'indifférence qualité de 0,5 dB.
- `teacher_x4` est rejeté du profil temps réel : son p95 bloc 64 dépasse la limite préenregistrée.
- `full_island_x2` est sélectionné : parité, latence 32 et p95 temps réel passent.
- ADAA reste exclue sans affecter x2/x4 faute d'un délai global compensable et d'une parité de compensation native.
