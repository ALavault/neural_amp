# FSSR-QUALITY-AA-v1

Cette lignée est la première étape du programme de qualité maximale sous
contrainte temps réel. Elle sélectionne uniquement un backend anti-aliasing
synthétiquement qualifié ; elle ne classe aucune architecture d'ampli.

Le test primaire mesure l'énergie des bins FFT qui ne sont ni DC ni des
harmoniques positives du sinus cohérent. DC est conservée comme métrique de
fidélité indépendante. Le plancher est calibré sur une identité float32 avant
tout rendu candidat, puis figé. Les références x8 et x16 sont directement
générées et doivent converger condition par condition.

Les routes x2 et x4 utilisent la même fixture et les mêmes coefficients
fonctionnels, avec une seule île pleine fréquence et 32 échantillons de
latence. ADAA n'entre dans la matrice que si un unique délai fractionnaire
causal explique toutes les fixtures à moins de 0,01 échantillon, et si sa
compensation native passe la parité dans le budget de 48 ; son exclusion
n'affecte pas les décisions x2/x4.

Un backend n'est admissible que s'il passe indépendamment les gates métrique,
fidélité, parité native, latence et temps réel. La décision finale est soit une
sélection sur le front qualité-coût, soit un rejet valide de toutes les routes.
Une défaillance d'instrumentation ou de provenance donne `INVALID`.
