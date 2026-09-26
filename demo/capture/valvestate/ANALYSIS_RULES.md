# Séance VS100R : règles d'analyse fixées avant les prises

Voie produit. Écrit le 2026-09-26, avant toute prise. L'utilisateur prévient qu'au gain 0 du canal
OD2, l'ampli ne sort quasiment aucun signal ; le protocole de capture ne change pas.

- **Prises concernées d'avance** (`settings.csv`) :
  - gain 0 : prises 6, 7, 10 et 27, toutes d'entraînement ;
  - gain 2,5, peut-être faibles aussi : prises 5, 26 et 32 (entraînement), 16 et 36 (test).
- **Alignement** : la latence de la chaîne, mesurée par les prises de bouclage, est appliquée à
  toutes les prises. Les bips de chaque prise ne servent qu'à détecter une perte d'échantillons, et
  seulement quand ils dépassent le bruit.
- **Rapport signal sur bruit (RSB)**, par prise et par voie, après alignement : RMS des sections DI
  (entraînement, validation, test) sur RMS de la section de silence de 5 s qui suit les bips de
  début.
- **Seuil : RSB < 20 dB → prise « faible signal ».** À 20 dB, le bruit seul impose un plancher
  d'ESR de 0,01, soit environ un tiers des ESR de test habituels.
  - En entraînement, la prise est gardée : le modèle doit apprendre qu'un gain de 0 éteint
    presque l'ampli.
  - En évaluation, son ESR est rapporté à part, hors des médianes, avec l'erreur de niveau RMS en
    dB.
- Le seuil ne change pas après lecture des chiffres, et aucune prise faible n'est remplacée ni
  refaite à un autre niveau.
