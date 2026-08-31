# État

- Lignée : `AMP-QUALITY-TEACHER-v1`.
- Statut : prospective, préflight en attente après une tentative
  d'infrastructure invalide.
- Parent : `AMP-SOTA-PROTOTYPE-v1.2`, supersédé après préflight passé et avant
  tout run scientifique.
- Runs scientifiques : 0.
- Gates évalués : 0.
- Tentatives de préflight invalides pour infrastructure : 2.
- Waveforms de test lus : 0.
- Archives Rodent/Fuzzy Logic téléchargées : non.
- Blackstar, UA1176 et `EXTERNAL_REPORT_ONLY` : fermés.
- La tentative 001 a passé `data-audit` et 424 tests, puis échoué sur le lint du
  snapshot propre; aucune preuve scientifique n'en est tirée.
- La tentative 002 a passé `data-audit`; 424 tests sur 425 ont passé, puis un
  test d'état canonique devenu obsolète après la tentative 001 a échoué.
- Prochaine action autorisée : corriger uniquement cet invariant de test, puis
  réexécuter le préflight inchangé, sans accès aux waveforms scellés.
