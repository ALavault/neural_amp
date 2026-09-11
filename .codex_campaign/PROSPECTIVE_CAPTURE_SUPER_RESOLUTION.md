# Piste prospective — super-résolution de captures

Statut : idée de recherche non préenregistrée, hors du protocole aliasing actif.

## Intention

Étudier une migration en deux temps des captures d'amplificateurs et d'effets
vers des fréquences d'échantillonnage supérieures.

1. À partir de captures physiques synchrones à 192 kHz, apprendre à reconstruire
   la sortie haute fréquence depuis leurs versions décimées à 96/48 kHz.
2. Appliquer le prior ainsi appris aux captures NAM ou équivalentes limitées à
   48 kHz afin de produire des versions haute fréquence avec information
   imputée.

Le reconstructeur devrait être conditionné par l'excitation dry et la sortie
wet basse fréquence. Sa construction devra préserver la mesure existante : la
redécimation gelée de la reconstruction devra reproduire la capture source.
L'information ajoutée devra rester dans le complément perdu par la décimation.

## Niveaux de preuve

- `MEASURED_HR` : capture physique 192 kHz, admissible comme vérité terrain.
- `RECONSTRUCTED_HR` : reconstruction d'une décimation d'une capture 192 kHz
  connue, utilisée pour valider la méthode.
- `IMPUTED_HR` : extrapolation d'une véritable capture historique 48 kHz,
  utilisable pour préentraînement ou augmentation, jamais comme holdout ou
  vérité terrain confirmatoire.

Une sortie imputée ne doit pas être présentée comme une mesure restaurée. Le
modèle devra idéalement produire une incertitude ou plusieurs réalisations
plausibles, car la bande supprimée et sa phase ne sont pas identifiables de
façon unique.

## Risques à traiter

- écart entre une décimation FIR propre et la chaîne ADC/DAC d'une capture NAM
  réellement réalisée à 48 kHz ;
- hallucination de continuation harmonique et de phase ;
- mémorisation des familles d'appareils au lieu d'une généralisation ;
- alias déjà repliés et donc ambigus dans les captures historiques ;
- confusion entre données mesurées et données synthétiques imputées.

La campagne devra employer des splits par appareil, réglage, session et chaîne
de capture, dont un test leave-device-family-out sur de vraies mesures 192 kHz.

## Test décisif proposé

1. Masquer les sorties 192 kHz d'appareils tenus à l'écart et reconstruire à
   partir de leurs seules versions 48 kHz.
2. Comparer à l'upsampling déterministe sur la bande haute, les harmoniques
   complexes, l'IMD, la cohérence de phase et la redécimation 48 kHz.
3. Entraîner le même ampli neuronal avec et sans captures `IMPUTED_HR`.
4. Juger l'utilité uniquement sur des captures physiques 192 kHz scellées,
   ainsi que sur leur décimation gelée à 48 kHz et le coût runtime final.

Cette piste pourra devenir une lignée séparée après qualification du banc
aliasing. Elle ne modifie pas les runs, seuils, verdicts ou conclusions R1/R2
et ne fait pas partie de la décision du goal aliasing actuel.
