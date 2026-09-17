# Addendum au pilote A — bras « rejoue » à la fourche 5

Écrit le 2026-09-17, avant tout run de la fourche 5, après lecture des runs de la
fourche 100 « décide » k = 1 à 3 et « rejoue » k = 1 et 2. Le verdict de la fourche 100
n'est pas établi (k = 3 « rejoue » et k = 4 manquent). Les prédictions de `pilot_A.md`
ne changent pas.

## Motivation

À la fourche 100, sous le calendrier du témoin, les enfants perturbés s'écartent de sa
courbe de validation puis la rejoignent vers l'époque 300 : écart de sortie au témoin
de 4 à 5·10⁻⁴ en ESR, Δ de −0,007 et −0,015. Les enfants qui décident eux-mêmes divisent
le learning rate à d'autres époques et s'en écartent : écart de sortie de 1·10⁻³ à
7·10⁻³, Δ de +0,10 à +0,41. Pour k = 1 et 2, le déficit est déjà là, à époque égale,
avant leur arrêt anticipé. La
fourche 100 est postérieure à la phase chaotique ; les graines, elles, diffèrent dès
le premier pas.

## Question

**B** : à la fourche 5, la sensibilité de la qualité finale à une perturbation d'un pas
float32 passe-t-elle encore surtout par les décisions de learning rate et d'arrêt, ou
par la dynamique des poids ?

## Dispositif

- Identique au pilote A : fourche 5 (pas 42, début de l'époque 6), perturbations
  k = 1…4 de mêmes graines.
- **« Décide »** k = 0…4 : les runs déjà prévus pour P0.
- **« Rejoue »** k = 0…4, ajouté : learning rate de chaque époque et pas d'arrêt du
  témoin « décide » k = 0 de la fourche 5, sans arrêt anticipé. Aucun changement du
  script d'entraînement.
- **Garde active, non rejouée**, comme dans le bras « rejoue » du pilote A. Imposer les
  bascules du témoin inverserait, jusqu'à la fin, un enfant dont la sortie a déjà le
  bon signe (ESR voisin de 4). Or le parent bascule aux époques 7, 8 et 9, juste après
  la fourche, là où une perturbation minime peut changer le signe de la corrélation.
  Le bras « rejoue » retire donc les décisions de learning rate et d'arrêt, pas celles
  de la garde. Les bascules de chaque enfant sont comparées à celles du témoin.
- **Ordre** (`scripts/product_fork_pilot_queue.sh`) : les cinq « décide » ; si P0 tient
  (s ≥ 0,10), « rejoue » k = 0, qui doit reproduire bit à bit « décide » k = 0, sinon
  le bras s'arrête ; puis k = 1…4.
- **Synchronisation**, descriptive (`scripts/product_fork_pilot_analysis.py`) : pour
  chaque enfant « rejoue » et chaque division imposée, log du rapport entre sa perte de
  validation moyenne sur les 10 époques qui précèdent la division et celle du témoin.
  L'enfant est synchronisé si |écart| ≤ 0,06 à chaque division : c'est l'étendue des
  enfants « rejoue » k = 1 et 2 de la fourche 100 (−0,044 à +0,056).

## Prédictions (seuils du pilote A)

- **Validité** : « rejoue » k = 0 identique bit à bit à « décide » k = 0.
- **B soutenue, le calendrier domine aussi tôt** : s(décide) ≥ 0,10,
  s(rejoue) ≤ s(décide) / 2, et moyenne des |Δk| en « décide » au moins double de celle
  en « rejoue ».
- **Les poids comptent tôt** : s(rejoue) ≥ 0,10 et rapport < 2. Si l'enfant « rejoue »
  de plus grand |Δk| est synchronisé et a les bascules du témoin, l'instabilité vient
  de la dynamique des poids. S'il n'est pas synchronisé, le calendrier du témoin est
  désaccordé pour lui et le pilote ne sépare pas dynamique et désaccord. S'il a d'autres
  bascules, la garde est une explication concurrente.
- **Rien ne compte tôt** : P0 échoue, s(décide) < 0,10 ; le bras « rejoue » n'est pas
  lancé.
- **Autres cas** : indécidable.
- **Portée** : 5 runs par bras, 4 degrés de liberté, niveau pilote.
- **Suite** : si B est soutenue, graines sous un calendrier commun fixe, protocole écrit
  avant les runs. Sinon, pas de run de graines sous calendrier commun sans discussion.
