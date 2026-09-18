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

## Confusion découverte en cours d'exécution (2026-09-18, après `replay k1`)

`replay k1` donne ESR 0,0894, soit Δ = +0,75 en log contre le témoin — **plus divergent que son
homologue « décide »** (+0,482), alors que le calendrier d'apprentissage et le pas d'arrêt lui sont
imposés. À la fourche 100, les quatre enfants « rejoue » tenaient dans −0,023 à −0,007.

La cause probable n'est pas le calendrier, c'est la garde de polarité, et elle est de ma
responsabilité. Le parent bascule aux époques **7, 8 et 9** (`demo/butterfly/butterfly_ssm_seed42_parent.json`).
Donc :

- **fourche 100** : les bascules sont antérieures à la fourche, les cinq enfants héritent tous de
  `[7, 8, 9]` et la garde ne se déclenche plus jamais. Le bras « rejoue » isole bien le calendrier ;
- **fourche 5** : la fourche précède les bascules. Chaque enfant redécouvre les siennes, comptées
  depuis la fourche — témoin `[2, 67]`, k1 `[2, 4]`, k2 `[3, 4]`, k3 et k4 aucune.

Or la garde est elle-même une décision pilotée par la validation, et la plus brutale des trois :
elle nie la dernière couche et les moments d'Adam. L'addendum la laisse active et ne la rejoue pas —
choix écrit d'avance, au motif qu'imposer les bascules du témoin inverserait un enfant déjà bien
orienté. Ce choix se paie ici : **le bras « rejoue » de la fourche 5 n'isole pas le calendrier**, il
laisse libre une décision de validation, et une grande dispersion n'y départage plus rien.

Ce que cela ne dit pas : que la garde explique l'écart. L'historique des bascules ne s'aligne pas
simplement sur l'ESR — k2 bascule deux fois tôt (`[3, 4]`) et reste à 0,0438, contre le témoin à
0,0422, tandis que k1 bascule deux fois tôt aussi (`[2, 4]`) et part à 0,0684. Les enfants qui ne
basculent jamais (k3, k4) restent près du témoin. C'est une confusion à contrôler, pas une cause
démontrée.

Ce que cela ne remet pas en cause : le verdict du pilote A, qui porte sur la fourche 100, où la
garde est éteinte et partagée.

Observation qui départagerait : une fourche placée **après la dernière bascule du parent** (époque 9)
et avant que le plateau ne se stabilise — par exemple à l'époque 25 — donnerait un bras « rejoue »
où la garde est éteinte comme à la fourche 100, tout en restant dans la phase précoce. C'est la seule
façon de poser à la fourche précoce la question que la fourche 100 a tranchée.

## Verdict du bras (2026-09-18, 22 h 53)

`scripts/product_fork_pilot_analysis.py`, sortie dans `pilot_A_results.json`.

| bras | ESR (k = 0 à 4) | Δ en log | s | pas |
|---|---|---|---|---|
| décide | 0,0422 0,0684 0,0438 0,0443 0,0422 | +0,000 +0,482 +0,036 +0,049 −0,001 | **0,207** | 7 266 à 9 471, propres |
| rejoue | 0,0422 0,0894 0,0371 0,0416 0,0386 | +0,000 +0,750 −0,129 −0,015 −0,090 | **0,365** | 7 266 pour tous |

Porte d'identité franchie : `rejoue k0` reproduit `décide k0` au bit près.

**Le critère pré-enregistré échoue, et dans le sens qui compte.** s(rejoue) = 0,365 n'est pas
inférieur à s(décide)/2 = 0,104 ; le rapport des moyennes des |Δ| vaut 0,58 pour un seuil de 2.
Imposer le calendrier du témoin **ne contient pas** la divergence à la fourche 5 : elle l'augmente.

## Correction de la mise en garde écrite plus haut

J'avais écrit, après `rejoue k1`, que le bras « n'isole pas le calendrier » et qu'« une grande
dispersion n'y départage plus rien ». La première partie reste vraie, la seconde était trop forte.

Les historiques de bascules sont **identiques entre les deux bras, enfant par enfant** :
`[2, 67]`, `[2, 4]`, `[3, 4]`, `[]`, `[]`. La raison est mécanique : les bascules tombent aux époques
2 à 4 après la fourche, très avant la première division du pas d'apprentissage (époque 156 au plus
tôt), donc les deux bras sont encore indiscernables quand la garde se déclenche. La confusion est
donc **symétrique** : à k égal, la seule différence entre les deux bras est bien le calendrier et le
pas d'arrêt.

Ce qui reste vrai de la mise en garde : s(rejoue) ne se lit pas comme « ce qui subsiste quand on
retire les décisions », puisque la garde varie d'un k à l'autre à l'intérieur de chaque bras et
contribue aux deux dispersions.

## Ce que le bras permet de conclure

- À la fourche 5, imposer le pas d'apprentissage de chaque époque et le pas d'arrêt ne réduit pas la
  dispersion d'une perturbation d'un pas float32. Le mécanisme établi à la fourche 100 ne s'étend
  donc pas à la phase précoce.
- Le mécanisme est visible enfant par enfant : les enfants forcés à courir jusqu'au pas du témoin
  s'améliorent (k = 2 passe de 0,0438 à 0,0371, k = 4 de 0,0422 à 0,0386) ou empirent (k = 1 passe
  de 0,0684 à 0,0894) selon qu'ils s'arrêtaient trop tôt ou trop tard de leur propre chef. Le
  calendrier commun déplace chaque enfant, il ne les rassemble pas.
- Image en deux régimes : tard, la divergence passe par les décisions ; tôt, elle existe sans elles.

## Ce qu'il ne permet pas de conclure

- La garde reste libre, donc on ne peut pas dire « la phase précoce diverge sans aucune décision » ;
  seulement « le calendrier et l'arrêt ne la contiennent pas ». Le pilote D, à la fourche 25, pose
  la question avec la garde éteinte.
- Cinq runs par bras, une graine, une fourche, une architecture. Et k = 1 porte à lui seul
  l'essentiel des deux dispersions : sans lui, les |Δ| valent 0,036 / 0,049 / 0,001 pour « décide »
  et 0,129 / 0,015 / 0,090 pour « rejoue ».
