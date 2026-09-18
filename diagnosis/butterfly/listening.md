# Écoute aveugle des enfants du papillon : lecture pré-déclarée

Page : `demo/listening/butterfly.html` (audio dans `demo/listening/audio/`, non versionné).
Générateur : `scripts/product_butterfly_listening.py`.
**Cette lecture est écrite avant toute écoute.** Aucun résultat d'écoute n'existe à ce jour.

## Hypothèse testée

Le pilote A mesure une divergence d'ESR de test entre le témoin (décide k = 0, ESR 0,0341) et
son enfant le plus divergent (décide k = 1, ESR 0,0512) : 41 % d'erreur relative en plus, née
d'une perturbation d'un pas float32 à l'époque 100. Cette divergence est-elle audible, ou
n'existe-t-elle que dans la métrique ?

## Méthode

Trois extraits pris dans un seul segment de test (0,3–4,7 s des 5 s), aux positions fixes 2, 6
et 10 — choisies par position, jamais sur le résultat. La fenêtre reste à l'intérieur d'un
segment parce que le protocole réinitialise l'état du modèle au début de chaque segment : une
fenêtre à cheval transporterait ce transitoire. Chaque côté modèle est corrigé d'un gain des
moindres carrés contre la cible, propre à l'extrait, pour qu'un écart de niveau ne donne pas la
réponse ; les corrections appliquées valent au plus 0,25 dB. L'épreuve porte donc sur ce qui
reste au-delà du niveau. A/B tiré au chargement, X tiré à chaque essai, dix essais par bloc.

Trois paires par extrait, dont deux témoins de l'épreuve elle-même :

| extrait | réel contre témoin | témoin contre décide k1 | témoin contre rejoue k1 |
|---|---|---|---|
| 1 | 2,61e−2 | 9,18e−3 | 2,85e−4 |
| 2 | 1,52e−2 | 1,68e−2 | 2,46e−4 |
| 3 | 1,91e−2 | 3,57e−3 | 6,44e−4 |

(ESR entre les deux côtés joués, après correction de gain.)

**À noter avant d'écouter :** sur l'extrait 2, les deux modèles diffèrent entre eux davantage
(1,68e−2) que le témoin ne diffère de l'appareil réel (1,52e−2). Sur ce matériel, la
perturbation d'un pas float32 déplace la sortie autant que l'erreur de modélisation elle-même.

## Baseline

L'épreuve est son propre étalon : la paire réel contre témoin donne le plancher d'audibilité du
matériel et du dispositif d'écoute ; la paire témoin contre rejoue k1, dont les sorties diffèrent
de 2 à 6e−4, donne le niveau de hasard attendu.

## Lecture pré-déclarée

Trois extraits × dix essais = trente essais par paire ; 20 justes sur 30 donnent p &lt; 0,05
(binomiale unilatérale), 21 sur 30 donnent p &lt; 0,02.

- **Témoin contre rejoue k1 détecté (≥ 20/30)** : l'épreuve fuit — repère visuel, indice de
  niveau, ou erreur de rendu. Les autres blocs ne valent alors rien et il faut corriger le
  dispositif avant d'en tirer quoi que ce soit.
- **Réel contre témoin au hasard** : l'épreuve n'a pas la sensibilité voulue sur ce matériel.
  Rien ne peut être conclu sur les enfants, et il faut changer d'extraits ou de dispositif avant
  de recommencer.
- **Réel contre témoin détecté, témoin contre décide k1 au hasard** : la divergence de 41 % d'ESR
  est inaudible sur ce matériel. L'effet papillon resterait alors un fait métrologique — il
  invaliderait les comparaisons de modèles publiées à un seul run, sans changer le produit.
- **Les deux détectés** : la divergence est audible. Deux runs qui ne diffèrent que par un pas
  float32 ne donnent pas le même appareil virtuel, et le choix du run devient une décision de
  conception, pas un détail de reproductibilité.

## Ce que l'épreuve ne pourra pas conclure

Un seul auditeur, un seul appareil (Big Muff S050_V100), trois extraits d'un même enregistrement,
un seul couple témoin/enfant. Un résultat positif dira que *cette* paire est distinguable, pas que
la divergence est perceptuellement importante, ni qu'un modèle est meilleur que l'autre : A/B/X
mesure la distinguabilité, pas la préférence ni la fidélité perçue. Un résultat négatif ne dira
rien des autres enfants ni des autres appareils.

## Prochaine expérience discriminante

Si les deux paires décisives sont détectées : refaire l'épreuve en préférence appariée contre
l'appareil réel (lequel des deux modèles sonne le plus proche du réel), avec les cinq enfants du
bras décide, pour savoir si l'ESR ordonne la fidélité perçue. Si elle ne l'ordonne pas, c'est la
métrique de sélection qui est en cause, pas seulement sa dispersion.
