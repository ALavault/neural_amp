# Où, dans le temps, se loge l'écart entre deux enfants ?

Lecture seule, CPU. Script : `diagnosis/butterfly/when_audible.py`. Sortie : `when_audible.json`.
Suite de `where_audible.md`, qui n'avait trouvé aucun prédicat dans quatre traits agrégés du segment
et avait mesuré un écart 16 à 21 dB sous le signal dans les mêmes demi-octaves. Un agrégat ne peut
pas voir un instant bref : ceci regarde image par image.

## Hypothèse testée

L'écart entre le témoin et son enfant divergent se concentre-t-il sur un moment du son — attaque,
tenue, fin de note ? Si oui, une fenêtre courte prise au bon endroit serait bien plus audible que
l'extrait de 4,4 s qui moyenne tout.

## Méthode

Images de 20 ms. Dans chacune, rapport en dB entre l'enveloppe de l'écart (témoin moins enfant, après
égalisation de gain sur le segment) et l'enveloppe du signal. Images classées par la pente de
l'enveloppe du signal : **attaque** au-dessus de +3 dB par image, **chute** en dessous de −3 dB,
**tenue** entre les deux. Les images sous −40 dB de la crête du segment sont écartées, comme non
pertinentes perceptivement. Enfin, la meilleure seconde de chaque segment, c'est-à-dire la fenêtre
glissante de 50 images dont le rapport moyen est le plus haut.

## Observations

| segment | médiane | attaque | tenue | chute | meilleure seconde | à t = |
|---|---|---|---|---|---|---|
| 0 | −15,9 | −19,0 | −16,0 | **−5,3** | −13,4 | 4,00 s |
| 1 | −24,3 | −21,4 | −25,3 | **−2,3** | −14,6 | 0,32 s |
| 2 | −28,3 | −25,5 | −28,7 | −18,4 | −18,6 | 3,72 s |
| 3 | −19,9 | −18,7 | −20,0 | −17,9 | −16,4 | 1,38 s |
| 6 | −22,1 | −20,3 | −22,4 | −20,5 | −17,2 | 2,92 s |
| 11 | −39,7 | — | −39,7 | −29,5 | −35,9 | 1,52 s |

**L'écart vit dans les fins de note.** Sur onze segments sur douze, la médiane des chutes est
au-dessus de celle des tenues, et l'écart est spectaculaire là où la divergence est forte : −5,3 dB
contre −16,0 sur le segment 0, −2,3 dB contre −25,3 sur le segment 1. Pendant une chute du
segment 1, les deux modèles diffèrent d'une quantité qui n'est que 2,3 dB sous le signal lui-même.

**Les attaques ne portent rien de particulier** : leur médiane est proche de celle des tenues, et
parfois en dessous (segments 7, 10).

**Une seconde bien choisie gagne jusqu'à +9,7 dB** sur la médiane de son segment (segment 1 :
−24,3 de médiane, −14,6 sur la meilleure seconde).

## Ce que cela permet de conclure

- Le mécanisme est plausible et cohérent avec l'appareil : sur une fuzz, la fin de note est le moment
  où le circuit se referme et « crachote ». Deux modèles qui ne diffèrent que d'un pas float32
  referment cette porte à des instants légèrement différents, ce qui produit un écart relatif énorme
  là où le signal s'effondre, et négligeable pendant la tenue où la saturation écrête tout de la
  même façon.
- L'échec de la première écoute s'explique entièrement : un extrait de 4,4 s moyenne des chutes
  brèves à −2 dB avec des tenues à −25 dB, et c'est la tenue qui occupe le temps. Le masquage
  mesuré dans `where_audible.md` reste vrai pour la tenue, et faux pour les chutes.
- Il existe donc une chance réelle que la divergence soit audible, à condition d'écouter une fenêtre
  courte centrée sur une fin de note.

## Ce que cela ne permet pas de conclure

- Le rapport mesuré pendant une chute est un rapport d'enveloppes sur 20 ms : un décalage temporel
  infime entre les deux modèles y produit mécaniquement un grand écart relatif. Rien ici ne
  distingue « la porte se referme autrement » de « la porte se referme 5 ms plus tard », et les deux
  ne s'entendent pas pareil.
- Une paire témoin/enfant, un appareil, un enregistrement. La classification en trois catégories par
  la pente de l'enveloppe est grossière et ne connaît pas les notes.
- Un rapport favorable n'est pas une audibilité : il faut l'épreuve en aveugle.

## Conséquence appliquée

La page d'écoute est reconstruite sur des fenêtres de 1,5 s centrées sur la meilleure seconde des
trois segments les plus divergents, au lieu de 4,4 s pris au milieu. C'est la troisième version, et
son statut est le même que la deuxième : une épreuve au meilleur endroit possible, en segment comme
en instant, et non un échantillon aveugle.
