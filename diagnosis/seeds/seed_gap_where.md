# Où vit l'écart entre la meilleure et la pire graine

**Hypothèse.** Le pilote C sépare la graine 42 (ESR de test 0,036) de la graine 49 (0,178), un
facteur 5. Les enfants du papillon n'étaient séparés que d'un facteur 1,4, et l'écoute n'y a rien
distingué. Si un écart d'ESR devient audible quelque part, c'est ici.

**Méthode.** Les deux modèles sont rendus sur les douze segments de test, chacun corrigé par un
gain aux moindres carrés contre la cible — sinon une différence de niveau se ferait passer pour une
différence de timbre. Par segment : l'écart modèle-à-modèle rapporté à l'erreur modèle-appareil de
la *bonne* graine, et la meilleure fenêtre de 1,5 s par rapport écart/signal en trames de 20 ms. La
fenêtre reste à 0,3 s des bords, le protocole remettant l'état du modèle à zéro à chaque segment.

**Référence.** Les mêmes quantités mesurées sur les enfants du papillon (`where_audible.md`,
`when_audible.md`) : écart/erreur plafonnant à 0,93, écart vivant 16 à 25 dB sous le signal en
sustain et −2,3 à −5,3 dB dans les queues de notes, et rien d'audible.

## Observations

| | enfants papillon | graine 42 contre 49 |
|---|---|---|
| écart modèle-à-modèle / erreur modèle-appareil | 0,93 au mieux | **2,81** (segments 0, 6, 2 : 2,81 / 2,77 / 2,10) |
| écart sous le signal, meilleure fenêtre | −2,3 à −5,3 dB | **−3,7 à −5,7 dB** |
| ESR de la pire, sur le segment le plus divergent | — | **0,434** contre 0,048 |

Trois choses à relever.

- **L'écart entre les deux graines dépasse l'erreur du bon modèle d'un facteur près de 3.** Les deux
  modèles diffèrent donc bien plus l'un de l'autre que le bon ne diffère de l'appareil. Chez les
  enfants, ce rapport restait sous 1.
- **La pire graine se dégrade très inégalement.** Son ESR par segment va de 0,004 à 0,434, contre
  0,004 à 0,082 pour la bonne. Le segment 11 les rend identiques (0,0041 toutes deux) ; le segment 0
  les sépare d'un facteur 9.
- **Un écart de niveau apparaît.** Le gain de recalage de la graine 49 atteint −2,4 dB sur l'extrait
  le plus divergent, contre −0,3 dB pour la graine 42 : la pire graine sort plus fort. Il est corrigé
  avant écoute, donc il ne peut pas trahir la réponse, mais c'est un symptôme à part entière.

## Ce qu'on peut conclure

Que le dispositif d'écoute a, cette fois, une chance de mesurer quelque chose : l'écart siège 4 à
6 dB sous le signal au lieu de 16 à 25, donc le masquage qui expliquait probablement l'échec
précédent ne s'applique plus.

## Ce qu'on ne peut pas conclure

Rien sur l'audibilité : ceci est une mesure de signal, pas une écoute. Et le choix des extraits est
fait **sur** la divergence, donc c'est un meilleur cas assumé — si l'écart est inaudible ici, il
l'est partout sur cet appareil ; s'il est audible ici, cela ne dit pas qu'il le serait sur un
extrait tiré au hasard.

## Prochaine expérience discriminante

L'écoute elle-même, `demo/listening/seeds.html`, générée par `scripts/product_seed_listening.py`.
Son échelle **descend** au lieu de monter, parce que l'écart est attendu audible : chaque barreau
oppose la graine 42 à l'interpolation (1 − f) × graine 42 + f × graine 49, avec f de 1 à 0,18, et
l'ESR de chaque barreau est affichée. Le barreau où l'auditeur retombe au hasard se lit directement
comme « audible jusqu'à cette ESR » — ce qui donnerait, pour la première fois sur ce dispositif, un
seuil d'audibilité en unités de la métrique du banc.
