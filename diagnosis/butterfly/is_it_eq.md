# L'écart entre deux enfants est-il un écart d'égalisation ?

Lecture seule, CPU. Script : `diagnosis/butterfly/is_it_eq.py`. Sortie : `is_it_eq.json`.
Motivé par le compte rendu d'écoute de la troisième version de la page : « aucune différence, sauf
peut-être un peu plus de grave parfois, un demi-décibel au maximum ».

## Hypothèse testée

Si ce qui sépare les deux modèles est une réponse en fréquence statique, la divergence d'ESR est
perceptivement bon marché et un seul filtre l'efface. Sinon, les deux modèles se comportent
différemment, et aucun réglage de tonalité ne les rapproche.

## Méthode

Sur les douze segments de test concaténés, spectres de Welch (8 192 points, recouvrement de moitié) :

- **|H(f)|**, module du meilleur filtre linéaire de la sortie de l'enfant vers celle du témoin,
  exprimé en dB par tiers d'octave et pondéré par le spectre du témoin — l'écart d'égalisation que
  l'oreille pourrait relever ;
- la **cohérence quadratique**, qui dit quelle part de l'écart un filtre statique, de forme
  quelconque, peut retirer. Le résidu 1 − cohérence est ce qu'aucune égalisation n'atteint.

Aucun filtrage n'est appliqué : l'énergie résiduelle du meilleur filtre linéaire vaut
S_yy·(1 − cohérence), donc le rapport se déduit des spectres. Un gain des moindres carrés est retiré
d'abord, pour que le résidu ne soit pas gonflé par un simple écart de niveau.

## Observations

| paire | ESR brut | après le meilleur filtre | part irréductible | pente grave→aigu |
|---|---|---|---|---|
| témoin / décide k1 | 0,0072 | 0,0071 | **99 %** | +0,03 dB |
| témoin / rejoue k1 | 0,0005 | 0,0005 | 98 % | +0,01 dB |

Réponse en fréquence, par tiers d'octave de 99 Hz à 3,2 kHz : entre −0,04 et +0,02 dB. Il n'y a
**pas** d'écart d'égalisation — ni dans le grave ni ailleurs. Le demi-décibel rapporté à l'écoute
vaudrait quinze fois l'écart mesuré le plus grand.

## Ce que cela permet de conclure

- L'écart entre les deux modèles n'est ni un écart de niveau, ni un écart d'égalisation, ni un
  décalage temporel constant — un retard fixe est un filtre linéaire, donc il serait retiré par la
  cohérence. Il est **dépendant du programme** : le modèle réagit différemment selon ce qu'on lui
  donne, ce qui rejoint la localisation dans les fins de note (`when_audible.md`).
- Cela tranche la réserve laissée ouverte dans `when_audible.md` : « la porte se referme 5 ms plus
  tard » est exclu si le décalage est constant. Un décalage variable selon la note ne l'est pas.
- Pour la métrique : 99 % de la divergence d'ESR entre deux runs porte sur du comportement, pas sur
  la réponse en fréquence. Deux runs séparés d'un pas float32 ne se rattrapent pas avec un filtre.

## Ce que cela ne permet pas de conclure

- La cohérence est estimée globalement, donc elle ne verrait pas un filtre qui changerait au cours du
  temps. Un « filtre lentement variable » resterait compté comme irréductible.
- Ce que l'auditeur a entendu n'est pas expliqué. Deux lectures tiennent : un effet dépendant du
  programme qui se lit comme « plus de grave parfois », ou une attente. La mesure exclut seulement
  qu'il s'agisse d'un écart statique.
- Une paire témoin/enfant, un appareil, un enregistrement.

## Prochaine expérience discriminante

Si l'on veut savoir laquelle des deux lectures est la bonne, l'épreuve existe déjà : c'est le bloc
« appareil réel contre témoin » de la page. S'il est au hasard lui aussi, l'épreuve n'a pas la
sensibilité voulue et rien ne peut être conclu sur les enfants ; s'il est franchi nettement, alors
l'inaudibilité de l'écart entre modèles est un résultat, pas un échec de dispositif.
