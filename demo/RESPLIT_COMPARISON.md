# Comparaison sur les splits propres (même prise, Big Muff)

## Pourquoi ce banc

Le diagnostic de données a montré que les trois fichiers originaux ne venaient
pas de la même prise : le rapport val/test de NAM A2 Full était de 4,4×, et un
WaveNet entraîné SUR la paire de test n'y descendait qu'à 0,060. Les splits
originaux ne permettent pas de comparer les architectures.

Ce banc redécoupe la prise d'entraînement de 120 s en trois segments contigus
(80/20/20 s), de propriétés acoustiques quasi identiques (RMS 0,084–0,089).
Le rapport val/test de NAM A2 tombe à 1,18×.

## Résultats

| Modèle | Paramètres | ESR train | ESR val. | ESR test | Rapport test/val |
| --- | ---: | ---: | ---: | ---: | ---: |
| S4-TFiLM large | 70 193 | 0,01108 | 0,03047 | **0,03696** | 1,21 |
| NAM A2 Full | 12 145 | 0,04656 | 0,08883 | **0,10513** | 1,18 |
| NAM A2 Lite | 1 870 | 0,07831 | 0,10568 | 0,13102 | 1,24 |

## Ce que le banc dit

**S4-TFiLM bat NAM A2 Full par un facteur 2,8 en ESR de test**, sur des splits
cohérents où les deux modèles généralisent normalement (rapport test/val
autour de 1,2).

Sur les anciens splits (prises séparées), l'écart n'était que de 1 % — les
deux architectures butaient au même endroit, mais cet endroit était un
artefact de données, pas un plafond de modèle.

## Ce que le banc ne dit pas

- Ce banc est interne : personne d'autre n'a ces splits. Il ne fonde pas une
  revendication SOTA publiable, seulement une décision d'ingénierie.
- S4-TFiLM n'est pas exportable en `.nam` : aucun parseur dans le moteur NAM.
- Le coût CPU n'est pas mesuré ici (il l'est dans `SOTA_COMPARISON.md` : S4
  est 1,5× plus cher que A2 en PyTorch pur sur CPU).
- La prise est toujours CC-BY-NC (ToneTwist). Un benchmark publiable demande
  des captures propres.
