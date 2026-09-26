# Séance VS100R du 2026-09-26 : contrôle qualité

Voie produit, jeu d'exploration : il sert à lever des hypothèses, pas à les trancher. Fichiers bruts
dans `datasets/amp_capture/Media/` (non versionnés). Scripts : `scripts/product_valvestate_passes.py`,
`product_valvestate_qc.py`, `product_valvestate_noise.py`. Sorties : `passes.json`, `qc.json`,
`noise.json`.

## Fichiers et appariement

- 47 prises Reaper, chacune avec deux pistes mono 24 bits à 192 kHz, enregistrées dans le même flux :
  `02-Looopbck-<horodatage>.wav`, le bouclage (le signal reçu par l'ampli), et
  `03-FX Out-<horodatage>.wav`, le départ de boucle d'effets.
- 47 passages complets de la source, un par fichier, tous calés à 4,011 s. Associés aux 47 lignes de
  `settings.csv` par ordre chronologique, comme indiqué par l'utilisateur.

## Chaîne de lecture et d'enregistrement

- **Sauts de micro-trame USB.** 17 prises sur 47 ont des sauts d'exactement 24 échantillons
  (125 µs à 192 kHz, une micro-trame USB haute vitesse) entre la source et le bouclage. Une fois
  ramené à un multiple de 24, l'écart restant est de 3 échantillons au plus sur 99,9 % des points.
  Certains sauts se compensent.
- **La paire bouclage / sortie FX reste synchrone.** Dans chaque prise, le décalage entre les deux
  pistes, mesuré sur les bips, est le même au début et à la fin, à 1 échantillon près. **Choix
  retenu pour ce jeu : le bouclage de chaque prise sert d'entrée des modèles**, pas le fichier
  source.
- **Niveau de la chaîne** : gain du bouclage face à la source de −5,25 à −5,28 dB dans les prises
  sans saut. Les écarts vus ailleurs (jusqu'à −6,06 dB) viennent du désalignement.
- **Bruit du bouclage** : −97,6 dBFS avant la lecture comme pendant le silence.

## Défauts par prise

- **Faible signal** (rapport signal sur bruit sous 20 dB, règle de `ANALYSIS_RULES.md`) :
  - prises 6, 7, 10 et 27 (gain 0 : environ 2 dB) ;
  - prises 41 et 42 (canal Clean : 17 et 19 dB).
- **Prise 14** : un échantillon isolé à −0,3 dBFS à 193,85 s (section DI de test), sur la piste FX
  seulement. C'est un pic numérique, à réparer ou à exclure.
- **Prise 42** : 385 échantillons écrêtés, tous dans le balayage à −3 dBFS (canal Clean, aigus à
  10). Les sections DI sont intactes.
- **Prise 3** : la piste FX a 4 096 échantillons de plus que le bouclage, à la fin du fichier. Le
  passage est aligné.
- **Niveaux** : sortie FX du DI vers −21 dBFS RMS sur OD2, vers −54 dBFS sur Clean, soit environ
  33 dB plus bas. Le volume Clean est à vérifier dans les valeurs fixes de la séance.

## Bruit de la sortie FX, dans les 5 s de silence (médianes, dBFS)

| canal | gain | Power Dim. | prises | bruit total | large bande | harmoniques secteur | raie 8 kHz | part cohérente avec le bouclage |
|---|---|---|---|---|---|---|---|---|
| Clean | — | off | 5 | −73,3 | −83,2 | −78,0 | −103,7 | 0,18 |
| OD2 | 0 | off | 4 | −80,7 | −102,5 | −83,6 | −101,4 | 0,31 |
| OD2 | 2,5 | off | 5 | −77,2 | −86,8 | −80,0 | −88,2 | 0,45 |
| OD2 | 5 | off | 22 | −67,1 | −72,9 | −68,8 | −77,8 | 0,33 |
| OD2 | 5 | on | 2 | −67,5 | −80,2 | −68,2 | −93,3 | 0,22 |
| OD2 | 7,5 | off | 5 | −58,8 | −69,8 | −59,4 | −80,7 | 0,28 |
| OD2 | 7,5 | on | 1 | −61,8 | −73,2 | −62,2 | −90,0 | 0,28 |
| OD2 | 10 | off | 3 | −56,6 | −67,8 | −57,1 | −78,8 | 0,30 |

Lecture, sous forme d'hypothèses :

- **Le ronflement domine partout.** Il s'agit surtout des harmoniques 150 et 250 Hz, et il monte de
  26 dB entre les gains 0 et 10. Une part est cohérente avec le bouclage : ce ronflement passe en
  partie par la liaison interface → boîte → ampli, ce qui fait penser à une boucle de masse.
- **La raie à 8 kHz sort de la Zen Go.** Elle est présente dans le bouclage (cohérence 0,95) et
  l'ampli l'amplifie.
- **Le bruit large bande est surtout propre à l'ampli.** Il monte de 30 dB entre les gains 0 et 5,
  puis se stabilise vers −70 dBFS. Il entrerait donc avant l'étage de gain (piste A du
  brainstorming bruit).
- **Conséquence pour l'entrée des modèles.** Un modèle nourri du bouclage apprendra à reproduire la
  raie à 8 kHz et la part de ronflement venue de la chaîne. Avec la source recalée sur le bouclage,
  ces bruits deviendraient imprévisibles.

## Pour le jeu contrôlé

- lever la masse sur la boîte de réamplification ;
- ordinateur sur batterie, ou autre port USB, pour la raie à 8 kHz ;
- réduire la taille de tampon, ou changer de port USB, contre les sauts de micro-trame ;
- recaler la source sur le bouclage et l'utiliser comme entrée ;
- un passage répété à l'intérieur d'une même prise, pour mesurer le bruit sans dérive ;
- volume Clean plus haut.
