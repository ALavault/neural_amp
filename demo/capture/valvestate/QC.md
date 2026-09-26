# Séance VS100R du 2026-09-26 : contrôle qualité

Voie produit, jeu d'exploration : il sert à lever des hypothèses, pas à les trancher. Fichiers bruts
dans `datasets/amp_capture/Media/` (non versionnés). Scripts : `scripts/product_valvestate_passes.py`,
`product_valvestate_qc.py`, `product_valvestate_noise.py`. Sorties : `passes.json`, `qc.json`,
`noise.json`. Relu par Codex en lecture seule ; ses corrections sont intégrées.

## Fichiers et appariement

- 47 prises Reaper, chacune avec deux pistes mono 24 bits à 192 kHz, enregistrées dans le même flux :
  `02-Looopbck-<horodatage>.wav`, le bouclage (le signal reçu par l'ampli), et
  `03-FX Out-<horodatage>.wav`, le départ de boucle d'effets.
- 47 passages complets de la source, un par fichier, tous calés à 4,011 s. Associés aux 47 lignes de
  `settings.csv` par ordre chronologique, comme indiqué par l'utilisateur.
- **Volumes notés par l'utilisateur** : OD2 vers 2,5 à 2,7. Clean à 1, baissé parce qu'il
  écrêtait, ce qui explique le niveau bas du canal Clean.

## Chaîne de lecture et d'enregistrement

- **Sauts de 24 échantillons.** 16 prises sur 47 (2, 7, 10, 11, 12, 15, 17, 18, 20, 23, 27, 31,
  35, 38, 45, 47) ont des sauts d'environ 24 échantillons entre la source et le bouclage.
  - Relecture Codex : les sauts se retrouvent aussi par corrélation directe avec le bouclage de la
    prise 1, entre 23 et 26 échantillons, avec une corrélation normalisée d'au moins 0,99998.
  - Profil local de décalage, un point toutes les 2 s : on écarte les estimations sur la borne de
    recherche ou trop faibles, il en reste 4 606 valides sur 4 653. Une fois ramenés à un multiple
    de 24, les écarts restants sont de 3 échantillons au plus pour 99,89 % de ces points.
  - 24 échantillons valent 125 µs, la durée d'une micro-trame USB haute vitesse. **C'est une
    hypothèse sur l'origine**, pas une démonstration.
- **La paire bouclage / sortie FX reste synchrone aux deux bouts.** Le décalage FX contre bouclage,
  mesuré sur les bips, est le même au début et à la fin de chaque prise, à 1 échantillon près,
  dans les 47 prises (`qc.json`, `blip_lag_start` et `blip_lag_end`). Cela n'exclut pas une
  désynchronisation passagère qui se compenserait entre les deux. **Choix retenu pour ce jeu : le
  bouclage de chaque prise sert d'entrée des modèles**, pas le fichier source.
- **Niveau de la chaîne** : gain du bouclage face à la source de −5,25 à −5,28 dB dans les prises
  sans saut. Les écarts vus ailleurs (jusqu'à −6,06 dB) viennent du désalignement.
- **Bruit du bouclage** : −97,6 dBFS avant la lecture comme pendant le silence.

## Défauts par prise

- **Faible signal** (rapport signal sur bruit sous 20 dB, règle de `ANALYSIS_RULES.md`) :
  - prises 6, 7, 10 et 27 (gain 0 : environ 2 dB) ;
  - prises 41 et 42 (canal Clean : 17 et 19 dB).
- **Prise 14** : un échantillon isolé à −0,25 dBFS, 33,6 fois plus fort que le plus fort de ses 16
  voisins, sur la piste FX seulement. Il est à 193,8507 s du début du passage (temps de la
  source), soit l'échantillon 37 989 410 du WAV, dans la section DI de test. Son origine n'est pas
  établie. Il est remplacé par la moyenne de ses deux voisins dans le jeu d'exploration.
- **Prise 42** : 385 échantillons à −0,1 dBFS ou plus (dont 120 exactement au maximum), tous dans le
  balayage à −3 dBFS. Canal Clean, aigus à 10. Les sections DI sont intactes.
- **Prise 3** : la piste FX a 4 096 échantillons de plus que le bouclage, à la fin du fichier. Le
  passage est aligné.
- **Niveaux** : sortie FX du DI vers −21 dBFS RMS sur OD2, vers −53 dBFS sur Clean.

## Bruit de la sortie FX, dans les 5 s de silence (médianes, dBFS)

Toutes les colonnes portent sur 100 Hz à 10 kHz. Les trois composantes, sur des masques disjoints,
reconstituent le bruit de la bande à 0,13 dB près. La cohérence est la part de chaque composante
expliquée linéairement par le bruit du bouclage. Son plancher d'estimation est d'environ 0,04.

| canal | gain | Power Dim. | prises | bruit 100 Hz–10 kHz | large bande | harmoniques secteur | raie 8 kHz | cohérence : large bande | secteur | 8 kHz |
|---|---|---|---|---|---|---|---|---|---|---|
| Clean | — | off | 5 | −76,9 | −83,2 | −78,0 | −103,7 | 0,04 | 0,20 | 0,71 |
| OD2 | 0 | off | 4 | −83,6 | −102,5 | −83,7 | −101,4 | 0,04 | 0,30 | 0,90 |
| OD2 | 2,5 | off | 5 | −79,1 | −86,8 | −80,7 | −88,2 | 0,23 | 0,41 | 0,92 |
| OD2 | 5 | off | 22 | −67,4 | −72,9 | −69,5 | −77,8 | 0,23 | 0,30 | 0,93 |
| OD2 | 5 | on | 2 | −67,7 | −80,2 | −68,2 | −93,3 | 0,21 | 0,21 | 0,91 |
| OD2 | 7,5 | off | 5 | −59,0 | −69,8 | −59,4 | −80,7 | 0,22 | 0,28 | 0,93 |
| OD2 | 7,5 | on | 1 | −61,9 | −73,2 | −62,3 | −90,0 | 0,24 | 0,28 | 0,91 |
| OD2 | 10 | off | 3 | −56,8 | −67,8 | −57,2 | −78,8 | 0,23 | 0,30 | 0,93 |

Lecture, sous forme d'hypothèses :

- **Le ronflement secteur est la composante la plus forte dans les 47 prises**, d'au moins 2 dB. Il
  monte d'environ 27 dB entre les gains 0 et 10.
- **La raie à 8 kHz est commune aux deux pistes** (cohérence de 0,90 à 0,93 sur OD2) et monte avec
  le gain. Une cohérence ne dit pas d'où vient physiquement un bruit : ce peut être la sortie de
  l'interface, reçue et amplifiée par l'ampli, ou un couplage commun.
- **Le bruit large bande monte d'environ 30 dB entre les gains 0 et 5.** Dès le gain 2,5, environ
  un quart (0,22 à 0,24) est expliqué par le bruit du bouclage, au plancher aux gains 0 et en
  Clean. Le reste peut être le bruit propre de l'ampli, un bruit que le bouclage ne voit pas, ou
  une relation non linéaire. **Où il entre dans le circuit n'est pas établi.** Ces groupes
  mélangent aussi plusieurs égalisations : à gain 5, le large bande varie de 8,9 dB d'une prise à
  l'autre.
- **Conséquence possible pour l'entrée des modèles.** Nourri du bouclage, un modèle pourrait
  reproduire la part des bruits qui passe par l'entrée, raie à 8 kHz comprise. Avec la source
  recalée sur le bouclage, elle deviendrait imprévisible.

## Pour le jeu contrôlé (hypothèses à tester)

- lever la masse sur la boîte de réamplification ;
- ordinateur sur batterie, ou autre port USB, pour la raie à 8 kHz et les sauts ;
- recaler la source sur le bouclage et l'utiliser comme entrée ;
- un passage répété à l'intérieur d'une même prise, pour mesurer le bruit sans dérive ;
- volume Clean plus haut, sans écrêter.
