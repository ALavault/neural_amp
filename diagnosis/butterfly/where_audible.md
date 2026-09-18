# Quel matériau porte la divergence entre deux enfants ?

Lecture seule, CPU. Script : `diagnosis/butterfly/where_audible.py`. Sortie : `where_audible.json`.
Motivée par une remarque de l'utilisateur après une première écoute infructueuse : « le choix de la
fuzz à fond n'est peut-être pas le plus intelligent ». Si un trait mesurable du matériau portait la
divergence, il indiquerait quoi enregistrer, et quoi écouter.

## Hypothèse testée

L'écart entre deux enfants d'une même graine se concentre-t-il sur un type de matériau
identifiable (niveau, facteur de crête, centroïde spectral, proportion de passages calmes) ?

**Précaution.** Une localisation « segments calmes » a déjà été affirmée puis **réfutée** pour un
effet voisin — l'écart entre graines — et remplacée par le facteur global (F-0007,
`paper/RESULTS_SSM_SEEDS.md` section 3). La question posée ici n'est pas celle-là : il s'agit de
deux enfants d'une **même** graine, donc du même découpage. La part globale est mesurée avant toute
affirmation de localisation.

## Observations

Par segment de test, écart modèle↔modèle (après égalisation de gain) rapporté à l'erreur du témoin
contre l'appareil :

| segment | 0 | 1 | 6 | 7 | 8 | 2 | 3 | 5 | 9 | 10 | 11 | 4 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| part | 0,93 | 0,43 | 0,65 | 0,38 | 0,28 | 0,26 | 0,24 | 0,21 | 0,12 | 0,17 | 0,10 | 0,05 |

**Négatif : aucun trait du matériau ne prédit cette part** (corrélations de rang, n = 12) —
RMS ρ = −0,12 (p = 0,71), facteur de crête ρ = +0,12 (p = 0,71), centroïde spectral ρ = +0,43
(p = 0,17), proportion de passages calmes ρ = +0,19 (p = 0,55). Le contre-exemple est net : les
segments 0 et 11 ont tous deux 50 % de passages sous −40 dB de leur crête, et leurs parts valent
0,93 et 0,10. La thèse des segments calmes reste réfutée, y compris pour cette question-ci.

**Nuance sur le facteur global.** Écart-type en log du rapport d'ESR segment par segment, qui
vaudrait zéro pour un facteur purement global :

| comparaison | médiane | sd(log) | plage |
|---|---|---|---|
| enfant k1 / témoin (un pas float32) | 1,37 × | 0,225 | 0,99 – 2,34 |
| graine 43 / graine 42 | 3,30 × | 0,629 | 0,83 – 8,82 |
| graine 42 répétée / graine 42 | 1,16 × | 0,286 | 0,85 – 2,42 |

Dans les trois cas il y a une part globale **et** un résidu par segment du même ordre en log. Entre
graines, le facteur global domine nettement (1,19 en log contre 0,63 de résidu), ce qui est
cohérent avec F-0007 ; mais « la graine multiplie l'erreur de chaque segment par un facteur
unique » est une approximation, pas une égalité : le rapport y varie d'un facteur 10 selon le
segment.

## Ce que cela permet de conclure

- La divergence entre deux enfants n'a pas de prédicat mesurable dans le matériau. Choisir un autre
  appareil ou un autre passage sur la foi d'un trait acoustique n'a donc aucune base ici.
- Elle est propre à chaque paire de runs : c'est sur le segment 0 que k1 s'écarte le plus du témoin,
  et rien n'indique qu'un autre enfant s'écarterait au même endroit. Les extraits d'écoute choisis
  sur la divergence valent pour cette paire, pas comme « les segments sensibles » du jeu de test.
- Le masquage, lui, reste une explication mesurée de l'inaudibilité : l'écart se tient 16 à 21 dB
  sous le signal dans les mêmes demi-octaves sur le segment 0, et 27 à 35 dB ailleurs.

## Ce que cela ne permet pas de conclure

- n = 12 segments d'un seul enregistrement, une seule paire témoin/enfant : les corrélations nulles
  ci-dessus n'excluent pas un prédicat plus fin (un transitoire, une tenue de note) que ces quatre
  traits grossiers ne capturent pas.
- Le tableau des rapports entre graines mêle deux causes, puisque la graine tire aussi le découpage
  train/validation. Il qualifie la formulation de F-0007, il ne la remplace pas : l'analyse de
  variance de E-0006 porte sur des données par segment que je n'ai pas reprises ici.

## Prochaine expérience discriminante

Comparer les enveloppes temporelles de l'écart sur le segment 0 : si l'excès de k1 se concentre sur
les attaques ou sur les fins de note plutôt que sur la tenue, le prédicat est temporel et non
spectral, ce que les quatre traits agrégés ci-dessus ne pouvaient pas voir. Coût : aucune inférence
nouvelle, les rendus sont déjà écrits dans `demo/listening/audio/`.
