# Protocole apparié — le résidu entre architectures vaut-il l'appariement ?

**Pré-enregistré le 2026-09-22, avant le premier run du dispositif.** Les runs `gate0_*` qui
précèdent sont des fumées de 200 pas destinées à vérifier que le mode déterministe fonctionne ; elles
n'entrent dans aucune analyse.

## Le problème, et pourquoi il ne se règle pas en ajoutant des graines

Sur huit graines en mode déterministe, l'écart-type du log de l'ESR de test vaut **0,535**
(`diagnosis/butterfly/pilot_C_common_schedule.md`). L'effet minimal détectable qui en découle, à 5 %
bilatéral et 80 % de puissance, est un **facteur 3,40 en ESR à trois graines** — quand la littérature
du domaine revendique couramment 10 à 20 %. Il faudrait **170 runs par bras** pour détecter 15 %.

S'y ajoute un défaut qui n'est pas une question de puissance. `pl.seed_everything(graine)` précède la
construction du processeur, qui consomme de l'aléa en quantité dépendante de l'architecture, **avant**
le tirage de la partition train/validation. Vérifié : à la graine 42, `ssm-wavenet` et `s4-tf-l-16`
partagent **un segment de validation sur douze** ; entre `ssm-wavenet` à 8 et à 4 blocs,
l'intersection est **vide**. Toute comparaison d'architectures et toute ablation qui change la taille
du modèle sont donc confondues avec un changement de données, sans que rien ne le signale.

L'option `--split-seed S` (`scripts/product_nablafx_bench.py:517`) réamorce juste avant `fit` et
corrige cela : vérifié, trois configurations d'architectures et de graines différentes reçoivent
alors une partition identique.

## La question posée

Fixer la partition permet une **comparaison appariée**, où la difficulté de la partition est commune
aux bras et s'annule. La question est de savoir **combien elle s'annule réellement**.

Le seul dispositif apparié existant est le pilote C — 8 partitions × 2 traitements, même
architecture. Il donne un résidu de **0,301** et une corrélation de rang entre bras de **+0,548** :
les partitions 48 et 49, les pires sous un bras, remontent 6ᵉ et 4ᵉ sous l'autre. La difficulté
d'une partition n'est donc pas une propriété de la partition.

Mais ses deux bras diffèrent par le **retrait de l'ordonnanceur**, c'est-à-dire du canal même par
lequel la partition agit (ρ passant de +0,81 à +0,26 entre les bras). C'est un cas d'interaction
proche du maximum. **Deux architectures gardant toutes deux leur ordonnanceur pourraient interagir
moins — et c'est ce que ce pilote mesure.**

## Dispositif

- **Deux architectures** : `ssm-wavenet` (11 329 paramètres) et `s4-tf-l-16` (70 193). Elles
  reçoivent les mêmes hyperparamètres publiés, (lr 0,01 ; poids 1,0 et 0,1). `s4-l-16` est **exclu** :
  il reçoit des poids de (10,0 ; 1,0) sous le même écrêtage par valeur à 1,0, donc il s'entraînerait
  en permanence sous écrêtage et l'on comparerait une échelle de perte.
- **Trois partitions déclarées** : `--split-seed` 925, 736 et 688, les trois premières d'un tirage
  `random.Random(20260922).sample(range(1000), 5)` = [925, 736, 688, 744, 369], inscrit ici avant
  tout run. Chevauchement maximal entre deux d'entre elles : 2 segments sur 12.
- **Une run par cellule, aucune réplique.** À budget fixé, Var(contraste) =
  2(σ²_interaction + σ²_erreur/R)/S : répliquer ne touche jamais l'interaction, qui domine. À 15 runs,
  S=5/R=1 donne 1,71 quand S=2/R=2 donne 2,15. De plus, en mode déterministe une réplique à graine
  identique est bit à bit identique (`diagnosis/butterfly/determinism.json`).
- **Mode déterministe** de bout en bout, `--deterministic`, vérifié fonctionnel sur les trois
  architectures par les fumées `gate0_*` du 2026-09-22.
- `--seed 42` partout : l'initialisation n'est pas le facteur étudié, et le pilote E a montré qu'à
  chaîne de données fixée elle ne contribue plus rien de mesurable.

Six runs, environ 12 heures.

## Prédiction, et ce qui la réfute

Soit **r = sd(différence en log entre les deux architectures, sur les trois partitions) ÷ √2**, le
résidu apparié réel entre architectures — la quantité que tout ce plan suppose et que personne n'a
mesurée.

- **Appariement soutenu** : r ≤ 0,25. Il tient nettement mieux qu'au pilote C, et le dispositif
  complet est dimensionné sur cette valeur.
- **Appariement réfuté** : r > 0,35. Il ne tient pas mieux qu'au pilote C (0,301), l'interaction
  domine, et un dispositif apparié à cinq partitions ne vaut pas mieux qu'un dispositif non apparié
  à budget égal. Le budget se réoriente.
- **Intermédiaire** : 0,25 < r ≤ 0,35, rapporté tel quel, et l'étape suivante dimensionnée sur la
  valeur observée plutôt que sur une supposition.

Ces six runs sont **inclus dans l'analyse finale**, et cela est écrit d'avance : un pilote qu'on jette
est un pilote qui ne coûte rien à contredire.

## Ce que ce pilote ne pourra pas dire

- **Trois partitions, deux architectures.** r est estimé sur trois différences : son intervalle de
  confiance est large et il sera rapporté. Ce pilote oriente un budget, il ne tranche pas une
  question scientifique.
- **Il compare des recettes publiées, pas des architectures.** Budget de paramètres (facteur 6,2),
  pas d'apprentissage, arrêt anticipé et garde de polarité restent ceux du protocole NablAFx pour
  chaque architecture. L'arrêt anticipé décide dans tous les runs — aucun n'atteint le plafond de
  15 000 pas — donc le budget d'entraînement est une conséquence de l'architecture, pas un contrôle.
- **La partition et l'ordre des lots restent inséparés** : `--split-seed` les fixe ensemble.
- **L'ordre des lots n'est garanti commun qu'à la première époque** : `shuffle=True` retire une
  permutation à chaque époque. Contrôle à faire avant d'élargir le dispositif, en journalisant les
  indices du sampler aux époques 1, 2 et 50.
- **Le biais de valeur absolue subsiste** : fixer la partition fige sa difficulté, le pilote E a
  mesuré 0,0831 contre 0,068 en moyenne sur les partitions tirées. Les ESR absolues de ce dispositif
  ne sont pas comparables à celles de la littérature ; seules les différences entre bras le sont.

## Relecture adverse (2026-09-22, avant tout run)

La première version de ce dispositif annonçait un effet détectable de 1,38 et prévoyait 18 runs avec
deux réplicats. Une relecture adverse l'a attaquée et la vérification lui a donné raison sur tous les
chiffres : le 1,38 supposait l'interaction nulle, le pilote C la mesure à 0,301, et répliquer est
strictement perdant à budget fixé. Le dispositif ci-dessus est la version corrigée — six runs au lieu
de dix-huit, aucune réplique, et un seuil portant sur la quantité mesurée plutôt que sur une
supposition. Les trois autres corrections retenues : `s4-l-16` exclu pour son échelle de perte, le
régime déterministe déclaré et tenu, et le contrôle de l'ordre des lots inscrit comme réserve.

## Incident d'exécution (2026-09-22, 11:05) — première file annulée, ma faute

La file lancée à 10:40 omettait `--discretization zoh`. Le drapeau **défaut à `free`**, et toutes
les runs saines du dépôt — pilote C, pilote E, le banc — utilisent `zoh`. Elle entraînait donc un
autre modèle.

La première run l'a montré sans ambiguïté, et c'est le contrôle de routine qui l'a attrapée avant
que les cinq autres ne s'enchaînent :

| | les 14 runs antérieures | la run annulée |
|---|---|---|
| rapport ESR `last` / `best` | 0,94 à 1,00 | **3,53** (1,14 contre 0,32) |
| époque d'arrêt | 408 à 1103 | **138** |
| bascules de la garde de polarité | 0 à 4 | **8**, dont 105, 106, 107 et 108 |

J'avais d'abord soupçonné la partition, puis le mode déterministe, puis la garde de polarité. C'était
plus simple : un drapeau manquant. La leçon, elle, ne l'est pas — **le protocole teste `last.ckpt`,
et cette run y valait 3,5 fois pire que le meilleur point qu'elle avait trouvé.** Sur les quatorze
runs antérieures ce choix était sans conséquence ; il vient de montrer qu'il ne l'est pas toujours.

Mesures prises, sans rien détruire : l'enregistrement est conservé sous
`demo/nablafx_bench/void_discretization_free_ssm_wavenet_s925.json` et son répertoire de run sous
`demo/runs/nablafx_void_discretization_free_*`, hors du périmètre du dispositif. La deuxième run a
reçu SIGTERM avant d'écrire quoi que ce soit. La file porte désormais le drapeau et un commentaire
disant pourquoi il n'est pas optionnel.

**Rien du dispositif pré-enregistré n'est modifié** : ni les architectures, ni les partitions
déclarées, ni les seuils sur r. Seule la ligne de commande est corrigée pour être celle qui était
décrite.
