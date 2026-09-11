# Piste prospective — transfert du prior `drum_ai` vers une banque externe

## Statut

Rapport issu d'un side thread, destiné au steering scientifique. Il ne constitue
ni une décision de campagne, ni une modification du protocole d'amplificateur
neuronal actif, ni un résultat expérimental.

## Clarification du concept

Dans cette discussion, **Apeira désigne `drum_ai`**, et non une banque sonore
utilisée comme simple référence stylistique.

L'idée est d'exploiter les représentations ou le prior génératif appris par
`drum_ai` pour transformer une banque DrumGizmo ou personnelle, tout en
conservant sa structure musicale : instruments, articulations, vélocités et
variations. Cette formulation relève davantage de l'adaptation de domaine ou de
la resynthèse guidée que du transfert de style classique.

Le projet Apeira vise la synthèse contrôlable de sons naturels de batterie de
haute qualité, ce qui rend cette direction conceptuellement cohérente :
[présentation du projet](https://apeira-technologies.fr/drum-synthesis/).

Les capacités techniques exactes de `drum_ai` — accès aux poids, espace latent,
conditionnement, fréquence d'échantillonnage, sorties multicanales et droits de
redistribution — n'ont pas été auditées dans ce side thread. Elles conditionnent
le choix de méthode.

## Deux objectifs à ne pas confondre

### Resynthèse dans le domaine `drum_ai`

La banque conserve son organisation MIDI, ses classes d'instruments, ses
articulations, ses paliers de vélocité et ses round robins. Chaque frappe est
cependant reconstruite par `drum_ai` ou par un modèle qui en a été distillé.

Cet objectif est le plus accessible. Il peut améliorer fortement la qualité
perçue, mais ne garantit pas la conservation de l'identité acoustique du kit
source.

### Amélioration fidèle de la banque source

Le modèle doit préserver l'accordage, les modes de résonance, le caractère des
fûts et cymbales, les articulations et la réponse aux vélocités. Seules la qualité
de captation, la définition des transitoires, la balance spectrale, la dynamique
et l'ambiance doivent converger vers le domaine appris par `drum_ai`.

Cette seconde formulation correspond réellement à un transfert de style. Elle
est plus ambitieuse et demande une séparation explicite entre contenu acoustique
et domaine de production.

## Méthodes candidates

### 1. Inversion latente

Pour chaque frappe source, rechercher un code latent `z` tel que la sortie de
`drum_ai` conserve ses propriétés principales :

- instrument et articulation ;
- intensité et enveloppe temporelle ;
- fréquence fondamentale ou modes dominants ;
- durée et structure de décroissance.

Le son reconstruit bénéficie directement du prior de `drum_ai`.

Avantages : faible développement initial, utilisation directe du modèle et
qualité potentiellement élevée. Limites : couverture incomplète du kit source,
optimisation lente par frappe et altération possible de son identité. Cette voie
exige un accès différentiable au générateur ou à son espace latent.

### 2. Encodeur d'inversion et adaptateurs

Après obtention de codes latents par optimisation, entraîner un encodeur rapide :

```text
frappe source -> encodeur -> latent drum_ai -> générateur drum_ai
```

Des adaptateurs légers, par exemple FiLM ou LoRA, pourraient élargir le domaine
du générateur sans réentraîner tout le système. Cette voie offre une conversion
rapide et une force de transfert réglable. Elle demande néanmoins une
régularisation forte pour éviter la dérive ou la mémorisation.

Si le modèle et ses poids sont accessibles, cette option est le meilleur
prolongement d'une inversion latente concluante.

### 3. Transformation neuronale résiduelle

Le modèle reçoit la frappe originale et prédit une correction :

```text
sortie = branche source préservée + correction apprise + ambiance générée
```

Le traitement peut séparer transitoire, corps/résonance et ambiance. La branche
source borne la transformation et aide à conserver l'identité du kit. Cette
approche est mieux adaptée à l'objectif d'amélioration fidèle, mais nécessite
des références appariées ou un entraînement non apparié soigneusement contrôlé.

### 4. Distillation depuis `drum_ai`

Si `drum_ai` sait générer selon l'instrument, l'articulation et la vélocité, il
peut servir de teacher :

1. générer un corpus cible conditionné ;
2. associer chaque source à une condition équivalente ;
3. entraîner un modèle de resynthèse ou de transformation ;
4. distiller vers un modèle plus léger si un usage direct l'exige.

Ces exemples ne sont pas appariés échantillon par échantillon. Une loss de forme
d'onde brute serait donc mal posée. L'apprentissage doit comparer des attributs
conditionnels et des distributions perceptuelles.

Cette méthode reste praticable si `drum_ai` est accessible comme boîte noire.

### 5. Traduction non appariée

Un encodeur sépare approximativement le contenu — instrument, articulation,
vélocité et accordage — du domaine sonore — captation, espace et production. Le
décodeur reconstruit ensuite la frappe dans le domaine `drum_ai`.

Des losses de reconstruction, identité, cycle, STFT multi-résolution, enveloppe,
transitoire et cohérence de vélocité seraient nécessaires. Cette solution est la
plus flexible, mais aussi la plus risquée : permutation d'articulations,
remplacement silencieux du kit, collapse de diversité ou mémorisation.

## Expérience initiale recommandée

Commencer avec une banque personnelle entièrement documentée et un périmètre
réduit : kick, snare et toms, micros proches ou rendu mono/stéréo simple,
articulations étiquetées, plusieurs vélocités et répétitions indépendantes.

Les cymbales devraient rester hors du premier test. Leur signal bruité, long et
très variable rend l'inversion et la conservation d'identité beaucoup plus
difficiles.

Ordre proposé :

1. auditer les interfaces et les droits réels de `drum_ai` ;
2. construire une baseline DSP : alignement, égalisation, enveloppe, dynamique,
   saturation et ambiance ;
3. tester l'inversion latente sans modifier le générateur ;
4. ajouter un encodeur et de petits adaptateurs seulement si la couverture du
   latent est suffisante ;
5. tester une transformation résiduelle pour mieux préserver le kit ;
6. comparer en écoute aveugle contre la source, la baseline DSP et la resynthèse
   complète.

Pour une conversion de banque hors ligne, le teacher peut être lourd : inversion
itérative, diffusion ou traitement multibande. Le résultat exporté reste de
l'audio pré-calculé et n'ajoute aucun coût au lecteur de samples. CPU et latence
ne deviennent des contraintes majeures que pour une génération en direct.

## Évaluation

### Conservation du contenu

- erreur de placement du transitoire ;
- accordage et déplacement des pics modaux ;
- enveloppe et temps de décroissance ;
- exactitude de classification de l'articulation ;
- monotonie du niveau et du timbre avec la vélocité ;
- cohérence des round robins.

### Acquisition du domaine `drum_ai`

- proximité spectrale et temporelle avec ses sorties ;
- comportement des transitoires et de la dynamique ;
- rapport son direct/ambiance ;
- préférence perceptuelle en écoute aveugle.

Une proximité d'embeddings seule n'est pas une preuve de succès : elle peut
récompenser le remplacement du kit plutôt que son amélioration fidèle.

### Diversité et mémorisation

- diversité intra-instrument et intra-vélocité ;
- absence de collapse vers quelques frappes ;
- recherche des plus proches voisins dans les données autorisées de `drum_ai` ;
- corrélations temporelles et similarités spectrales anormalement élevées ;
- test sur un instrument ou un kit entièrement tenu hors entraînement.

### Écoute

L'écoute doit poser séparément trois questions :

1. quelle version paraît la mieux enregistrée ou produite ?
2. quelle version ressemble le plus au kit source ?
3. les articulations, vélocités et variations restent-elles crédibles ?

Des motifs MIDI complets sont nécessaires en plus des frappes isolées. Ils
révèlent les ruptures entre vélocités et les round robins incohérents.

## Hypothèse falsifiable

> Le prior de `drum_ai` peut améliorer la qualité perceptuelle d'une banque
> personnelle au-delà d'une transformation DSP, sans dégradation significative
> de l'identité du kit, de ses articulations, de sa réponse aux vélocités ni de sa
> diversité.

Une préférence accrue accompagnée d'une perte nette d'identité ne valide que la
resynthèse dans le domaine `drum_ai`, pas le transfert fidèle.

Les seuils quantitatifs doivent être préenregistrés après un pilote de métrologie,
pas inventés dans cette note prospective.

## Provenance et risque commercial

Le risque est fortement réduit si les droits sur `drum_ai`, ses poids, ses
données d'entraînement et ses sorties sont maîtrisés, et si la banque source est
personnelle avec autorisations documentées.

DrumGizmo ne doit pas être traité comme une licence unique. Les fichiers peuvent
déclarer les informations de licence propres à chaque kit :
[documentation du format](https://drumgizmo.org/wiki/doku.php?id=documentation%3Afile_formats).
Certaines banques, comme
[DRSKit](https://drumgizmo.org/wiki/doku.php?id=kits%3Adrskit), sont annoncées en
CC BY 4.0, mais chaque kit doit être audité séparément et ses obligations
respectées.

La combinaison la plus sûre est donc : `drum_ai` contrôlé en interne, banque
personnelle documentée, registre de provenance et audit anti-mémorisation.

## Architecture de référence à explorer

```text
source
  |-- analyse causale du contenu
  |-- projection ou conditionnement drum_ai
  |-- reconstruction par le prior génératif
  `-- mélange avec une branche résiduelle source
```

Un contrôle de force permettrait de parcourir le continuum suivant :

```text
restauration légère <- préservation du kit -> resynthèse drum_ai complète
```

## Pertinence pour la campagne d'amplificateur neuronal

Cette piste concerne la batterie et doit rester une lignée prospective distincte.
Elle ne justifie aucune modification du goal d'amplificateur actif.

Elle offre cependant plusieurs analogies méthodologiques utiles : teacher lourd
hors ligne, étudiant ou branche résiduelle déployable, séparation entre contenu
physique et domaine perceptuel, ablation contre une baseline DSP, tests de
préservation et validation d'écoute indépendante des métriques numériques.
