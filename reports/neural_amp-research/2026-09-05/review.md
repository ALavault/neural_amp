---
title: "Au-delà du score moyen : pistes pour une émulation d'amplificateur fidèle"
date: "2026-09-05"
lang: fr
bibliography: review.bib
link-citations: true
---

# Résumé

La piste prioritaire est de relier les erreurs à la mémoire réellement nécessaire au dispositif,
puis de choisir une représentation et un entraînement adaptés à cette mémoire.
Un observer plus gros n'est pas, à lui seul, une hypothèse scientifique suffisante.

Cette revue exploratoire retient 33 références distinctes. Elle articule trois résultats attendus
pour une prochaine étude : un diagnostic des erreurs temporelles, une comparaison des modèles sur
les mêmes coordonnées et historiques, et un essai borné d'une voie lente compacte.
Aucun résultat du job en cours n'a été consulté pour cette recherche et aucun entraînement additionnel
n'a été lancé. Les pistes proposées ne modifient pas la campagne gelée.

# Périmètre et méthode

Question : quelles limites de représentation, d'apprentissage et de mesure empêchent une émulation
dry/wet causale de reproduire fidèlement un dispositif analogique, et quelles expériences peu
coûteuses permettent de départager ces explications ?

Période recherchée : 2016 au 5 septembre 2026. Inclusion : travaux originaux sur émulation
d'effets, filtres différentiables, évaluation et données; travaux adjacents sur états et gradients
lorsqu'un mécanisme transférable est explicite. Les travaux de synthèse audio générale sont
classés comme preuves indirectes. Prépublications et communications courtes sont admises mais
ne sont pas assimilées à des réplications indépendantes.

Sources : arXiv, archive DAFx, PMLR; complément ISMIR, Frontiers et dépôts institutionnels.
DBLP sert au contrôle bibliographique. Le CLI `paper-search` est absent; recherche web ciblée
utilisée en remplacement. Il ne s'agit pas d'une interrogation exhaustive des API de trois bases.
Les requêtes et exclusions sont conservées dans `sources/search_log.md`.

Les titres/résumés des 33 notices retenues ont été examinés; des extraits du texte intégral de
six travaux centraux ont complété la lecture. Aucun accord inter-évaluateurs, score de citations
ou évaluation formelle de risque de biais n'est revendiqué. Les intérêts financiers et financements
n'ont pas été extraits systématiquement. On privilégie la pertinence et la qualité du protocole
audio plutôt que la notoriété des auteurs.

La déduplication fusionne notamment la prépublication 2502.14405 et son article Frontiers.
Wright 2019 et Wright 2020 restent deux publications, mais leurs preuves ne sont pas indépendantes.
Les deux publications Comunità 2025 ont des fonctions distinctes : framework et comparaison.

Le corpus satisfait la cible de 30-50 références du skill. La recherche reste une revue exploratoire,
pas une revue systématique PRISMA complète : le nombre brut de résultats du moteur web n'est pas
un export exhaustif et n'est pas inventé.

Flux de sélection documenté :
- Résultats bruts multi-requêtes : nombre non normalisé.
- Notices primaires retenues et dédupliquées : 33.
- Lecture : notices/résumés pour le corpus; extraits de texte intégral pour six références.
- Méta-analyse : aucune; jeux de données, réglages et métriques hétérogènes.

Les DOI présents sont contrôlés séparément par le vérificateur du skill. Un DOI qui résout ne
prouve ni la qualité de l'étude ni la reproductibilité d'un score. Pour les références sans DOI
renseigné, la notice primaire liée est l'identifiant bibliographique.

Vérification effectuée : quatre DOI résolvent; trois métadonnées CrossRef récupérées.
Pour KLANN, le DOI résout mais CrossRef n'a pas fourni de métadonnées au vérificateur;
titre, auteurs, année et DOI sont corroborés par la notice institutionnelle Aalto.
Les 29 autres références sont reliées à leurs notices primaires; aucun DOI absent n'est inventé.

# Ce que la littérature permet de conclure

## Architecture et dispositif doivent être distingués

Les résultats publiés ne désignent pas un vainqueur universel. La comparaison multi-effets de
Comunità et collaborateurs favorise les SSM et le conditionnement temporel dans son cadre,
alors que la comparaison de Simionato et Fasciani décrit des avantages différents selon
distorsion, égalisation et compression. Les études ne partagent pas toutes les mêmes données,
budgets ou critères : leur divergence n'est pas une contradiction expérimentale démontrée.
[@comunita2025; @simionato2024a]

Des récurrences modestes sont des baselines crédibles pour l'amplification. Les travaux Wright
montrent leur intérêt face à des convolutions plus lourdes; ils ne garantissent pas une
généralisation à toutes les mémoires ou tous les réglages. Le résultat historique obtenu sur
référence SPICE par Damskägg doit être distingué d'un résultat sur matériel capturé.
[@wright2019; @wright2020; @damskagg2018]

Conséquence pour le projet : comparer la capacité utile à budget contrôlé, avec modèles
correctement initialisés et mêmes sources, plutôt que déduire la qualité du nombre de paramètres.
NablAFx est une base d'implémentation et d'analyse à examiner; son existence ne valide pas
automatiquement un adaptateur local. [@nablafx2025]

## La mémoire conservée et le gradient appris ne sont pas la même chose

Dans notre configuration connue, 12000 / 48000 = 0,25 seconde de TBPTT. Un état peut transporter
une information au-delà de cette durée, tandis que le gradient est coupé à la frontière.
Cela ne prouve pas un échec; cela expose une hypothèse d'optimisation non testée.

Tallec et Ollivier décrivent le biais de la troncature et une correction pondérée.
Aicher et collaborateurs adaptent l'horizon sous des hypothèses de décroissance du gradient.
Leurs résultats sur tâches synthétiques ou de langage ne démontrent pas directement un gain
sur amplificateurs. [@tallec2017; @aicher2020]

Les études sur compresseurs optiques rendent cette question concrète : les historiques et
les combinaisons attaque rapide / release lente restent difficiles dans leur évaluation.
S4, S4D, LRU et Mamba fournissent des mécanismes de mémoire, pas un substitut à la vérification
de l'horizon effectivement appris. [@simionato2024b; @gu2021; @gu2022; @orvieto2023; @gu2023]

## Des dynamiques compactes méritent un vrai contrôle

KLANN utilise une représentation non linéaire et des filtres linéaires dans l'espace latent.
Les IIR différentiables, filtres all-pole et tone stacks state-space montrent d'autres façons
de factoriser la dynamique. Leur intérêt ici est d'offrir des contrôles compacts et interprétables,
pas de garantir une supériorité du gray-box. [@huhtala2024; @kuznetsov2020; @yu2024; @sinjanakhom2024]

L'analogie avec l'identification de systèmes est particulièrement utile : les encodeurs d'état
initial permettent d'apprendre sur sections indépendantes. Mais la méthode Beintema utilise
un historique d'entrées ET de sorties. Une adaptation black-box déployable ne peut pas réclamer
le wet réel à l'inférence : encodeur dry-only, préchauffage causal ou observateur autonome
doivent être explicitement séparés du contrôle oracle. [@beintema2021]

DDSP, les ODE neuronales et le contrôle latent d'un phaser alimentent cette factorisation.
Les états de circuit accessibles en simulation et les solveurs continus changent toutefois
les conditions d'identification. [@engel2020; @parker2019; @wilczek2022; @carson2023]

## Les scores et les oreilles répondent à des questions différentes

La loss perceptuelle de Wright et l'étude de Cassidy séparent précision, réalisme et préférence.
Un meilleur ESR ne démontre donc pas une préférence en écoute. Les courbes de résidu sont
utiles pour localiser une erreur, mais un fichier presque silencieux peut afficher un ESR élevé
pour une très faible erreur absolue. [@wrightloss2020; @cassidy2023]

Pour le failure mining, rapporter conjointement ESR, MAE, niveau cible, résidu spectral,
transitoires et position temporelle. Conserver les exemples difficiles sans les transformer
en données de sélection de la campagne confirmatoire. EGFxSet enrichit les observations sur
matériel réel, mais ses notes isolées ne remplacent pas des phrases musicales et histoires
longues. [@pedroza2022]

La répétabilité matérielle reste une inconnue locale : sans répétitions de captures identiques,
on ne peut pas attribuer tout résidu à la modélisation. Une nouvelle capture ne doit jamais
être simulée en modifiant le dry tout en conservant artificiellement le même wet.

## Antialiasing et coût sont des axes séparés

Le lissage des activations et l'ADAA récurrent donnent des pistes pour réduire les images
repliées; leur effet sur la tonalité doit aussi être mesuré. Adapter une fréquence d'échantillonnage
change la dynamique d'un RNN si sa récurrence n'est pas adaptée. [@sato2025; @mikkonen2025; @carson2024]

Les grandes dilatations TCN peuvent offrir du contexte à moindre coût. TFiLM combine modulation
récurrente et convolution; SaShiMi exploite plusieurs échelles pour la génération audio.
Ces idées motivent une voie lente basse cadence, mais ne prouvent pas qu'elle préservera les
transitoires d'un effet conditionné par son entrée. [@steinmetz2021; @birnbaum2019; @goel2022]

WaveNet est à l'origine un modèle de génération autorégressive, distinct de notre régression
dry/wet. La prépublication iOS de 2026 concerne surtout pruning et exécution sparse :
un gain de déploiement ne vaut pas un nouveau record de fidélité. [@oord2016; @sato2026]

# Idéation : 15 candidats conservés

Les moteurs créatifs utilisés sont la reformulation du problème, l'analogie avec l'identification
de systèmes et l'analyse des contradictions mémoire/gradient/coût. Le classement ci-dessous
est un jugement de recherche, non une mesure empirique.

| ID | Proposition | Mécanisme et test discriminant | Décision |
|---|---|---|---|
| I01 | Carte d'horizon mémoire utile | Même suffixe, historique contrôlé; mesurer dépendance du résidu au contexte | Priorité 1 |
| I02 | Observer lent basse cadence | États compacts au rythme du contrôle, voie audio rapide inchangée | Priorité 2 |
| I03 | État initial dry-only appris | Estimer l'état depuis le passé dry; comparer au préchauffage causal | Priorité 3 |
| I04 | Gradient long seulement pour la voie lente | Détacher le fast plus souvent que les états lents | Pilote conditionnel |
| I05 | KLANN minimal comme contrôle | Lift + biquads + projection contre gros observer | À inclure dans I02 |
| I06 | Cartographie des constantes de temps | Prédire échecs attaque/release depuis dynamique estimée | À inclure dans I01 |
| I07 | Borne de répétabilité du réel | Captures répétées du même probe à réglage fixe | Priorité métrologique |
| I08 | Acquisition informative par désaccord | Capturer surtout où deux modèles figés divergent | Futur, demande nouvelles captures |
| I09 | Loss sensible aux transitoires | Pondération préenregistrée des attaques, contrôle ESR global | Après diagnostic I01 |
| I10 | Résidu séparé harmonique/bruit | Modèle déterministe plus estimation du plancher aléatoire | Attendre répétabilité I07 |
| I11 | Transfert de fréquence cohérent | Garder les constantes de temps physiques lors du resampling | Hors cycle qualité actuel |
| I12 | Activations antialiasées | Réduire ASR sans confondre baisse ESR et baisse aliasing | Futur protocole séparé |
| I13 | Expert léger par régime | Routage causal selon niveau/historique, sans contrôle futur | Complexité non justifiée encore |
| I14 | Recherche d'architecture plus grosse | Ajouter couches/états sans mécanisme de défaut identifié | Rejetée pour l'instant |
| I15 | Pruning/distillation mobile | Convertir fidélité acquise en coût de déploiement | Hors scope actuel |

# Trois pistes à approfondir

## I01 — Identifier la mémoire manquante avant d'ajouter des paramètres

Pitch : les modèles peuvent bien reproduire le timbre moyen tout en échouant sur une
attaque dépendante de ce qui précédait. Nous proposons d'estimer l'horizon utile du contexte
et du gradient afin de localiser si l'erreur vient de l'état, de son apprentissage ou de la loss.

Trois expériences proposées :
1. Contrôle synthétique à mémoire connue, avec même débit, loss et architecture; varier
   l'horizon de gradient et mesurer l'erreur après transitions.
2. Sur validation autorisée, comparer le modèle figé avec historique continu et états réinitialisés
   à différentes distances du passage scoré. Le wet reste celui de la capture originale.
3. Après préenregistrement distinct, comparer deux horizons d'entraînement sur mêmes sources,
   seeds et budgets. Séparer le gain sur transitoires du gain global.

Invalidateur : aucune sensibilité reproductible à l'historique au-delà de 250 ms, ou aucun gain
de l'horizon plus long à budget comparable. Objection forte : on mesure seulement un artefact de
reset. Réponse : un contrôle à mémoire connue et une fenêtre de scoring après préchauffage
doivent départager cette explication. Originalité non établie; diagnostic d'abord.

## I02 — Une mémoire lente compacte plutôt qu'une seconde grosse voie audio

Pitch : une grande partie de la dynamique pourrait être pilotée par quelques états de contrôle
lents, alors que les harmoniques exigent un traitement rapide. Un petit observateur causal
multicadence modulerait la voie rapide, avec un contrôle KLANN/IIR de même budget.

Trois expériences proposées :
1. Observer simple constitué d'enveloppes attaque/release apprenables; vérifier son pouvoir
   explicatif avant toute architecture sélective.
2. Comparer cadence audio et cadence de contrôle réduite avec filtre causal et délai explicite.
3. Comparer IIR latent compact, S4D et modèle actuel sur les régimes identifiés par I01.

Invalidateur : gain absent face aux enveloppes simples, ou erreurs transitoires dues à la cadence
réduite. Objection : gray-box et TFiLM existent déjà. Réponse : la contribution doit être la
démonstration d'une décomposition valide par régime et son coût, pas le nom de l'architecture.

## I03 — Apprendre l'état initial avec des entrées accessibles

Pitch : des segments courts facilitent l'optimisation mais réinitialisent souvent le système
dans un état irréaliste. Un encodeur causal du passé dry fournirait un état initial cohérent
sans exiger la sortie du dispositif pendant l'inférence.

Trois expériences proposées :
1. État nul contre préchauffage dry explicite, à contexte réellement identique.
2. Encodeur dry-only contre oracle entrée/sortie strictement marqué non déployable.
3. Vérifier que l'amélioration persiste en streaming complet et après changement de niveau.

Invalidateur : le gain disparaît lorsque le préchauffage de référence est équitable, ou l'encodeur
nécessite du wet futur. Objection : états latents non identifiables. Réponse : juger la simulation
hors fenêtre et la robustesse, sans prétendre retrouver les variables physiques exactes.

# Pilote de deux semaines proposé

Aucun pilote n'est lancé dans ce document.

- Jours 1-2 : tableau des conditions d'évaluation, références exécutables, tailles et origines
  des contextes, délais et supports de scoring. Fixer les critères avant nouveaux scores.
- Jours 3-5 : I01 sur un système synthétique connu puis sur validation autorisée; conserver
  également les résultats négatifs et les résidus en faible énergie.
- Jours 6-9 : un seul contrôle compact I02, avec deux seeds au minimum pour un signal exploratoire;
  plafond proposé de 12 GPU-h au total, distinct de la campagne actuelle.
- Jours 10-12 : comparaison du gain par régime et du coût complet, pas uniquement temps d'update.
- Jours 13-14 : décision continuer/abandonner; protocole confirmatoire séparé si justifié.

Critères candidats à préenregistrer : amélioration ESR d'au moins 10 % dans le régime visé,
pas de régression globale supérieure à 5 %, stabilité numérique et parité streaming.
Ces nombres sont des propositions de pilotage, pas des seuils ajoutés au benchmark en cours.

# Ce que cette recherche ne permet pas d'affirmer

Aucune supériorité SOTA du projet, aucune causalité entre nos échecs logiciels et une famille
d'architectures, aucune garantie qu'une nouvelle loss améliorera l'écoute.

Le matériel du corpus inclut amplis, pédales et compresseurs : généraliser un résultat Ampeg
à Rodent/Fuzzy Logic sans confirmation serait injustifié. Un avantage du teacher sur un
contrôle ablaté n'est pas une victoire contre le meilleur comparateur ouvert.

Un même nombre d'updates n'est pas un budget de calcul égal. Des coordonnées de scoring
identiques n'impliquent pas des historiques identiques. Enfin, les clocks murales, la validation
et l'admission GPU doivent être tracées pour rendre un avantage de coût interprétable.

# Livrables et limites de cette note

- `sources/corpus.json` : 33 notices, rôle, limites et niveau de lecture.
- `sources/search_log.md` : requêtes, exclusions et limites de recherche.
- `review.bib` : références utilisables avec citeproc.
- `sources/citations_to_verify.md` : DOI renseignés à contrôler automatiquement.
- PDF : rendu de secours avec pdfLaTeX installé; pandoc/xelatex absents et installation système
  sans mot de passe indisponible. Les citations du PDF restent liées à la bibliographie.
- La vérification automatisée des DOI et le rendu sont séparés de la rédaction scientifique.

# References
