# Protocole prospectif AMP-QUALITY-ARCH-v1

Le contrat exécutable est `configs/amp_quality_arch_v1/protocol.yaml`, copié
exactement dans `PROTOCOL_LOCK.yaml` avant le premier run scientifique. Le
preflight gelé est enregistré; toute dérive entre les deux fichiers échoue.

## Question

Une architecture sans contrainte de parenté avec FSSR peut-elle battre le plus
fort comparateur ouvert et reproductible sur fidélité objective et perceptuelle,
tout en respectant une enveloppe C++ temps réel stricte ?

## Stratégie

Les quatre comparateurs exécutables sont réentraînés loyalement. Six familles candidates
intègrent les idées side-thread sans leur accorder de promotion : S6 pur,
micro-TCN, hybride observateur S6/bus physique/micro-TCN, bus déterministe,
RF2047-TFiLM et cascade deux-clippers. Les deux losses autorisées par famille
sont départagées à 1 000 updates sur validation INTERNAL_DEV; la meilleure seule
continue à 5 000. Seuls `micro_tcn_x2` et `phys_s6_tcn_x2` peuvent devenir
teachers larges; au plus trois finalistes restent permis par le cap global.

L'observateur physique reçoit seulement l'entrée dry et ses descripteurs causaux.
Les descripteurs wet sont des cibles auxiliaires d'entraînement et sont absents de
l'interface d'inférence. Son poids auxiliaire est choisi uniquement sur fixtures
synthétiques parmi trois valeurs préenregistrées.

La qualification de loss réserve au plus 36 IDs et 44 subtrajectoires de
modèle. Tout candidat qui échoue au mécanisme est retiré sans remplacement; le
compte effectif devient `4 × (4 comparateurs + candidats survivants)`.

## Séquence fail-closed

Le preflight vérifie d'abord sources, données, caps, IDs et frontières scellées.
Le mécanisme doit ensuite passer numérique, aliasing et benchmark C++ de squelette
avant toute lecture physique. Après screen et teacher, un student n'est distillé
que si son teacher passe la double gate et que sa version déployable échoue.
Architecture, loss, schedule et export sont gelés avant Blackstar/UA1176.

Après le gel, les modèles sont entraînés sur leurs paires train avec un schedule
fixe; aucune sélection de checkpoint ni adaptation architecturale Blackstar/UA
n'est permise. Les sorties test ne sont ouvertes qu'une fois après parité C++ et
benchmark. L'écoute ne démarre que si ESR, ASR et runtime passent.

Le benchmark C++ de squelette est désormais acquis dans `NATIVE_SKELETON.json`.
Il a testé les 24 couples famille/profil sans poids entraînés, avec 30 répétitions
A/B entrelacées contre le NAM A2 épinglé. Les six profils déployables retenus sont
`max`; ce choix est gelé avant toute lecture des signaux physiques.

Les adaptateurs locaux TCN–TFiLM et S4–TFiLM passent une parité directe contre
le commit NablAFx épinglé. Leurs variantes S/L sont deux subtrajectoires sous un
même run de famille et sont départagées globalement sur les validations Fulltone
et Big Muff, jamais par appareil. Leur TFiLM utilise le maximum du bloc courant :
la campagne aligne donc explicitement des délais causaux conservateurs de
508–1 270 échantillons. Ils restent comparateurs de fidélité, mais la gate
runtime ≤48 échantillons ne s'applique qu'au candidat final déployable.

Le préflight `PHYSICAL_TRAINING_FEASIBILITY.json` a ensuite mesuré les onze
charges réelles, avec leur contexte gelé et la loss NablaFX exacte. Toutes
passent le cap de 6 h projetées par trajectoire; le cascade candidat est limitant
à 4,598 h projetées pour 5 000 updates.

## Interprétation

`GO-ARCH` exige toutes les gates simultanément. Toute issue scientifique valide
qui manque une gate donne `NO-GO-ARCH`. Une exposition scellée prématurée, une
panne de métrique, un run hors budget ou une reprise interdite donne `INVALID`.
