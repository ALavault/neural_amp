# Audit technique et scientifique de la gate mécaniste — 2026-08-28

## Verdict

La gate est une issue scientifique valide `NO-GO-ARCH`, pas une panne
d'instrumentation. Les dix trajectoires prévues sont complètes, finies et
ledgerées; les sorties Blackstar/UA1176 et toute donnée physique sont restées
fermées. Aucune des cinq familles n'a passé simultanément ses critères.

## Résultats discriminants

- Le contrôle `micro_tcn_x2` obtient ESR 0,16238 sur le système statique et
  0,52893 sur le dynamique, mais échoue aux gardes d'amplitude : `gain_error`
  −0,391/−0,690, et corrélation dynamique 0,803 <0,9.
- `phys_det_tcn_x2` régresse de 0,21 % sur l'ESR dynamique par rapport au
  contrôle. Sa régression statique de 0,42 % reste sous le plafond de 5 %.
- `phys_s6_tcn_x2` régresse de 0,32 % sur le dynamique au meilleur poids 0,01.
  Les poids 0,05 et 0,10 sont encore légèrement plus faibles; sa régression
  statique de 0,71 % reste sous le plafond.
- La cascade réduit l'ESR deux-clippers de 0,64329 à 0,48470, soit 24,65 % :
  progrès réel mais inférieur à la gate de 50 %. Elle échoue aussi aux gardes
  (`gain_error` −0,622, corrélation 0,772).
- Le mono-stage RF2047 est le plus faible sur cette fixture : ESR 0,64329,
  `gain_error` −0,737, corrélation 0,639.

## Audit technique

Tous les gradients, losses et rendus sont finis. Les losses train diminuent
pour chaque trajectoire et les validations sont cohérentes avec elles. Les dix
checkpoints, historiques, snapshots de source et cinq entrées du registre global
sont présents. Il n'y a donc pas d'indice de NaN, de reset incorrect, de split
ouvert, de run manquant ou de corruption de mesure.

Le facteur commun est une sous-estimation d'amplitude. Sur les 2 048 derniers
échantillons de validation, les systèmes cibles demandent environ +6,46 dB
(statique), +10,52 dB (dynamique) et +8,82 dB (deux-clippers) par rapport au dry.
Après 500 updates, les prédictions restent autour de 0,29–0,30 RMS contre des
cibles à 0,47–0,75 RMS. Les gradients finaux restent non nuls : les courbes ne
démontrent pas une convergence achevée.

## Audit scientifique

La gate confond deux questions : compétence minimale du modèle de contrôle et
valeur relative d'une idée architecturale. Comme le contrôle micro-TCN lui-même
échoue aux gardes anti-effondrement, la campagne ne peut pas attribuer l'échec
global à une limite fondamentale des architectures. Le planning de 500 updates
n'avait pas de gate de compétence préalable ni de justification de convergence;
il est insuffisant pour les gains synthétiques imposés par les fixtures.

Les comparaisons appariées restent néanmoins informatives dans cette enveloppe :
les deux bus physiques n'apportent aucun gain à budget égal, alors que la cascade
apporte +24,65 % sans atteindre le seuil préenregistré. Le verdict signifie donc
« aucune famille ne passe AMP-QUALITY-ARCH-v1 tel que gelé », et non « ces idées
ne peuvent jamais battre l'état de l'art ».

## Steering recommandé pour une éventuelle v2

Ne pas reprendre ni modifier ces runs. Ouvrir une nouvelle lignée qui sépare :

1. une gate de compétence non comparative du micro-TCN, avec budget de
   convergence fixé par un pilot indépendant ou une courbe d'apprentissage
   préenregistrée;
2. les comparaisons architecturales seulement après passage des gardes par le
   contrôle;
3. un seuil cascade motivé par puissance/variance plutôt qu'un gain arbitraire
   de 50 %;
4. une initialisation ou tête de gain capable de représenter immédiatement les
   +6 à +11 dB des fixtures, sans normaliser chaque split ni masquer la physique.

La nouvelle lignée devra conserver ce `NO-GO-ARCH` et ne pourra réutiliser les
résultats v1 pour choisir rétrospectivement son seuil final.
