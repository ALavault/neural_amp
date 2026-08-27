# Steering scientifique après le NO-GO FSSR-NAM

**Audience :** responsable scientifique et équipe d'implémentation  
**Date :** 27 août 2026  
**Question :** quelle campagne minimale peut distinguer un échec de protocole,
d'optimisation ou d'architecture, sans effacer le NO-GO M4 ni ouvrir une nouvelle
recherche incontrôlée ?

## Réponse directe

Le NO-GO reste valide pour **S3/S4 tels qu'entraînés à M4**, mais il ne démontre
pas que l'idée fast-slow structurée-résiduelle est sans avenir. Le prochain geste
scientifique n'est ni une grande matrice ni un nouveau module exotique : c'est une
campagne de compétence et d'identification en quatre étages.

1. Reproduire une baseline Big Muff publiée sur les données natives.
2. Séparer l'effet de la loss et du budget de celui de l'architecture.
3. Tester un résidu à horizon intermédiaire avec apprentissage par étapes.
4. Tester un cœur à deux non-linéarités en cascade seulement si les trois premiers
   étages l'autorisent.

L'antialiasing et l'état lent deviennent conditionnels à des diagnostics positifs.
Le cœur scientifique proposé est désormais : **quelles contraintes de structure et
quelle procédure d'identification permettent à un modèle compact de rejoindre le
front noir ?**

## Portée et hypothèses

- Le registre M0-M6, le classement NO-GO et les runs existants sont immuables.
- Seules les données `INTERNAL_DEV` et synthétiques sont dans le périmètre.
- `EXTERNAL_REPORT_ONLY` reste verrouillé.
- Les comparaisons publiées servent de contrôle de plausibilité, pas de résultat
  directement comparable lorsque split, réglage ou budget diffèrent.
- Le rapport privilégie les articles, appendices, codes et dépôts officiels jusqu'au
  27 août 2026. Aucun résultat 2026 pertinent n'a été trouvé dans la recherche
  ciblée ; les sources décisives vont de 2015 à 2025.

## Ce que le NO-GO établit réellement

Le smoke test a exécuté 24 runs sans NaN. Sur Fulltone, l'ESR médiane est 0,0639
pour A2, 0,2112 pour S3 et 0,2158 pour S4. Sur Big Muff, elle est respectivement
0,5969, 0,9670 et 0,9595. Le recovery à largeur 31 ne ferme que 0,4 % du gap
Fulltone et 25,2 % du gap Big Muff. Ces faits justifient pleinement l'arrêt M4 et
le NO-GO consigné dans l'[audit final](../FINAL_AUDIT.md).

Mais M4 contient trois confusions expérimentales :

- **Budget non saturé.** A2 atteint son meilleur checkpoint Big Muff à l'update
  200, la frontière imposée. La littérature entraîne la baseline récurrente Big
  Muff pendant 20 heures sur 5 min 42 s de train, alors que M4 utilise 120 s et
  200 updates. Dans l'étude ToneTwist/NablAFx, les modèles non récurrents disposent
  d'un plafond de 15 000 updates ([Wright et al., DAFx-19](https://dafx.de/paper-archive/2019/DAFx2019_paper_43.pdf),
  [Comunità et al., Frontiers 2025](https://www.frontiersin.org/journals/signal-processing/articles/10.3389/frsip.2025.1580395/full)).
- **Objectif non aligné.** M4 minimise MSE brute + `0.0005` MR-STFT, avec
  sélection sur ESR. L'énergie quadratique moyenne du train Big Muff vaut
  0,001548, contre 0,012501 pour Fulltone, soit 8,1 fois moins. La baseline
  historique Big Muff optimise au contraire un ESR préaccentué et un terme DC ;
  le code officiel expose `ESRPre` et le filtre passe-haut
  ([implémentation de Wright au commit consulté](https://github.com/Alec-Wright/Automated-GuitarAmpModelling/blob/e3146386b0fd0b562bc393231be3a5938cf9feac/dist_model_recnet.py)).
  Cette loss M4 reproduit fidèlement la configuration A2 figée ; le problème
  possible n'est donc pas une baseline A2 mal traitée, mais l'hypothèse non testée
  que son objectif est neutre entre architectures.
- **Capacité temporelle non testée.** Le résidu S3 couvre 31 échantillons
  (0,65 ms à 48 kHz), contre 6 347 échantillons (132 ms) pour A2. Le recovery
  augmente la largeur, jamais l'horizon. Il ne teste donc pas l'explication
  « mémoire intermédiaire manquante ».

Le même Big Muff, au même réglage sustain 5 / volume 10, atteint 0,1076 d'ESR
avec S4-TFiLM dans l'appendice ToneTwist, alors que les gray-box à une seule
non-linéarité restent entre 0,5869 et 0,7042. Cela ne fournit pas un score cible
strict pour notre prétraitement, mais démontre que le couple publié contient
beaucoup plus d'information apprenable que M4 n'en extrait
([appendice complet](https://github.com/mcomunita/nnlinafx-supp-material/blob/bbcb628afd9418777df254e901f252d4b17e37f4/Differentiable_Black_box_and_Gray_box_Modeling_of_Nonlinear_Audio_Effects___Appendix___Arxiv.pdf)).

## Matrice d'explications concurrentes

| Hypothèse | Indices favorables | Indices contraires | Confiance | Test discriminant |
|---|---|---|---:|---|
| P1 — apprentissage trop court / loss mal calibrée | A2 Big Muff s'améliore encore à l'update 200 ; baseline publiée très supérieure ; énergie cible faible | A2 apprend mieux que S3 sous la même loss | Haute | courbes 200/1k/5k avec deux losses |
| P2 — optimisation conjointe non identifiable | S3 démarre près de l'identité sur Fulltone puis progresse peu ; la littérature gray-box signale l'importance des LR par bloc et de l'initialisation | aucune preuve directe par Hessienne ou permutation de blocs | Moyenne-haute | apprentissage joint contre apprentissage étagé |
| P3 — trou de mémoire intermédiaire | RF S3 = 31 contre RF A2 = 6 347 ; élargir sans allonger échoue ; TFiLM aide les distorsions/fuzz | Big Muff peut être dominé par une non-linéarité quasi statique | Haute pour Fulltone, moyenne pour Big Muff | résidu RF 31 contre RF 2 047 à coût borné |
| P4 — topologie mono-waveshaper insuffisante | Big Muff comporte deux étages de clipping à diodes en cascade ; gray-box mono-WH publiées échouent | une spline et un résidu universels peuvent théoriquement approximer le système | Moyenne-haute | un étage contre deux étages, entraînement identique |
| P5 — dynamique lente déterminante | littérature TFiLM et état-espace positive sur fuzz/dynamique | S1 n'a pas secouru M4 ; ToneTwist classe ce Big Muff comme distortion | Moyenne-faible | test bursts/hystérésis avant toute nouvelle branche |
| P6 — aliasing cause principale du gap | S4 réduit les parasites synthétiques de 17,15 dB | S3 et S4 ont la même fidélité physique ; une sortie silencieuse peut tromper l'ASR | Faible | différer jusqu'au franchissement du gate de fidélité |

## Lecture de la littérature

La littérature renforce le principe fast-slow, mais pas son implémentation actuelle.
TFiLM applique une modulation lente à **chaque couche** du backbone court et divise
environ par deux l'erreur MR-STFT sur le fuzz étudié, alors que l'allongement naïf
du champ réceptif ne suffit pas toujours
([Comunità et al., ICASSP-23](https://mcomunita.github.io/files/comunita2023gcntfilm-paper.pdf)).
Notre GRU lent ne module que trois paramètres globaux du cœur et ne peut donc pas
réparer directement les représentations du résidu.

La faiblesse des gray-box simples n'est pas propre à ce dépôt. Sur 16 dispositifs,
les modèles gray-box ont une perte médiane supérieure aux familles black-box ; les
auteurs indiquent aussi que des multiplicateurs de learning rate par bloc sont
essentiels pour atteindre de meilleurs optima
([Comunità et al., Frontiers 2025](https://www.frontiersin.org/journals/signal-processing/articles/10.3389/frsip.2025.1580395/full)).
Les méthodes block-oriented plus anciennes procèdent par identification étagée et
utilisent des signaux de faible et fort niveau séparément pour éviter les ambiguïtés
entre filtres et non-linéarité
([Eichas et al., DAFx-15](https://dafx.de/paper-archive/2015/DAFx-15_submission_21.pdf)).

Enfin, la topologie est un vrai levier, pas seulement une préférence esthétique. Le
Big Muff mesuré comporte deux étages de clipping à diodes en cascade
([Wright et al., DAFx-19](https://dafx.de/paper-archive/2019/DAFx2019_paper_43.pdf)).
Un préprint DDSP multi-étages rapporte une fidélité comparable aux black-box avec
moins de 10 % des opérations par échantillon, ce qui rend la piste multi-blocs
crédible mais non encore confirmée pour nos pédales
([Yeh et al., arXiv 2024](https://arxiv.org/abs/2408.11405)).

## Steering proposé : campagne R1

### R1.0 — Gate de compétence de la baseline

Reproduire, à 44,1 kHz et sur les splits publiés complets, le LSTM-64 Big Muff du
code Wright. Commencer par la seed 0 ; si l'ESR test est ≤ 0,15, compléter avec
les seeds 1 et 2. Le papier rapporte 0,041 ; 0,15 est une borne volontairement
tolérante face aux différences de versions.

**STOP :** si aucune seed n'atteint 0,15, ne modifier aucun modèle. Auditer
prétraitement, alignement, TBPTT, loss et checkpointing.  
**GO :** si la compétence est démontrée, revenir au protocole interne 48 kHz.

### R1.1 — Factoriel loss × budget

Sur les deux dispositifs, seed 0, entraîner A2 et S3 jusqu'à 5 000 updates avec
checkpoints à 200, 1 000 et 5 000 :

- loss M4 (`MSE + 0.0005 MR-STFT`) ;
- loss de référence (`ESR préaccentué + DC`).

Les régularisations de spline et de résidu restent identiques entre ces deux
conditions ; leur curriculum n'est testé qu'à R1.2.

Cela représente huit trajectoires, pas 24 runs indépendants. Enregistrer énergie
de sortie, gain, gradient par bloc et énergie du résidu à chaque checkpoint.

**GO entraînement :** S3 Big Muff doit sortir du bassin silencieux
(`gain_error > -0.5`) et réduire l'ESR d'au moins 25 % par rapport à S3-M4.
**STOP loss :** si changer la loss n'affecte ni gain ni ESR à budget égal, ne plus
chercher les poids de loss.

### R1.2 — Identification étagée et horizon résiduel

Comparer deux S3 de coût proche : RF 31 et RF ≈ 2 047 échantillons (≈ 43 ms), le
second utilisant des convolutions causales dilatées séparables. Procédure :

1. ajuster gain/DC et réponse linéaire sur passages de faible niveau ;
2. ajuster spline et drive, cœur seul ;
3. geler le cœur et apprendre l'erreur avec le résidu, sans pénalité ;
4. dégel conjoint avec rampe de `lambda_residual` de 0 à sa valeur finale.

**GO horizon :** fermeture d'au moins 50 % du gap S3→A2 sur Fulltone et au moins
25 % sur Big Muff, sans hausse de coût théorique au-delà d'A2.  
**STOP horizon :** si le RF long n'active pas le résidu ou ferme moins de 10 % du
gap, passer au test topologique sans élargir davantage.

### R1.3 — Cœur cascade, seulement si autorisé

Créer un candidat minimal :

```text
H0 → spline1 → H1 → spline2 → H2 → + résidu RF43ms
```

D'abord, il doit battre d'au moins 50 % le cœur mono-spline sur un système
synthétique à deux clippers séparés par un filtre. Ensuite seulement, comparer les
deux cœurs sur Big Muff, à loss, seed, horizon, nombre d'échantillons vus et budget
CPU identiques.

**GO cascade :** au moins 30 % d'amélioration ESR Big Muff et réduction de moitié
de l'erreur de gain, sans régression Fulltone supérieure à 5 %.  
**STOP architecture :** sinon, abandonner la famille mono/cascade structurée pour
la fidélité H1 et basculer explicitement vers la distillation H2 ou le résultat
négatif.

### R1.4 — État lent et antialiasing deviennent conditionnels

N'activer la modulation lente que si des bursts et transitions de niveau révèlent
une trajectoire attaque/relâchement reproductible que le modèle statique ne capture
pas. Dans ce cas, moduler les activations du résidu (TFiLM léger), pas uniquement
drive/offset/gain.

Ne reprendre H3 qu'après `ESR < 0.15`, `gain_error > -0.2` et corrélation > 0,9 sur
le dispositif concerné. Sato et Smith ont explicitement exclu des modèles dont le
faible ASR provenait d'une sortie quasi silencieuse
([DAFx-25](https://dafx.de/paper-archive/2025/DAFx25_paper_50.pdf)).

## Budget et décision

Le steering diagnostique est plafonné à 19 nouvelles trajectoires : 3 pour la
compétence, 8 pour loss×budget, 4 pour horizon/étagement et 4 pour la cascade
(deux synthétiques, puis deux contrôles physiques). Une branche qui échoue son
gate libère son budget ; il n'est pas réalloué à une nouvelle idée. Seul le
meilleur mécanisme passe ensuite à trois seeds lorsqu'il ne les possède pas déjà.

Le résultat publiable visé n'est plus nécessairement « FSSR bat A2 ». Trois sorties
honnêtes sont possibles :

- **R-A :** une procédure d'identification étagée supprime le collapse et rend la
  structure compétitive ;
- **R-B :** la cascade ou l'horizon explique causalement le gap, donnant une
  contribution d'architecture bornée ;
- **R-C :** aucun mécanisme ne ferme le gap, renforçant le résultat négatif avec
  une autopsie mécaniste plutôt qu'une simple comparaison de scores.

## Limites et désaccords

- Le résultat Fulltone publié utilise un drive plus élevé que `O050`; il ne doit
  pas servir de seuil direct. Le Big Muff publié correspond au même réglage mais
  peut différer par prétraitement et rééchantillonnage.
- Les résultats DDSP multi-étages proviennent d'un préprint sur un amplificateur
  paramétrique, pas d'une réplication indépendante sur Big Muff.
- La loss ESR préaccentuée n'est pas nécessairement optimale pour A2 ; elle est
  ici un instrument diagnostique, pas un choix final.
- Aucun accès aux résultats externes n'a eu lieu pendant cette recherche.

## Arrêt de la recherche documentaire

Deux vagues ont couvert les sources primaires de baseline, les résultats complets
ToneTwist, les modèles multi-échelles, les gray-box block-oriented, les architectures
DDSP multi-étages et l'antialiasing. Les affirmations décisives ont été recoupées
avec l'appendice, le code officiel ou les artefacts locaux. Les recherches 2026
n'ont pas produit de source primaire supplémentaire susceptible de modifier la
décision ; poursuivre aurait surtout ajouté des variantes redondantes.
