# Steering scientifique après le NO-GO

## Décision

Le NO-GO reste valide pour FSSR-NAM S3/S4 **dans le protocole M4 exécuté**. Il ne
justifie cependant pas d'abandonner immédiatement l'hypothèse structurée-résiduelle.
M4 n'a pas encore séparé trois causes : apprentissage trop court, optimisation
défavorable aux blocs structurés et topologie insuffisante.

Je recommande une campagne R1 bornée à 19 trajectoires de diagnostic, avec gates
séquentiels. Seul le gate de compétence est promu jusqu'à trois seeds. La campagne
ne modifie ni les runs M0–M6, ni le classement NO-GO, ni le gel des données
externes.

## Pourquoi le résultat actuel n'est pas le dernier mot

Sur Big Muff, A2 atteint son meilleur checkpoint médian à l'update 200, exactement
la limite du smoke test : sa convergence n'est pas démontrée dans M4
([rapport M4](M4_CORE.md)). La baseline historique
utilise 5 min 42 s de train, une loss ESR préaccentuée + DC et 20 heures
d'entraînement ; M4 utilise 120 s et 200 updates
([Wright et al., DAFx-19](https://dafx.de/paper-archive/2019/DAFx2019_paper_43.pdf)).
Une étude plus récente autorise jusqu'à 15 000 updates aux modèles non récurrents
([Comunità et al., Frontiers 2025](https://www.frontiersin.org/journals/signal-processing/articles/10.3389/frsip.2025.1580395/full)).

M4 minimise une MSE brute alors que le checkpoint est choisi sur ESR
([configuration M4](../configs/training/m4_smoke.yaml)). L'énergie
quadratique du train Big Muff est 8,1 fois plus faible que celle de Fulltone ; les
termes spectral et de régularisation changent donc de poids relatif entre appareils.
Cela correspond au collapse observé (`gain_error ≈ -0.93`), sans le prouver à lui
seul. Cette formulation est bien celle d'A2 : le doute porte sur sa neutralité pour
une architecture structurée, pas sur la fidélité de la reproduction A2.

Enfin, le recovery a élargi le résidu sans allonger son champ réceptif : 31
échantillons (0,65 ms) contre 6 347 (132 ms) pour A2
([code du résidu](../src/fssr_nam/models/residual.py),
[audit A2](../experiments/summaries/m2_a2_architecture/architecture.json)). Il n'a donc jamais testé une
mémoire intermédiaire comparable. Sur le Big Muff au même réglage, une étude
ToneTwist rapporte 0,1076 d'ESR pour S4-TFiLM, mais 0,5869–0,7042 pour des gray-box
à une seule non-linéarité
([appendice complet](https://github.com/mcomunita/nnlinafx-supp-material/blob/bbcb628afd9418777df254e901f252d4b17e37f4/Differentiable_Black_box_and_Gray_box_Modeling_of_Nonlinear_Audio_Effects___Appendix___Arxiv.pdf)).
Le signal est apprenable ; la structure simple ne l'est peut-être pas assez.

## Hypothèses classées

| Rang | Hypothèse | Test qui peut la tuer |
|---:|---|---|
| 1 | Budget/loss ont créé un échec d'optimisation | courbes 200/1k/5k avec loss M4 contre ESR préaccentué |
| 2 | Le résidu manque d'horizon et s'active trop tard | RF 31 contre RF 2 047, apprentissage étagé |
| 3 | Le cœur mono-spline est topologiquement faux pour Big Muff | une contre deux non-linéarités en cascade |
| 4 | La dynamique lente est nécessaire seulement sur certains appareils | bursts et transitions de niveau avant ajout de TFiLM |
| 5 | L'antialiasing explique le gap de fidélité | différé : S3 et S4 sont actuellement ex æquo |

Le troisième point est particulièrement concret : le Big Muff mesuré contient deux
étages de clipping à diodes en cascade, tandis que S3 n'en contient qu'un
([description du dispositif, DAFx-19](https://dafx.de/paper-archive/2019/DAFx2019_paper_43.pdf)).
Les méthodes block-oriented recommandent par ailleurs une identification en étapes
pour éviter que filtres et non-linéarité s'échangent leurs rôles
([Eichas et al., DAFx-15](https://dafx.de/paper-archive/2015/DAFx-15_submission_21.pdf)).

## Campagne R1

### 1. Gate de compétence

Reproduire le LSTM-64 Big Muff officiel à 44,1 kHz sur les splits complets. Commencer
par la seed 0, puis compléter avec les seeds 1 et 2 seulement si
`ESR_test ≤ 0.15`. Le papier rapporte 0,041 ; la borne plus large tolère les
différences de versions.

Si ce gate échoue, arrêter toute invention architecturale et auditer alignement,
TBPTT, loss et checkpointing.

### 2. Factoriel loss × budget

Sur Fulltone et Big Muff, seed 0, entraîner A2 et S3 jusqu'à 5 000 updates avec
checkpoints 200/1 000/5 000 :

- loss M4 : `MSE + 0.0005 MR-STFT` ;
- loss de référence : `ESR préaccentué + DC`.

Huit trajectoires. Suivre ESR, gain, énergie de sortie, gradients par bloc et énergie
résiduelle. Continuer seulement si S3 Big Muff atteint `gain_error > -0.5` et réduit
son ESR M4 d'au moins 25 %.

### 3. Résidu long et entraînement étagé

Comparer RF 31 à RF ≈ 2 047 (43 ms) au moyen de convolutions causales dilatées
séparables. Entraîner successivement : réponse linéaire/gain, spline, résidu sur
l'erreur cœur gelé, puis fine-tuning conjoint avec une rampe de pénalité résiduelle.

Gate : fermer ≥50 % du gap S3→A2 sur Fulltone et ≥25 % sur Big Muff, sans dépasser
le coût théorique d'A2. TFiLM est crédible pour la modulation multi-échelle, mais
ne sera ajouté que si les diagnostics temporels le réclament
([Comunità et al., ICASSP-23](https://mcomunita.github.io/files/comunita2023gcntfilm-paper.pdf)).

### 4. Cœur cascade conditionnel

Tester d'abord sur un système synthétique à deux clippers séparés par un filtre :

```text
H0 → spline1 → H1 → spline2 → H2 → + résidu RF43ms
```

Le cascade doit réduire d'au moins 50 % l'ESR du cœur mono-spline avant tout run
physique. Sur Big Muff, exiger ≥30 % d'amélioration ESR et une division par deux de
l'erreur de gain. Sinon, arrêter cette famille. Une architecture DDSP multi-étages a
déjà montré qu'une structure physique plus riche peut rester très économique, mais
sur un autre amplificateur : c'est une piste, pas une preuve transférée
([Yeh et al., arXiv 2024](https://arxiv.org/abs/2408.11405)).

## Ce que nous ne faisons pas encore

- Pas de nouvelle grande matrice ni de données externes.
- Pas d'ADAA, x4 ou nouveau SSM avant le gate de fidélité.
- Pas de branche lente sans hystérésis mesurée.
- Pas d'argument H3 avec une sortie faible : Sato et Smith montrent explicitement
  qu'un ASR bas peut provenir d'un modèle quasi silencieux
  ([DAFx-25](https://dafx.de/paper-archive/2025/DAFx25_paper_50.pdf)).

## Position scientifique

La meilleure question de papier n'est plus immédiatement « FSSR bat-il A2 ? »,
mais :

> Une identification étagée, un horizon résiduel intermédiaire et une profondeur
> non linéaire adaptée au dispositif ferment-ils le gap entre gray-box compactes et
> modèles noirs ?

Trois résultats seraient publiables et honnêtes : suppression du collapse par la
procédure d'entraînement, gain causal dû à l'horizon ou à la cascade, ou confirmation
négative que même ces mécanismes ne ferment pas le gap. Le rapport source et le
registre des preuves sont conservés dans `reports/steering/`.
