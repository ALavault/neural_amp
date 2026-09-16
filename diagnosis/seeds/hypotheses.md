# Hypothèses, écrites avant toute mesure (2026-09-16 07:40)

Écart visé : la dispersion inter-seeds de l'ESR de test, CV 56 % pour SSM-WaveNet
(0.0358 / 0.1277 / 0.0832) et 51 % pour son ablation, contre 27 % et 15 % pour
S4-TF-L-16 dans les deux mêmes régimes d'entraînement. Trois graines (42, 43, 44)
par condition, quatre conditions, douze runs.

**H1 — Point d'arrêt (optimisation).** L'early stopping (patience 50 époques sur
la perte de validation, min_delta 0) et le palier de learning rate s'appuient sur
12 segments de validation ; les runs s'arrêtent à des maturités très différentes
(SSM : 4 823 à 7 014 pas). Plus un run va loin, plus ses pôles longs sont
affinés. *Mesure* : corrélation de rang entre `global_step` et l'ESR de test,
dans chaque condition. *Prédiction si vraie* : ρ ≤ −0,8 dans les conditions SSM ;
le run le plus long (seed 42, 7 014 pas) est le meilleur, le plus court (seed 43,
4 823) le pire.

**H2 — Partition train/val (données).** La graine choisit les 12 segments retirés
de l'entraînement sur 130. Elle fixe donc à la fois le signal d'arrêt et la
matière retirée ; une partition qui retire du matériel de faible niveau dégrade
les segments de test calmes, où se concentre l'erreur (2,6 à 4,8 fois l'erreur
des segments forts). *Mesure* : reconstruction de la partition de chaque graine,
statistiques de niveau, corrélation avec l'ESR de la moitié calme du test.
*Prédiction si vraie* : l'ESR de la moitié calme suit une statistique de niveau
de la partition ; la graine 43, dont la validation contient un segment quasi
silencieux, s'arrête tôt et score mal.

**H3 — Tirage des pôles (capacité, initialisation).** SSM-WaveNet a 8×16×4 = 512
pôles complexes contre 8×16×32 = 4 096 pour S4-TF-L-16. Avec huit fois moins de
pôles, le tirage log-uniforme de Δ sur deux décades laisse une grande variabilité
sur l'ensemble des constantes de temps atteignables, qu'un tirage pauvre en modes
lents ne peut pas compenser. *Mesure* : distribution des constantes de temps
apprises τ = −1/ln|p| dans les six checkpoints SSM. *Prédiction si vraie* : la
meilleure graine a nettement plus de pôles à τ > 10 ms que la pire, et l'écart de
dispersion SSM/S4 suit le rapport du nombre de pôles.

**H4 — Définition de la métrique (mesure).** Le protocole moyenne l'ESR segment
par segment sur 12 segments ; les six plus calmes portent 2,6 à 4,8 fois l'erreur
des six plus forts. La moyenne est donc dominée par quelques segments, ce qui
amplifie les différences entre runs. *Mesure* : ESR pondéré par l'énergie
(énergie d'erreur totale / énergie cible totale sur les 60 s) pour les douze
runs. *Prédiction si vraie* : le CV inter-seeds baisse d'au moins 30 % en
relatif sous cette métrique.
