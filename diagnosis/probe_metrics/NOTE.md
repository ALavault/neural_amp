# Sondes d'analyse comme métriques de fidélité : contrat, données, protocole

Voie produit (D-PRODUCT-001), aucune revendication scientifique. Ouvert le 2026-10-09 sur la
mission corrigée par l'utilisateur. **Écrit avant toute mesure** ; les sections marquées
« gelé » ne changent plus après les premiers chiffres de test.

## Objet

Question : quelles différences entre une sortie de référence et une sortie évaluée, pour une
même entrée, chaque sonde détecte-t-elle, avec quelle sensibilité, quelle robustesse et quels
angles morts ?

Les sondes (FIR régularisé, Hammerstein, Volterra–Laguerre réduit, Wiener–Hammerstein) sont des
**instruments d'analyse**, ajustés séparément à (x, y_ref) et à (x, y_eval). Ce ne sont pas des
candidats d'émulation : aucun classement « Hammerstein contre Volterra contre neuronal », aucun
nouvel émulateur, aucune nouvelle comparaison à un modèle neuronal, aucune métrique reposant
sur un réseau appris, pas de SPICE. On parle de « score d'écart » ou de « pseudo-distance » tant
que les propriétés d'une distance ne sont pas démontrées.

## Contrat d'entrée

`fssr_nam.product.fidelity.analyze_pair(x, y_ref, y_eval, fs, protocol, metadata)` :
- `x`, `y_ref`, `y_eval` : dictionnaires nom de segment → tableau, de même longueur par
  segment. Le protocole dit quels segments servent à l'estimation (A) et lesquels à la lecture
  (B). Les segments A sont joints avec `taps` zéros entre eux : chacun part d'un état nul.
- `fs` : fréquence d'échantillonnage commune.
- `metadata` : niveau, réglages, session, calibration, état initial ; transmis tel quel.
- Aucun réalignement, gain ou égalisation n'est appliqué à une sortie. Une correction de la
  chaîne d'acquisition, si elle existe, est appliquée en amont, identique pour toutes les
  sorties d'une même session.

Sortie par segment de lecture : `d_probe`, `d_norm` (échelle RMS(y_ref) avec plancher fixé,
référence quasi silencieuse signalée), `d_audio`, `r_ref`, `r_eval`, part non représentée,
contrôle de la borne |d_audio − d_probe| ≤ r_ref + r_eval, décomposition par branche (diagnostic
conditionné par la base) et par octave, avertissements de capacité ; version du protocole et
de la sonde (empreinte des paramètres).

## Données : ce qui existe, ce qui sert à étalonner, ce qui manque

**Captures réelles disponibles** (inventaire du 2026-10-09) :
- VS100R, envoi de boucle d'effets, 26/09, 192 kHz : 47 prises, x = bouclage de la prise. Par
  prise : silence, bruit rose −30 dBFS RMS (une trame, pas un multisinus), trois sinus glissants
  exponentiels 20 Hz–20 kHz en 5 s à −20/−10/−3 dBFS crête (**non synchronisés** au sens de
  Novak : f0·L = 14,48 non entier ; `product/hammerstein.py` contourne par une matrice de
  mélange mesurée), DI 120 + 30 + 30 s. Sources indépendantes : `di_test` (idmt-bass,
  idmt-gtr4-ib) seulement ; `di_validation` reprend les fichiers de `di_train` à d'autres
  instants et n'en est donc **pas** indépendant. 16 prises ont des glissements de 24 échantillons
  source↔bouclage (bouclage et envoi restent ensemble).
- VS100R micro, 08/10 : 4 prises, une seule aux réglages connus (1044) ; deux tons, salves,
  polarité, paliers de niveau sur ces prises seulement ; ronflette du micro : plancher −20 à −24 dB
  sur la DI.
- ToneTwist (CC BY-NC) : Big Muff, Fulltone (un réglage chacun), JVM410H OD1 (37 réglages,
  PreampOut et SpeakerOut, une entrée sèche commune par partition).

**Données synthétiques** : références DSP et variantes à défauts connus, uniquement pour
étalonner les métriques (section « Banc d'étalonnage »). Les entrées peuvent être des
enregistrements DI réels ; les sorties sont calculées par le DSP, jamais présentées comme des
réponses matérielles.

**Non mesurable avec ces données** (à rapporter comme tel, jamais approché en silence) :
- BLA au sens strict (variances de bruit et de distorsion séparées) sur données réelles : ni
  multisinus périodique, ni réalisations répétées. Sur le réel, la sonde linéaire s'appelle
  « FIR régularisé conditionné par l'excitation ».
- Phase des réponses harmoniques au sens du sinus glissant synchronisé (substitut : matrice de
  mélange mesurée de `product/hammerstein.py`, à signaler).
- Dépendance au niveau au-delà des trois niveaux de sinus glissant ; aucun sinus par paliers.
- Répétabilité d'un jour à l'autre ; prises froid/tiède/chaud ; calibration en volts.
- Tout contenu au-dessus de 20 kHz (sinus glissants) ou 24 kHz (DI).
- Sortie VS100R après l'étage de tonalité (sortie ligne émulée non enregistrée).
Multiplier une entrée et sa sortie enregistrées ne fabrique pas une capture à un autre niveau.

## Principe de mesure

Pour chaque famille de sonde j : ajustement sur A, séparément pour y_ref et y_eval, avec les
mêmes caractéristiques, la même régularisation et le même prétraitement (fonctions de x_A
seulement), puis rendu sur B. Métrique principale : D_probe = RMS(q_eval − q_ref) sur B.
**Interdit comme métrique principale** : |erreur_fit_ref − erreur_fit_eval|.

Contrôles mathématiques implémentés et testés (`tests/unit/test_probe_metrics.py`) :
- sondes linéaires en paramètres : fit(y_eval) − fit(y_ref) = fit(y_eval − y_ref) ;
- D_probe² = Δθᵀ G_B Δθ (G_B Gram des caractéristiques de lecture) ;
- |D_audio − D_probe| ≤ R_ref + R_eval sur B (inégalité triangulaire, sur ces données seulement) ;
- contributions et résidus ne sont pas additionnés comme s'ils étaient orthogonaux.

## Sondes

- **FIR** (`Probe(degree=1)`) : FIR causal à `taps` coefficients plus constante.
- **Hammerstein** (`Probe(degree=K)`) : branches x^k, k = 1..K (pairs et impairs), chacune
  filtrée, mises à l'échelle par leur RMS sur x_A ; K = 3 et 5. Les contributions par branche
  sont des diagnostics conditionnés par la base (monômes), pas une décomposition physique.
- **Réponses harmoniques D_k** (sur sinus glissant, `product/hammerstein.identify`) : ordre
  harmonique k ≠ degré polynomial n ; la conversion D_k ↔ G_n passe par la matrice de mélange.
  Comparaison sur les composantes observables (plancher commun), composantes présentes dans la
  seule sortie évaluée signalées.
- **Volterra–Laguerre réduit** (à venir) : base de Laguerre commune gelée, ordres 2 et 3 avec
  termes croisés. L'emboîtement avec Hammerstein n'est pas supposé : un noyau diagonal
  Hammerstein n'est en général pas dans l'espan des produits de Laguerre ; vérifié
  numériquement avant toute comparaison de scores.
- **Wiener–Hammerstein** (à venir) : h_out ∗ φ(h_in ∗ x), φ paramétrique ou spline ; comparaison
  par rendu sur B uniquement (pas de norme de paramètres) ; variabilité entre initialisations
  mesurée sur les mêmes données, rapportée face à l'écart référence/évalué.

## Banc d'étalonnage (défauts connus)

48 kHz, CPU. Référence : chaîne DSP préfiltre → saturation asymétrique → post-filtre, plus une
variante dynamique à enveloppe pour les défauts d'attaque/relâchement. Défauts, trois
sévérités chacun : gain ; coloration (cloche) ; retard fractionnaire ; passe-tout ; terme
quadratique ou cubique ; redistribution H2/H3 à THD égal à un niveau de référence ; préfiltre
avant saturation ; interaction x[n]·x[n−d] ; constante de relâchement ; bruit additif ; défauts
hors capacité (mémoire au-delà de `taps`, ordre au-delà du degré).

Excitations, chacune en réalisations indépendantes (graines) et à trois niveaux : bruit coloré,
sinus glissant synchronisé, deux tons, multisinus à raies de détection, salves et changements
de niveau, DI réelle (A : sources de `di_train` ; B : sources de `di_test`). Chaque famille est
analysée séparément : une signature conditionnée par une distribution d'entrée n'est pas mélangée
avec une autre.

Comparées aux mêmes défauts : ESR (`metrics/time.py`), MR-STFT de `metrics/spectral.py`
(choisie comme référence parce qu'elle ne dépend pas de la pile d'entraînement ; la variante
auraloss du banc d'entraînement n'est pas utilisée ici), erreurs harmoniques et
d'intermodulation complexes (`metrics/nonlinear.py`) sur sinus et deux tons.

Mesures : sensibilité (score contre sévérité, seuil face au bruit des répétitions),
répétabilité entre graines, monotonie observée (sans l'imposer), dépendance au niveau et au
contenu, coût de calcul, angles morts, redondance entre métriques. Tableau principal :
**métrique × défaut × sévérité × condition d'excitation.**

## Développement puis gel

Hyperparamètres du protocole choisis sur un jeu de développement : graines 100–102 (lecture sur
la graine + 50), bruit coloré et sinus glissant synchronisé, trois niveaux, **référence statique
seule** (aucun défaut lu). Les degrés 3 et 5 sont deux sondes distinctes, gardées toutes deux.
Fixés d'avance : durée non notée 0,05 s, plancher 1e-4, estimation 4 s, lecture 4 s (sinus glissant
: ~4,8 s), 48 kHz.

Règle de choix, écrite avant les chiffres, fondée sur la qualité d'extraction et non sur la
détection des défauts :
1. **taps** ∈ {16, 32, 64, 128, 256}, ridge 1e-6 : la plus petite valeur dont la médiane de
   r_ref / RMS(y_ref) sur B (toutes conditions de développement) est à moins de 10 % (relatif) du
   minimum sur les valeurs de taps, pour chaque sonde.
2. **ridge** ∈ {1e-8, 1e-6, 1e-4, 1e-2}, avec ces taps : la plus grande valeur dont cette médiane
   reste à moins de 1 % (relatif) de sa valeur au plus petit ridge.
3. Rapportés sans seuil : conditionnement de la Gram régularisée et dispersion entre graines.

Le résultat est consigné ici dans une section « Protocole gelé v1 », commitée avant toute
lecture d'un défaut, des graines de test (200 et suivantes) et des autres familles. Le protocole
n'est jamais ajusté sortie par sortie.

## Attentes écrites avant les mesures (descriptives, sans seuil)

- FIR : sensible au gain, à la coloration, au retard, au passe-tout ; voit les changements de
  non-linéarité seulement par leur effet sur le gain équivalent au niveau de l'excitation.
- Hammerstein : sépare les changements statiques de termes pairs et impairs (contre-exemple
  F/G) ; voit peu x[n]·x[n−d] et les constantes de temps d'enveloppe, qui doivent apparaître dans
  la part non représentée.
- Bruit additif : petit score de sonde, grand d_audio (la sonde ne représente pas le bruit).
- Défauts hors capacité : faible score, grande part non représentée ; un faible score n'y vaut
  pas fidélité.

## Protocole gelé v1 (FIR, Hammerstein 3 et 5) — 2026-10-09

Résultat de `dev_select.py` (`dev_select.json`), règle appliquée telle qu'écrite :

| Sonde | taps | ridge | r_ref / RMS(y_ref), médiane dév. | conditionnement (médiane) |
|---|---|---|---|---|
| FIR | 256 | 1e-2 | 0,094 | 3,3e3 |
| Hammerstein 3 | 256 | 1e-4 | 0,062 | 7,0e5 |
| Hammerstein 5 | 256 | 1e-4 | 0,062 | 1,3e6 |

Constats de développement, sans changer la règle :
- **taps au bord de la grille.** La capacité s'améliore encore de 128 à 256 taps (0,13 → 0,094
  pour le FIR) et continuerait sans doute au-delà : la référence a un passe-haut à 120 Hz dont la
  réponse dépasse 256 échantillons (5,3 ms). La grille préinscrite s'arrête à 256 ; elle n'est
  pas étendue après coup.
- **Le degré 5 n'ajoute presque rien** sur ces excitations (0,0619 contre 0,0624) : le résidu de
  la référence vient surtout de la mémoire tronquée, pas de l'ordre.
- Dispersion entre graines de cette part : au plus 0,0009 (absolu).

Restent fixés : durée non notée 0,05 s, plancher 1e-4, avertissement de capacité à 0,3,
octaves de 22 Hz à 11,36 kHz et au-delà, signature harmonique (ordre 5, nfft 8192, 100 Hz–8 kHz,
plancher −80 dB re max |D_1|), estimation et lecture de 4 s (DI : sections entières, A =
`di_train`, B = `di_test`), plancher de répétition : même chaîne mesurée deux fois avec un
bruit indépendant à −80 dB.

Test : graines 200–202 (lecture sur la graine + 50), toutes les familles, trois niveaux, tous
les défauts aux trois sévérités. Volterra–Laguerre et Wiener–Hammerstein auront leur propre
règle de développement, écrite avant leurs chiffres.
