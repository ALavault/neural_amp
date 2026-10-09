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

**Écart au protocole, 2026-10-09, avant la passe de test.** Un essai sur la graine 200 (deux
tons et sinus glissant, −10 dBFS) a montré que les deux tons d'estimation et de lecture
n'avaient pas les mêmes fréquences : la sonde était lue hors de son domaine de validité (scores
Hammerstein au-dessus de d_audio). Correction : la paire de fréquences dépend de graine % 50,
commune à A (graine s) et B (s + 50) ; seules les phases sont indépendantes. Les hyperparamètres
gelés ne changent pas (le développement n'utilisait pas les deux tons). L'essai est conservé
(`calibration_smoke_two_tone_sweep.csv`) et ne compte pas dans les résultats.

## Règles de développement v2 : Volterra–Laguerre et Wiener–Hammerstein (avant leurs chiffres)

Mêmes conditions de développement que v1 (référence statique seule, bruit coloré et sinus
glissant synchronisé, trois niveaux, graines 100–102, lecture sur graine + 50), même critère de
capacité r_ref / RMS(y_ref) sur B.

**Volterra–Laguerre**, ordres 2 et 3 = deux sondes distinctes. Grille : nombre de filtres
{4, 8}, pôle {0,5 ; 0,8 ; 0,9 ; 0,95}, ridge 1e-6. Choix : la plus petite taille dont la meilleure
médiane (sur les pôles) est à moins de 10 % de la meilleure médiane globale ; pour cette taille,
le pôle de plus petite médiane. Puis ridge ∈ {1e-8, 1e-6, 1e-4, 1e-2} : le plus grand qui reste à
moins de 1 % de la médiane au plus petit ridge. L'emboîtement avec Hammerstein est mesuré à part
(part de chaque colonne d'une sonde hors de l'espan de l'autre, sur x_A), jamais supposé.

**Wiener–Hammerstein** : structure fixée sans grille (h_in 16 taps, h_out 64, spline 12 nœuds
sur ±4 RMS, 3 départs, 20 itérations, estimation et lecture de 1 s), à cause de son coût. Au
développement, sur bruit coloré −20 dBFS RMS, graines 100–102 : on rapporte la capacité et la
dispersion entre départs (même données). Aucun choix n'en dépend ; si la dispersion médiane
dépasse la moitié de la part inexpliquée, la sonde est déclarée peu fiable dans cette
configuration avant tout défaut.

## Lecture des résultats d'étalonnage (fixée avant les chiffres de test)

Écrite pendant la passe de test, avant d'en lire un chiffre. Seul l'essai court de la graine 200
(sinus glissant et deux tons à −10 dBFS, deux tons alors erronés) a été vu ; il a motivé l'écart
ci-dessus et la mise en évidence de la sensibilité de la MR-STFT au bruit, rien d'autre.

Pour chaque métrique m (sondes FIR, H3, H5, signature harmonique totale et par ordre, ESR,
MR-STFT, IMD) et chaque condition c (famille × niveau) :
- **plancher** F(m, c) : médiane sur les graines du score « même chaîne, deux bruits à −80 dB » ;
- **rapport de détection** R = médiane sur les graines du score / F(m, c). « Détecté » si R ≥ 10,
  « non détecté » si R < 3, sinon « limite ». Seuil d'instrument, pas d'audibilité ;
- **répétabilité** : écart-type entre graines du log du score ;
- **monotonie observée** : part des (condition, graine) où le score croît strictement avec les trois
  sévérités ; rapportée, jamais imposée (le défaut de redistribution, par exemple, n'a pas de
  raison d'être monotone pour toutes les métriques) ;
- **dépendance au niveau** : rapport des médianes au niveau le plus fort et au plus faible ;
- **dépendance au contenu** : dispersion entre familles, à sévérité et niveau médians ;
- **angle mort** : (défaut, condition) où la métrique est « non détectée » à la sévérité maximale
  alors que d_audio (RMS de la différence) est « détecté » ;
- **redondance** : corrélation de Spearman entre métriques sur toutes les cellules (log des
  scores), et, pour chaque paire, les défauts qui les séparent ;
- **part non représentée et borne** : pour les sondes, médiane de la part non représentée et
  nombre de violations de la borne (attendu : 0).
Le coût de calcul est mesuré à part, sur une condition fixe, par métrique.

Ces lectures disent ce que chaque métrique voit sur ce banc ; aucune validité perceptive n'en
découle sans données d'écoute.

## Paires réelles sans modèle neuronal (phase 6) — attentes avant les chiffres

`real_pairs.py`, protocole gelé v1, 48 kHz pour la VS100R (44,1 kHz pour la JVM).

VS100R, envoi FX : 3 répétitions du réglage central (prises 1, 20, 39), 12 contrôles négatifs
(basses, médiums ou aigus déplacés seuls : l'envoi est pris avant l'étage de tonalité), 8
changements réels (gain ou contour déplacé seul, dont gain 0 en faible signal). Chaque prise est
appariée à la prise centrale la plus proche dans le temps. Entrée commune : le bouclage de la
prise de référence ; la prise évaluée est décalée d'un bloc (bouclage ET envoi) par le retard
entier qui aligne son bouclage, et le désaccord d'entrée restant est rapporté. Section exclue si
ce retard change à l'intérieur (glissement). Plancher réel de chaque score : médiane des trois
répétitions.

Attentes (descriptives) :
- contrôles négatifs au niveau des répétitions (rapport au plancher réel < 3) pour toutes les
  métriques ; une métrique qui les « détecte » lit la dérive ou l'acquisition, pas l'ampli ;
- changements de gain et de contour au-dessus du plancher réel (rapport ≥ 10) ;
- le désaccord d'entrée entre prises fixe une limite : si sa contribution approche le plancher,
  la borne et les scores de la paire sont signalés « entrée non commune ».

JVM410H : échelles d'un seul bouton autour du centre (gain, basses, médiums, aigus), sorties
préampli et haut-parleur. A et B sont deux segments disjoints du même fichier d'entrée sans
frontières de sources documentées : l'indépendance A/B n'est pas établie et c'est écrit avec
chaque chiffre. Pas de répétition : aucun plancher réel, seulement l'ordre des scores le long de
chaque échelle (monotonie observée, non imposée).

**Écart de lecture, avant les chiffres de test.** En essayant les figures sur l'essai court, il est
apparu que l'ESR et l'erreur d'intermodulation sont des rapports d'énergie : leur rapport au
plancher est le carré de celui d'une métrique en amplitude, et un même seuil (10) n'y voudrait pas
dire la même chose. Elles sont lues en amplitude (racine carrée) dans `summarize.py`. Les sondes,
d_audio et la signature harmonique sont déjà en amplitude ; la MR-STFT (convergence spectrale +
écart de log-magnitude) n'a pas d'échelle commune avec les autres et reste telle quelle, ce qui
est rapporté avec ses chiffres.

**Écart de procédure, paires réelles, avant tout score.** Premier passage (`real_pairs_vs100r.csv`
initial, 2 paires sur 23 traitées, scores non lus) : 21 paires exclues pour « glissement ». Les
retards et désaccords d'entrée seuls ont été examinés. Causes : retard entier estimé après
décimation (les prises diffèrent de quelques échantillons à 192 kHz, d'où un décalage
fractionnaire à 48 kHz et un désaccord de −6 dB) ; glissement « détecté » en comparant le premier
et le dernier quart d'un sinus glissant, dont les contenus fréquentiels diffèrent. Même aligné à
192 kHz, le bouclage diffère de −42 à −46 dB entre les prises 1 et 39 (même réglage, aucun
décalage) : c'est un plancher de l'entrée réelle, à rapporter.
Nouvelle procédure : chaque sonde est ajustée sur sa PROPRE entrée enregistrée (bouclage de sa
prise, synchrone de son envoi) et les deux sont lues sur le bouclage de la prise de référence ;
le score de sonde ne dépend plus de l'alignement entre prises (test
`test_own_input_fit_removes_an_input_mismatch`). Les comparaisons directes d'enregistrements
(d_audio, borne, ESR, MR-STFT) restent sur l'entrée alignée à 192 kHz et sont marquées
`input_common` seulement si le retard est le même aux deux bouts et le désaccord ≤ −40 dB.
Signature harmonique : chaque sortie déconvoluée par sa propre entrée.

## Protocole gelé v2 (Volterra–Laguerre, Wiener–Hammerstein) — 2026-10-09

Résultat de `dev_select_v2.py` (`dev_select_v2.json`), règles appliquées telles qu'écrites :
- **Volterra–Laguerre ordre 2** : 8 filtres, pôle 0,8, ridge 1e-4 ; capacité médiane 0,220.
- **Volterra–Laguerre ordre 3** : 8 filtres, pôle 0,9, ridge 1e-4 ; capacité médiane 0,119.
  Capacités moins bonnes que Hammerstein 5 (0,062) sur la référence statique : 8 filtres de
  Laguerre couvrent moins de mémoire que 256 taps.
- **Wiener–Hammerstein** : capacité médiane 0,187, dispersion médiane entre départs 0,090 (même
  données) ; sous le seuil « peu fiable avant tout défaut » (0,5 × 0,187 = 0,094), de justesse.
- **Emboîtement mesuré, absent dans les deux sens** (bruit coloré, graine 100, 32 768
  échantillons, moindres carrés sans régularisation) : part de chaque colonne Hammerstein xᵏ
  retardée hors de l'espan Volterra–Laguerre 3 de 0,13 (x, retard 0) à 0,96 (retard 255) ; part
  médiane d'une colonne Volterra hors de l'espan Hammerstein 5 : 0,66 (max 0,94). Un écart de
  score entre ces deux sondes ne se lit donc pas comme une différence de capacité.

Étalonnage v2 : mêmes graines de test, toutes les familles et niveaux sauf la DI, lue au niveau
médian seulement (coût). WH : bruit coloré à −20 dBFS, graines 200–202, 1 s, comme développé.

## Relecture Codex de la campagne (2026-10-09) et correctifs

Relecture en lecture seule de a51da63..fa99355. Points vérifiés et retenus :
- **Wiener–Hammerstein** : la renormalisation de h_in recalculait l'échelle de u et annulait le
  pas de Gauss–Newton accepté ; h_out était ajusté sur tout le support de convolution, les autres
  étapes sur les échantillons retenus. Corrigés (u invariant à la renormalisation ; h_out sur les
  échantillons retenus ; pente d'extrapolation de la spline exacte). Test : le coût ne remonte
  jamais. **Les résultats WH antérieurs (dev v2 et `calibration_test_wh_noise.csv`) sont
  recalculés ; les anciens restent dans l'historique git.**
- **Profils** : lue en amplitude, l'ESR est identique à d_audio ; sa corrélation de profil avec
  les sondes (ρ ≈ 0) n'était que du bruit d'arrondi. Retirée de cette lecture.
- **Figure VS100R** : l'ESR n'y était pas lue en amplitude ; les points à entrée non commune
  n'étaient pas distingués. Corrigé (points creux).
- **Sinus glissants de test identiques à ceux du développement** (durée tirée de graine % 3 :
  graines de test 200, 201, 202 = développement 102, 100, 101). Le développement n'a lu que la
  référence statique, mais la lecture sur sinus glissant ne vaut pas généralisation à des
  réalisations nouvelles. Limite rapportée, non corrigée après coup.
- **AUC des paires réelles** : les comparaisons directes (ESR, MR-STFT) doivent exclure les
  sections à entrée non commune ; recalculées ainsi dans le rapport.
- Cas limites : référence harmonique silencieuse → « non mesurable » ; WH à plusieurs segments
  d'estimation refusé (φ(0) ≠ 0 entretient h_out pendant les zéros) ; estimation plus courte que
  le FIR refusée.
- Chiffres du brouillon de rapport antérieurs à la DI : régénérés dans la version finale.

## Détecteur ad hoc de dynamique d'enveloppe (E2) — préinscription, 2026-10-09

**Pourquoi.** Aucune métrique existante n'isole la dynamique. `envelope_error` réagit autant à
un gain (R ≈ 1,4e4) qu'au relâchement (9,5e3). Sur DI réelle, toutes les sondes ne captent qu'environ
0,6 d'un changement de relâchement. Conçu avec deux relectures du conseiller ; JVM : aucune
dépendance à l'historique sur 0,5 s, le réel restera descriptif.

**Excitation « sauts de niveau »** (`defect_bench`, famille `steps`). Bruit coloré par segments de
1,2 s, alternance fort / faible (le faible 20 dB sous le fort, ±2 dB tirés par graine),
44 segments : 22 sauts fort→faible (relâchement) et 21 sauts faible→fort (attaque, contrôle
interne). Niveau du fort ∈ {−12, −9, −6} dBFS RMS, là où l'enveloppe de la chaîne dynamique
réduit nettement le drive.

**Indice** (`fssr_nam/product/dynamics.py`). Pour chaque saut :
1. trajectoire de gain 20 log10(RMS_5ms(y) / RMS_5ms(x)) de 0 à 1,2 s ;
2. **moyenne sur les sauts de même type**, puis soustraction de la valeur stabilisée (moyenne sur
   1,0–1,2 s), ce qui retire les effets statiques ;
3. indice = RMS sur 10 ms – 0,8 s de l'écart entre les trajectoires moyennes de y_eval et y_ref,
   en dB, sans échelle. Fenêtre de score et fenêtre de stabilisation sont disjointes ;
4. un indice « relâchement » (sauts fort→faible) et un indice « attaque » (faible→fort).

**Banc.** Les 13 défauts sont appliqués comme variantes de la **chaîne dynamique** (sag 2,
relâchement 50 ms) : la question est de savoir si l'indice réagit à des changements statiques d'un
système qui a une dynamique. Plancher : même chaîne, deux bruits à −80 dB. Graines 100–102
(contrôle d'implémentation seulement) puis 200–202 (lecture).

**Co-répondants légitimes, déclarés d'avance** : le préfiltre (l'enveloppe suit u = passe-haut(x))
et l'écho (énergie retardée après le saut). Ils sont exclus de la spécificité, mais rapportés.

**Critères** (graines 200–202, médiane sur les graines, par niveau) :
- **monotonie** : indice de relâchement strictement croissant de 50 (plancher) à 60, 100 et 200 ms,
  à au moins 2 niveaux sur 3 ;
- **sensibilité** : relâchement 60 ms ≥ 3 × l'indice du plancher, à au moins 2 niveaux sur 3 ;
- **spécificité** : relâchement 200 ms ≥ 3 × le plus grand indice de relâchement des autres
  défauts (co-répondants exclus) à leur sévérité maximale, à au moins 2 niveaux sur 3 ;
- **contrôle interne** : indice d'attaque du défaut de relâchement ≤ 1/3 de son indice de
  relâchement.
Rapportés sans seuil : dispersion entre graines, réponses des co-répondants, indices de tous les
défauts.

**Réel (descriptif).** Pas de saut documenté sur l'envoi FX VS100R (les sauts de gain ToneTwist de
la DI ne sont pas datés ; une détection depuis l'enveloppe d'entrée serait fragile) ; `bursts_recovery`
n'existe que sur une prise micro aux réglages connus. Attente écrite d'avance : aucune lecture
réelle ne sera présentée comme validation.
