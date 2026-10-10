# Gain conditionné par l'historique : définition et validation préinscrites

Voie produit (D-PRODUCT-001), aucune revendication scientifique. Piste D-0014 (boîte grise
approfondie, « meilleure mesure des effets mémoire »). **Écrit le 2026-10-10, avant tout
chiffre de cette mesure.**

## Pourquoi

Le détecteur E2 (`src/fssr_nam/product/dynamics.py`) a besoin de paliers de niveau.
Les jeux DI + ampli n'en contiennent pas : la VS100R (envoi FX) et le JVM410H (sortie
haut-parleur) n'ont que des notes de guitare. E2 y est donc « non mesurable ».

Les sondes de Volterra n'y suffisent pas non plus : elles échouent sur une saturation dure
(`DIAGNOSIS_greybox.md`, contrôle d'instrument).

Il faut une mesure de la mémoire qui fonctionne sur n'importe quel jeu de notes.

## Définition (`src/fssr_nam/product/history.py`)

Calcul par trame de 10 ms (pas de 5 ms) :
- **a**, niveau présent : puissance de l'entrée sur la trame, en dB ;
- **o**, niveau de sortie : puissance de la sortie sur la trame, en dB ;
- **h_T**, niveau d'historique : puissance de l'entrée sur les T secondes qui finissent 10 ms
  avant la trame, pour T = 30, 100 et 300 ms.

On écarte les trames à plus de 40 dB sous la trame d'entrée la plus forte. On classe ensuite
les trames par tranches de 2 dB de a, et on ne garde que les tranches d'au moins 20 trames.

La pente d'historique γ_T est la pente de régression, à l'intérieur des tranches, de o sur
d = h_T − a. Elle s'exprime en dB de sortie par dB d'excès passé.

Lecture :
- Un système sans mémoire plus longue que 10 ms donne γ ≈ 0, quelle que soit sa courbe
  statique.
- Un gain qui remonte lentement après un passage fort (affaissement, compression lente)
  donne γ < 0.
- On rapporte aussi la part de variance, à l'intérieur des tranches, expliquée par la pente.
- Entre une référence et une sortie évaluée, l'écart est Δγ_T = γ_T(évaluée) − γ_T(référence).

**Correction de la définition (2026-10-10, relecture Codex, après la première validation).**
La version ci-dessus est fausse en général.

- **Le défaut.** Retirer la moyenne par tranche de 2 dB ne fixe pas a, car a varie encore à
  l'intérieur d'une tranche. Or o et d = h − a dépendent tous deux de cette variation : sur un
  bruit blanc, l'identité y = x donnait γ = −0,54, −0,82 et −0,94.
- **La version corrigée** (`history.py`). Dans chaque tranche, on centre o, a, (a − moyenne)²
  et h_T. γ_T est alors le coefficient de h_T dans la régression de o sur les trois termes,
  c'est-à-dire la pente sur l'historique à niveau présent tenu. `explained` est la part de la
  variance laissée par a et a² que h_T retire.
- **Les cas indéfinis** (trop peu de trames, sortie entièrement fonction de a) donnent NaN,
  au lieu de 0 et 1.
- **Deux contrôles d'instrument** sont ajoutés à la validation : l'identité (γ = 0 par
  construction) et tanh(10x) sans filtre.

## Validation sur chaînes synthétiques (`demo/greybox_deep/history_check.py`)

**Entrée** : les boucles DI réelles, `di_test` des 8 prises de test de la VS100R à 48 kHz.

**Chaîne de base S0, sans mémoire longue** :
1. passe-haut du 1er ordre à 100 Hz ;
2. tanh(g·x), avec g tel que la crête de l'entrée sature (g = 10) ;
3. passe-bas du 1er ordre à 5 kHz.

Les variantes ne changent qu'un élément de S0 :
- **S1, affaissement** : g divisé par 1 + k·e(t), où e est l'enveloppe de |x| avec une
  constante de temps de 100 ms. On fixe k pour que la médiane de 1 + k·e vaille 3 dB (S1a)
  puis 6 dB (S1b).
- **S2, baffle** : S0, suivie d'un FIR de 150 ms (bruit décroissant à 20 ms, bande
  80 Hz–5 kHz, comme `diagnosis/probe_metrics/nolly_standin.py`).
- **S3, polarisation qui se déplace** : tanh(g·x + b·e(t)) − tanh(b·e(t)), avec la même
  enveloppe de 100 ms et b fixé pour que la médiane de b·e vaille 0,3.

**Attentes, sans seuil de décision** : ce sont des repères de lecture, rapportés tels quels.
- **S0** : |γ_T| ≤ 0,05 pour les trois longueurs. C'est le plancher dû aux filtres courts et
  à la saturation.
- **S1** :
  - γ_100 et γ_300 < 0 ;
  - |γ| plus grand pour S1b que pour S1a ;
  - γ_100(S1a) − γ_100(S0) ≤ −0,05.
- **S2** : γ s'écarte de S0, attendu > 0, puisque la queue du baffle ajoute de l'énergie
  passée. C'est un faux positif connu, qu'on rapporte comme limite.
- **S3** : |γ − γ(S0)| petit. Un déplacement de polarisation change surtout les harmoniques
  paires, pas le niveau. C'est un angle mort attendu, qu'on rapporte comme limite.
- **Résolution** : dispersion de γ d'une prise à l'autre pour S0. Un écart entre systèmes plus
  petit que cette dispersion n'est pas lu.

**Mémoire propre des appareils**, mesurée après la validation :
- γ_T de la VS100R : 8 prises de test, envoi FX ;
- γ_T du JVM410H : sortie haut-parleur, réglages de test ;
- chaque valeur est rapportée à côté de celle de S0 sur la même entrée.

## Ce que la mesure ne dira pas

- Rien sur une mémoire plus courte que 10 ms, qui est dans la trame.
- Rien sur les harmoniques (voir S3).
- Une pente non nulle sur un appareil avec baffle ne se lit pas comme une mémoire non linéaire
  (voir S2).

## Résultats de la validation (2026-10-10, définition corrigée)

Script `history_check.py`, sortie `history_check.json`. Valeurs de γ_T : médiane
[min, max] sur les 8 prises (sur les 9 réglages pour le JVM).

| Système | γ_30 | γ_100 | γ_300 |
|---|---|---|---|
| identité (contrôle) | 0,000 | 0,000 | 0,000 |
| tanh(10x) seul (contrôle) | −0,001 | +0,009 | +0,013 |
| S0, sans mémoire longue | +0,015 | −0,052 | −0,123 |
| S1a, affaissement 3 dB | −0,145 | −0,246 | −0,354 |
| S1b, affaissement 6 dB | −0,263 | −0,388 | −0,522 |
| S2, baffle 150 ms | +0,648 | +0,424 | +0,337 |
| S3, polarisation qui se déplace | −0,080 | −0,169 | −0,260 |
| VS100R, envoi FX | +0,072 [+0,050 ; +0,099] | +0,044 [−0,023 ; +0,064] | +0,033 [−0,125 ; +0,063] |
| S0 sur l'entrée JVM | +0,052 | +0,075 | +0,107 |
| JVM410H, sortie HP | +0,054 [+0,016 ; +0,073] | +0,103 [+0,073 ; +0,121] | +0,198 [+0,155 ; +0,236] |

Part de variance expliquée par h_T, à niveau tenu :

| | à 100 ms | à 300 ms |
|---|---|---|
| S1b | 0,35 | 0,51 |
| VS100R | au plus 0,040 | au plus 0,048 |
| JVM | au plus 0,021 | au plus 0,063 |

**Lecture face aux attentes écrites :**
- **Contrôles : tenus.**
  - L'identité donne γ = 0 exactement.
  - Une saturation sans filtre donne |γ| ≤ 0,013 : la mesure n'invente pas de mémoire pour
    une courbe statique.
- **S0 : attente non tenue à 100 et 300 ms** (−0,052 et −0,123 pour une borne de 0,05).
  - Le contraste avec tanh seul **établit** que ce plancher vient des filtres de S0.
  - *Supposé*, sur le mécanisme : à niveau présent égal, les trames de déclin, plus graves,
    sont moins excitées après le passe-haut.
  - Le plancher dépend aussi de l'entrée : le même S0 donne −0,123 sur l'entrée VS100R et
    +0,107 sur l'entrée JVM, à 300 ms.
  - Conséquence : γ seul ne se lit pas comme une mémoire. Seul Δγ se lit, entre deux systèmes
    sur la même entrée.
- **S1 : attentes tenues.**
  - Écart à S0 à 100 ms : −0,19 (3 dB) et −0,34 (6 dB) ; à 300 ms, −0,23 et −0,40.
  - |γ| croît avec l'affaissement.
- **S2 : attente tenue.** Faux positif de +0,48 à 100 ms. Sur un appareil avec baffle, γ est
  inutilisable sans compensation.
- **S3 : attente non tenue.** Vu à −0,12 de S0 à 100 ms : le décalage réduit le gain petit
  signal de tanh. La mesure ne le distingue pas d'un affaissement.
- **Résolution : non mesurée.** Les 8 prises VS100R ont la même entrée DI, puisque c'est le
  même fichier source. Leur écart sur S0 (≤ 0,002) n'est que le bruit du bouclage, et le JVM
  n'a qu'une entrée.

**Mémoire propre des appareils :**
- **VS100R** : écart à S0 sur la même entrée de +0,10 à 100 ms.
- **JVM** : écart de +0,03 à 100 ms et de +0,09 à 300 ms.
- **Signe** : les deux écarts sont positifs, de sens opposé à un affaissement.
- **Variance expliquée** : au plus 0,063.

Ces écarts sont du même ordre que la variation du plancher de S0 d'une entrée à l'autre
(de −0,12 à +0,11 à 300 ms). **Aucun affaissement de l'ordre de S1a (3 dB) n'est détecté sur
ces deux sorties.** L'absence de mémoire n'est pas établie pour autant : la référence sans
mémoire propre à chaque appareil est inconnue.

**Usage retenu** : Δγ_T entre l'appareil et un modèle, sur la même entrée
(`history_deviation`, qui rapporte aussi le nombre de trames). Il dit si le modèle reproduit
la dépendance à l'historique de l'appareil, quelle qu'en soit l'origine. On le rapporte à côté
du plancher de S0 et de l'ordre de grandeur de S1a (0,19).

### Première version, invalide (conservée)

Chiffres obtenus avec la définition fautive, avant la relecture Codex. Sur cette entrée de
guitare, ils diffèrent des chiffres corrigés de moins de 0,01 : le défaut pesait surtout
quand h varie peu, comme sur un bruit blanc.

Deux phrases de cette version étaient fausses (relecture Codex) :
- la plage de −0,13 à +0,10 venait du **même** S0 sur deux entrées, pas d'une structure à
  l'autre ;
- le plafond « au plus 0,04 » ne valait qu'à 100 ms.

#### Tableau et lecture de la première version

Script `history_check.py` ; sa sortie de l'époque est remplacée par celle de la version corrigée. Valeurs de γ_T : médiane
[min, max] sur les 8 prises.

| Système | γ_30 | γ_100 | γ_300 |
|---|---|---|---|
| S0, sans mémoire longue | +0,010 | −0,059 | −0,131 |
| S1a, affaissement 3 dB | −0,150 | −0,252 | −0,360 |
| S1b, affaissement 6 dB | −0,267 | −0,393 | −0,527 |
| S2, baffle 150 ms | +0,642 | +0,416 | +0,329 |
| S3, polarisation qui se déplace | −0,085 | −0,176 | −0,267 |
| VS100R, envoi FX | +0,070 [+0,049 ; +0,095] | +0,042 [−0,030 ; +0,061] | +0,030 [−0,131 ; +0,061] |
| S0 sur l'entrée JVM | +0,051 | +0,072 | +0,101 |
| JVM410H, sortie HP (9 réglages) | +0,051 [+0,012 ; +0,078] | +0,102 [+0,066 ; +0,129] | +0,192 [+0,146 ; +0,249] |

Part de variance expliquée par la pente, à 100 ms : 0,35 pour S1b, de 0,000 à 0,038 pour
la VS100R, de 0,005 à 0,021 pour le JVM.

**Lecture face aux attentes écrites :**
- **S0 : attente non tenue à 100 et 300 ms** (−0,059 et −0,131 pour une borne de 0,05).
  - Une chaîne sans mémoire longue ne donne pas γ ≈ 0. Ce plancher dépend de la structure
    (filtre avant saturation) : il vaut +0,072 sur l'entrée JVM, avec le même S0.
  - *Supposé* : effet spectral. À présent égal, les trames de déclin, plus graves, sont moins
    excitées après le passe-haut que les attaques.
  - Conséquence : γ seul ne se lit pas comme une mémoire. Seule une différence sur la même
    entrée se lit (Δγ entre deux systèmes).
- **S1 : attentes tenues.**
  - Écart à S0 à 100 ms : −0,19 (3 dB) et −0,33 (6 dB) ; à 300 ms, −0,23 et −0,40.
  - |γ| croît avec l'affaissement.
  - L'écart vaut 3 à 6 fois le plancher de S0.
- **S2 : attente tenue.** Le baffle donne un faux positif de +0,48 à 100 ms, plus gros que
  l'affaissement de 6 dB. Sur un appareil avec baffle, γ est inutilisable sans compensation.
- **S3 : attente non tenue.** Le déplacement de polarisation est vu, à −0,12 de S0 à
  100 ms : le décalage réduit le gain petit signal de tanh. La mesure le confond avec un
  affaissement, sans pouvoir les distinguer.
- **Résolution : non mesurée.** Les 8 prises VS100R ont la même entrée DI, puisque c'est le
  même fichier source. Leur écart sur S0 (≤ 0,002) n'est que le bruit du bouclage, et le JVM
  n'a qu'une entrée. La dispersion due au matériau joué reste inconnue.

**Mémoire propre des appareils** :
- **VS100R** : écart à S0 sur la même entrée de +0,10 à 100 ms.
- **JVM** : écart de +0,03 à 100 ms et de +0,09 à 300 ms.
- **Signe** : les deux écarts sont positifs, de sens opposé à un affaissement.
- **Variance expliquée** : au plus 0,04.

Ces écarts sont du même ordre que la variation du plancher sans mémoire d'une structure à
l'autre (de −0,13 à +0,10). **Aucun affaissement de l'ordre de S1a (3 dB) n'est détecté sur
ces deux sorties.** On ne peut pas en dire plus. En particulier, l'absence de mémoire n'est
pas établie : la référence sans mémoire de chaque appareil est inconnue.

**Usage retenu** : Δγ_T entre l'appareil et un modèle, sur la même entrée
(`history_deviation`). Il dit si le modèle reproduit la dépendance à l'historique de
l'appareil, quelle qu'en soit l'origine. On le rapporte à côté du plancher de S0 et de
l'ordre de grandeur de S1a.
