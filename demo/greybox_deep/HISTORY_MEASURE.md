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
