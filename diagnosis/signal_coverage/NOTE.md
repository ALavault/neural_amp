# Où se loge l'erreur des modèles VS100R ? Niveau, mémoire, désaccord des graines (exploration)

Voie produit, régime exploration, 2026-10-06. Chaque affirmation sera marquée **établie** (mesure
citée) ou **supposée**. Origine : brainstorming sur les signaux synthétiques de la prochaine
campagne. Trois observations hors ligne y étaient proposées pour savoir quelle propriété manque au
signal d'entraînement, avant d'ajouter des signaux.

**Décision que la mesure change** : la priorité des sections ajoutées à la source de la séance au
micro. La source contient déjà niveaux, polarité et salves (`demo/capture/valvestate_mic/`) ; la
mesure dira lesquelles analyser d'abord et lesquelles garder pour une campagne plus longue.

## Mesure (prédictions écrites avant la mesure)

Script : `error_structure.py`. Lecture seule, sur CPU.
- Modèles : A48 (`vs100r_gc_film_joint_s925_r2`, `vs100r48_film_s43`, `vs100r48_film_s44`) et
  bras C (`vs100r48fast_film_s42`, `_s43`, `_s44`).
- Données : section di_test (jamais entraînée) des 35 prises hors faible signal, décimée à
  48 kHz, région notée de `product_vs100r_eval.py`.
- Trames de 10 ms : niveau efficace de l'entrée (dBFS), énergie de la cible et de l'erreur. Une
  ESR par groupe de trames = somme des énergies d'erreur / somme des énergies de cible.

**O1, niveau d'entrée rare.** Densité d'entraînement par classe de niveau : part des trames de
10 ms des entrées `trainval` (31 prises, 150 s chacune) dans chaque classe de 5 dB, de −60 à
−10 dBFS. Corrélation de Spearman entre la densité (log) et l'ESR (log) des classes ayant au moins
200 trames de test, par modèle.
- H_niveau : l'erreur se concentre aux niveaux rares en entraînement ; ρ ≤ −0,5 en médiane sur
  les 6 modèles.
- Contraire : |ρ| < 0,3.

**O2, mémoire après un événement fort.** Trame forte : niveau au-dessus du 90ᵉ centile des trames
de sa prise. Pour les trames de niveau moyen (entre les 30ᵉ et 70ᵉ centiles de leur prise), on
classe le temps écoulé depuis la dernière trame forte : < 50 ms, 50–200 ms, 200–1 000 ms et
≥ 1 000 ms.
- H_mémoire : à niveau courant comparable, l'erreur est plus forte juste après un événement fort.
  ESR(50–200 ms) / ESR(≥ 1 000 ms) ≥ 1,5 en médiane sur les modèles.
- Contraire : rapport entre 0,8 et 1,25.

**O3, désaccord des graines.** Par bras (A48, C) et par trame : désaccord = moyenne sur les
graines de l'énergie de (sortie − moyenne des sorties), erreur = moyenne sur les graines de
l'énergie de (sortie − cible), toutes deux divisées par l'énergie de la cible, ce qui retire
l'effet du niveau. Spearman entre désaccord et erreur sur les trames dont la cible dépasse
−50 dBFS, par prise, puis médiane sur les prises.
- H_désaccord : le désaccord prédit l'erreur ; ρ ≥ 0,5 pour les deux bras.
- Contraire : ρ < 0,2.

Ces hypothèses ne s'excluent pas. Lecture : chacune « soutenue », « contraire » ou
« intermédiaire ».

**Limites connues d'avance** :
- L'ESR brute contient le décalage d'environ un échantillon de A48 (`DIAGNOSIS_vs100r_192k.md`),
  qui pèse surtout dans les aigus. Les trois observations comparent des groupes de trames d'un
  même modèle, donc ce biais est commun, mais il peut écraser les différences.
- La guitare lie niveau, attaque et décroissance : O2 ne contrôle que le niveau courant, pas le
  contenu spectral.
