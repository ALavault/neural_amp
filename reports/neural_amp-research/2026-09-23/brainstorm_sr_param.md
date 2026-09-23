# Brainstorming : indépendance à la fréquence (montée) et modèle paramétrique

2026-09-23. Skill `ideation:creative-thinking-for-research` ; exploration puis critique,
séparées. Données : `demo/sr_transfer/RESULTS.md`. Aucune piste n'est un plan.

## 0. Examen du cadrage

« Rendre le modèle agnostique en montée » suppose trois choses discutables :

1. **La bonne cible à 96 kHz serait le modèle à 48 kHz.** L'appareil au-dessus de 24 kHz
   est inconnu : ni la capture (44,1 kHz) ni l'entraînement ne le montrent. La question
   produit est plus étroite : à 96 kHz, la sortie 0–20 kHz égale l'appareil, et si possible
   avec moins de repliement.
2. **La divergence en montée serait un défaut du modèle.** Elle peut être un défaut de la
   *discrétisation* (voir §1).
3. **Un même modèle discret devrait tourner à toutes les fréquences.** Autre lecture : le
   modèle est un objet à temps continu, et la fréquence est un choix de solveur à
   l'inférence.

## 1. Une observation post hoc dans les données

Relue dans les JSON, non prédite. L'écart de cohérence en montée (`delta_vs_around`,
ESR) suit presque exactement le **carré** de τ = 1/48000 − 1/fs :

| rapport | prédit par τ² | SSM s925 / s736 / s688 | S4 s925 / s736 / s688 |
|---|---|---|---|
| ESR 88,2 / ESR 96 | 0,83 | 0,84 / 0,84 / 0,84 | 0,84 / 0,88 / 0,79 |
| ESR 192 / ESR 96 | 2,25 | 2,09 / 2,08 / 2,14 | 2,15 / 2,08 / 1,91 |

La même loi prédit aussi la descente : à 44,1 kHz, τ = −1,8 µs, soit un écart attendu
d'environ 0,002 en ESR, ce qui est cohérent avec un `delta` à 44,1 kHz qui ne coûte rien.
Un ESR proportionnel à τ² correspond à une erreur de phase proportionnelle à τ, donc à une
erreur de discrétisation du premier ordre. Un mécanisme de repliement devrait plutôt
saturer entre 96 et 192 kHz, puisqu'il reste peu d'harmoniques à déplier. Cela désigne le
**maintien d'ordre zéro (ZOH) appris** plutôt que le repliement.

## 2. Exploration

### A. Le maintien est dans ce que le modèle a appris (reformulation)

- **Mécanisme.** Le noyau ZOH vaut K[k] = ∫ h sur [k·dt, (k+1)·dt]. Le modèle a donc
  appris h convolué avec une porte de 1/48000 s : un retard de dt/2 et un léger filtrage
  en sinc. Rééchelonner Δ rétrécit cette porte à 1/fs, ce qui change le système continu
  effectif, pas seulement son échantillonnage.
- **Correction exacte et causale.** À fs, garder la porte de 48 kHz : le coefficient
  d'entrée devient b = (e^{λ·T48} − 1)/λ · dt′/T48, avec T48 = 1/48000, au lieu de
  (e^{λ·dt′} − 1)/λ. Une ligne par mode, identité à 48 kHz, valable aussi pour un rapport
  non entier. C'est ce que le bras `lead` aurait dû être (relecture Codex : exp(λε) sur C
  n'est pas cette opération).
- **Hypothèse fragile.** La porte domine l'écart. Or les non-linéarités ponctuelles entre
  les SSM rendent la correction exacte couche par couche, mais pas nécessairement pour la
  composition des huit couches.
- **Ce qui départage.** L'écart à 96 kHz avant et après correction. La piste prédit une
  division par plus de 5, et la disparition de la loi en τ² dans le rapport 192/96.

### B. Le modèle a appris son propre repliement (explication concurrente)

- **Mécanisme.** À 48 kHz, les harmoniques repliées faisaient partie de la sortie
  entraînée, et les poids les compensent. À 96 kHz, elles ne se replient plus, et la
  compensation devient une erreur.
- **Hypothèse fragile.** La part repliée serait importante pour un Big Muff à 48 kHz.
- **Ce qui départage.** Mêmes bras avec une entrée à −20 dB, où les non-linéarités sont
  presque linéaires. A prédit un écart relatif inchangé, B un écart qui s'effondre. P8 (le
  sur-échantillonnage dégrade la fidélité) s'expliquerait alors par la porte (A), pas par
  la perte d'un repliement utile.

### C. Ne plus apprendre de maintien (négation de « ZOH »)

Entraîner avec une discrétisation d'ordre supérieur (maintien du premier ordre, ou
transformation bilinéaire comme dans le S4 original). L'erreur de phase devient en dt²,
donc l'ESR de transfert en dt⁴. Prédiction : à 96 kHz, l'écart passerait d'environ 0,07 à
moins de 0,005 sans correction à l'inférence. Coût à vérifier : la déformation de
fréquence de la transformation bilinéaire près de Nyquist.

### D. Entraîner sur plusieurs Δ (contrainte déplacée)

Chaque batch est rééchantillonné à une fréquence tirée entre 44,1 et 96 kHz, avec Δ
ajusté, pour forcer un seul système continu. La revue du 22/09 indique que cela n'existe
pas pour les effets. **Où l'idée cesse de marcher** : au-dessus de 24 kHz il n'y a pas de
cible, donc l'entraînement n'apprend rien sur le haut du spectre ; il empêche seulement le
modèle d'exploiter la porte.

### E. StyleGAN3 comme analogie structurelle

Le « texture sticking » de Karras et al. (2021) correspond à notre repliement appris. Leur
remède : sur-échantillonnage, non-linéarité, passe-bas, sous-échantillonnage, ce qui rend
le réseau équivariant à la grille. Transposition : sur-échantillonner ×2 les activations
pendant l'entraînement (`src/fssr_nam/models/oversampling.py` existe).
**Où l'analogie casse** : filtres à phase nulle en image, causaux et retardants en audio ;
un fuzz produit des harmoniques bien au-delà de ×2. Ne répond qu'à B.

### F. Un bouton est un changement d'échelle de temps (bisociation, spéculatif)

- **Mécanisme.** Un potentiomètre change R, donc un pôle −1/(RC), souvent en loi
  logarithmique ; changer fs change Δ·λ. Les deux transforment des pôles.
- **L'idée.** Les réglages modulent directement log Δ ou Re λ par canal, au lieu (ou en
  plus) du FiLM sur les activations. Fréquence et boutons deviennent une seule opération,
  et le modèle reste indépendant de la fréquence par construction (intuition du prototype
  grey-box).
- **Hypothèse fragile.** Les boutons du Marshall (tone stack) agissent surtout sur des
  pôles et zéros linéaires ; ceux d'un fuzz sur un gain avant la non-linéarité.
- **Ce qui départage.** `B100_M000_T000` du Marshall : FiLM sur les activations contre
  FiLM sur les pôles, à paramètres égaux (Miklánek : la structure bat la quantité de
  données sur la combinaison extrême).

### G. Ne pas rendre le modèle agnostique (négation de l'objectif)

Cœur à 48 kHz et rééchantillonnage par l'hôte, au prix de la latence d'un filtre
polyphase (0,235 ms, arXiv 2501.18470). Leçon du 23/09 : en descente, Δ fait mieux que le
rééchantillonnage hors ligne, sans latence. D'où l'hybride : Δ en dessous de 48 kHz,
rééchantillonnage au-dessus, tant que A ou C ne sont pas validées.

### H. La fréquence comme entrée de conditionnement (dialectique, spéculatif)

log fs ajouté aux réglages du FiLM, avec un entraînement multi-fréquence (D). Le modèle a
le droit d'être différent à chaque fréquence, l'inverse de l'indépendance ; c'est le seul
moyen d'absorber un effet que A et C n'expliquent pas. Aucune garantie d'extrapolation à
une fréquence non vue (192 kHz).

### I. Créer la vérité terrain au-dessus de 48 kHz (possible dès maintenant)

La chaîne de capture existe (`demo/capture`, `PROSPECTIVE_CAPTURE_SUPER_RESOLUTION.md`).
Une capture à 192 kHz d'un appareil réel est la seule mesure de *fidélité* en montée ;
sans elle, tout reste relatif au modèle à 48 kHz.

## 3. Critique

- **A** est la piste la plus solide : mécaniste, peu coûteuse (CPU, checkpoints
  existants), falsifiable par un seul bras. Faiblesses : la loi en τ² repose sur trois
  points trouvés après coup, et S4 s688 s'en écarte (1,91 contre 2,25). Un écart de phase
  uniforme du rééchantillonneur de référence donnerait aussi une loi en τ ; il faut
  l'écarter.
- **B** n'est pas exclue et peut coexister avec A. Le test à −20 dB les départage à bas
  coût.
- **C** et **D** demandent du GPU. C ne vaut la peine que si A se confirme (le maintien
  comme cause) ; D si A échoue.
- **F** est la plus originale et relie les deux volets, mais c'est la plus risquée : elle
  suppose des boutons qui agissent sur des constantes de temps.
- **E** est coûteuse et ne répond qu'à B. **H** est un repli. **G** est la réponse de
  production actuelle. **I** est la seule vérité terrain, mais dépend du matériel.

Deux mesures sur CPU suffiraient à trancher entre les explications : A (bras avec la
porte de 48 kHz), puis B (entrée à −20 dB).

## 4. Vérification bibliographique (à faire)

Pour les pistes retenues seulement :

- A et C : discrétisation FOH ou bilinéaire contre ZOH, et correction de maintien, dans
  S4, S5, Piano-SSM et Massi et al. (FNO indépendants de la fréquence, DAFx-26).
- F : conditionnement des pôles par les réglages dans les modèles grey-box (Miklánek,
  tone stack par méthode DK).

Une recherche bornée qui ne trouve rien n'établira pas la nouveauté.
