# Brainstorming : super-résolution de captures et « couche de sur-échantillonnage »

2026-09-23, à partir d'une idée de l'utilisateur. Skill
`ideation:creative-thinking-for-research` ; exploration puis critique, séparées. Aucune
piste n'est un plan.

Idée de départ : avec une capture à 96/192 kHz et une capture à 48 kHz non décimée d'un
même appareil, peut-on apprendre de quoi extrapoler les captures libres à 48 kHz vers
d'autres fréquences, et offrir une capacité de sur-échantillonnage par une couche
« astucieuse » ?

Antécédent dans le dépôt : `.codex_campaign/PROSPECTIVE_CAPTURE_SUPER_RESOLUTION.md`
(apprendre à reconstruire le 192 kHz depuis ses décimations, puis imputer sur des
captures NAM à 48 kHz ; niveaux `MEASURED_HR` / `RECONSTRUCTED_HR` / `IMPUTED_HR` ;
contrainte de redécimation). Cette note n'a jamais été préenregistrée. Le présent
brainstorming en reprend la première moitié et ajoute la couche de sur-échantillonnage.

## 0. Cadrage : la capture haute fréquence n'apprend pas l'anti-repliement

Une capture à 48 kHz passe par le filtre anti-repliement d'un convertisseur
sur-échantillonné : **elle ne contient pas de repliement**. Le repliement naît dans le
*modèle*, quand ses non-linéarités calculent à 48 kHz. Deux conséquences :

- **Réduire le repliement** peut se faire avec les seules données à 48 kHz : c'est une
  propriété du calcul, pas des données.
- **Une capture à 96/192 kHz** n'apporte que ce que fait l'appareil au-dessus de 24 kHz.
  C'est inaudible en soi ; cela ne compte que si (a) un étage non linéaire en aval
  transforme ces ultrasons en intermodulation audible, ou (b) si l'on veut un modèle
  vraiment fidèle en montée de fréquence.

L'idée contient donc deux problèmes distincts, chacun plus simple une fois séparé.

Second point : **les captures libres à 48 kHz sont surtout des modèles `.nam`, pas de
l'audio** (TONE3000, ToneHunt ; revue du 22/09). Mais un modèle est une capture
inépuisable : on peut le faire jouer sur n'importe quelle excitation. Le « jeu de données
» à extrapoler est l'écosystème NAM lui-même.

## 1. Exploration

### A. Sur-échantillonner un modèle pré-entraîné par interpolation de ses noyaux

La « couche astucieuse » la plus directe, sans réentraînement.

- **Mécanisme.** À 96 kHz, chaque convolution dilatée est remplacée par l'interpolation à
  bande limitée de son noyau appris à 48 kHz : sa réponse entre 0 et 24 kHz est exactement
  celle d'origine. Les non-linéarités calculent à 96 kHz. Pour une entrée limitée à
  24 kHz, chaque signal avant non-linéarité est exactement celui du modèle à 48 kHz,
  sur-échantillonné ; la seule différence est l'absence de repliement.
- **Effet de bord.** Des noyaux à bande limitée coupent au-dessus de 24 kHz à la couche
  suivante : on obtient automatiquement la « non-linéarité filtrée » de StyleGAN3.
- **Autres modèles.** Pour les SSM, l'équivalent est la correction de maintien
  (`brainstorm_sr_param.md`, piste A). NAM-Oversampler multiplie seulement les
  dilatations, ce qui ne préserve pas la réponse en fréquence.
- **Hypothèse fragile.** Le modèle n'a pas appris à compenser son propre repliement
  (`brainstorm_sr_param.md`, piste B). S'il l'a fait, supprimer le repliement change le
  timbre dans la bande audio.
- **Ce qui départage.** Sorties à 48 et 96 kHz, en sinus puis en programme : l'énergie
  inharmonique doit chuter (métriques validées en M1), les harmoniques rester identiques,
  et l'ESR contre la capture après redécimation ne pas monter. S'il monte, B est
  confirmée.

### B. Distillation à harmoniques masquées (sans capture haute fréquence)

- **Mécanisme.** Le modèle à 48 kHz est l'enseignant ; l'élève tourne à 96 kHz.
  Excitations à fréquences connues (sinus, multi-sons) : le repliement de l'enseignant
  tombe sur des fréquences inharmoniques calculables. La perte ne porte que sur les
  harmoniques et l'intermodulation légitime ; les bins repliés sont retirés de la cible.
  L'élève apprend la réponse harmonique sans le repliement de l'enseignant.
- **Hypothèse fragile.** Des sinus ne décrivent pas la dynamique (attaque, compression,
  mémoire). Il faut des excitations riches à fréquences traçables : multi-sons à
  fréquences premières entre elles, sinus glissants lents.
- **Ce qui départage.** Sur le corpus synthétique à 192 kHz (E), l'élève contre la vérité
  non repliée.

### C. Prolonger les harmoniques au-delà de Nyquist (l'idée de départ, reformulée)

- **Mécanisme.** Les captures haute fréquence de quelques appareils servent à apprendre un
  a priori sur la façon dont les spectres harmoniques se poursuivent au-dessus de 24 kHz :
  pente, alternance paire/impaire, dépendance au niveau. Dans B, cet a priori fournit la
  cible là où l'enseignant ne sait rien.
- **Analogie.** Réplication de bande spectrale (SBR) des codecs HE-AAC : bande haute
  reconstruite depuis la bande basse, plus une enveloppe. **Où elle casse** : la SBR vise
  la plausibilité perceptive, pas la phase exacte ; ici la phase compte seulement s'il y a
  un étage non linéaire en aval.
- **Hypothèse fragile.** La prolongation se transfère d'un appareil à l'autre. Pour une
  non-linéarité quasi instantanée y = f(x), les harmoniques hautes sont fixées par la
  courbure de f, que la bande basse contraint en partie ; pour un appareil à mémoire forte,
  rien ne le garantit.
- **Niveau de preuve.** `IMPUTED_HR`, jamais une mesure.

### D. Quand les ultrasons deviennent audibles : les chaînes

Négation de « au-dessus de 24 kHz, c'est inaudible, donc sans importance ».

- **Mécanisme.** Pédale puis ampli : les produits ultrasonores de la pédale, entrés dans
  la non-linéarité de l'ampli, créent de l'intermodulation dans la bande audio. Un modèle
  de pédale à 48 kHz les supprime.
- **Ce qui départage.** Deux systèmes du corpus synthétique à 192 kHz en chaîne : sortie
  finale avec signal intermédiaire complet contre limité à 24 kHz.
- **Où l'idée casse.** Un baffle coupe au-dessus de 5 à 8 kHz : suivi d'une IR de baffle,
  l'effet peut disparaître. Pour une chaîne de pédales en DI, il peut subsister.

### E. Le banc d'essai existe : le corpus synthétique

`configs/data/synthetic.yaml` définit un maître à 192 kHz et ses dérivés à 96 et 48 kHz
pour 8 systèmes (M1) : la vérité à 192 kHz de systèmes non linéaires, sans matériel. Banc
de `RECONSTRUCTED_HR` pour A, B, C et D ; il n'établit rien sur de vrais appareils.

### F. ADAA plaquée sur un modèle pré-entraîné (spéculatif)

Remplacer chaque tanh par sa version antidérivée, qui réduit le repliement à fréquence
fixe. **Où l'idée casse** : les portes du WaveNet multiplient deux branches
(tanh·sigmoïde), ce qui n'est plus une fonction d'une seule entrée ; l'ADAA ajoute un
demi-échantillon de retard, ce qui recrée le problème de maintien.

## 2. Critique

- **A et B sont les plus prometteuses et ne demandent aucune capture haute fréquence.** A
  est un outil de conversion testable tout de suite sur nos modèles A2 et SSM et sur tout
  `.nam` : le plus fort levier produit. Sa fragilité est la compensation du repliement,
  que son test mesure directement. B y résiste mieux, puisque l'élève est réentraîné, au
  prix d'un entraînement par modèle.
- **C est le cœur de l'idée de départ, et la plus risquée** : identifiabilité nulle
  au-dessus de 24 kHz, transfert entre appareils non garanti. Sa valeur dépend de D : si
  les ultrasons ne changent rien en aval, C n'a pas d'usage audible.
- **D décide de l'intérêt de C**, et se teste sans matériel sur E.
- **F** est limitée par les portes du WaveNet.
- **Risque transversal** : ne jamais présenter une sortie imputée comme une mesure.

Enchaînement logique si l'on poursuit : A (couche d'interpolation sur un modèle existant,
repliement et ESR), puis D sur le corpus synthétique. C ne mérite d'être construite que si
D répond oui.

## 3. Vérification bibliographique à faire

- Anti-repliement par fine-tuning, Carson et al. DAFx-25 : l'enseignant est-il
  sur-échantillonné ?
- NAM-Oversampler : multiplication des dilatations seulement, ou interpolation des
  noyaux ?
- ADAA pour les RNN (Mikkonen et Werner, DAFx-25).
- Extension de bande neuronale : NU-Wave 2, AudioSR.
- Intermodulation ultrasonore dans les chaînes d'effets.
