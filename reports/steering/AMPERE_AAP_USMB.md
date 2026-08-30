# AMPERE - Apprentissage Multi-cadence pour la Prédiction Économe des Réponses Électroniques

> Brouillon pour l'AAP Recherche USMB, volet 1 - Recherches exploratoires. Cible : six pages maximum, références comprises, dans le modèle DOCX officiel. Les champs entre crochets exigent une décision humaine avant dépôt.

**Nom du projet :** AMPERE - Apprentissage Multi-cadence pour la Prédiction Économe des Réponses Électroniques<br>
**Cas d'étude :** systèmes audio électroniques non linéaires en temps réel<br>
**Coordinateur ou coordinatrice :** [nom, prénom, statut]<br>
**Laboratoire et axe :** [unité et axe scientifique exact]<br>
**Autres laboratoires :** [unités confirmées, ou aucun]<br>
**Demande conjointe :** [OUI / NON]<br>
**Durée :** 12 mois [à confirmer]

## 1. Enjeux scientifiques

Les systèmes électroniques non linéaires combinent génération harmonique quasi instantanée, mémoire audio intermédiaire et dynamiques lentes de compression, polarisation ou alimentation. Leur jumeau numérique doit restituer ces régimes sous contraintes de causalité, latence, stabilité et calcul. L'audio fournit un banc d'essai exigeant : 48 kHz, petits blocs et état continu. AMPERE vise un prototype de recherche pour ce périmètre ; aucun transfert à d'autres systèmes électroniques n'est revendiqué.

Les réseaux récurrents et convolutionnels émulent déjà des effets en temps réel [1-3]. Un champ réceptif court confond des historiques lointains ; l'allonger uniformément augmente calcul et difficulté d'optimisation. Les non-linéarités dépassent Nyquist ; suréchantillonner tout le réseau réduit le repliement mais multiplie son coût [4-6]. GCNTF/TFiLM décime déjà les représentations d'un TCN et les module par LSTM [2]. Les SSM [9], Slimmable NAM [8], l'élagage sparse [10] et Prism [11] traitent mémoire, coût ou bandes, sans isoler les trois cadences sous un même protocole.

La nouveauté testée n'est donc pas FiLM, mais une attribution causale appariée : six descripteurs du signal sec alimentent un observateur sélectif réduit ; ses contrôles modulent un coeur long ; le x2 reste local aux non-linéarités ; un contrôle de même graphe force la modulation lente à zéro. Chaque cadence doit justifier son coût. Les bénéficiaires visés sont chercheurs, PME et industries créatives ; l'impact mesuré porte sur fidélité, mémoire, calcul et reproductibilité, sans allégation énergétique non instrumentée.

Le socle local valide alignement, métriques, causalité, reset et parité par blocs ; un backend x2 a passé un gate synthétique. Vingt-quatre entraînements physiques historiques `INTERNAL_DEV`, non confirmatoires, ont favorisé la baseline ouverte NAM A2 sur deux dispositifs de développement. Trois campagnes se sont closes par `NO-GO` (M0-M6), `NO-GO-COMPETENCE-v2` et `NO-GO-MECHANISM` ; cette dernière n'a promu ni activation exotique, ni reconstruction causale, ni filtre équi-ondulation. L'architecture qualité v3 est `INVALID` après interruption d'un worker. La lignée v1.2 est également `INVALID` : trois trajectoires ont terminé, mais un contrat tuple/list incompatible a invalidé le gate. Les `NO-GO` bornent leurs mécanismes ; les lignées `INVALID` ne soutiennent aucune conclusion d'efficacité et leurs sorties restent inéligibles à la sélection.

Le projet s'inscrit prioritairement dans **Services et Industries du Futur**, par des jumeaux numériques compacts et auditables. [Relier ici AMPERE aux axes publiés de chaque laboratoire et à la stratégie USMB.] Son caractère exploratoire réside dans une hypothèse risquée et réfutable : la séparation des cadences peut améliorer fidélité et coût, mais le chemin lent ou l'îlot x2 peuvent être inutiles. AMPERE est présenté comme un projet autonome de 12 mois [confirmer, ou expliciter son programme d'accueil].

## 2. Objectifs et contributions

**Question centrale.** Une architecture causale répartissant mémoire et non-linéarité entre plusieurs cadences dépasse-t-elle une frontière ouverte et gelée tout en respectant le temps réel ? « Dépasser » signifie ici battre des baselines ouvertes réentraînées sur les mêmes paires, splits, budgets et métriques. La portée dite commerciale est limitée à des critères d'usage mesurables : temps réel, latence, mémoire, déterminisme et robustesse en blocs. Aucune connaissance ni supériorité d'un produit propriétaire n'est revendiquée.

- **O1 - Décomposition temporelle :** isoler la valeur du long contexte, de l'état lent et du x2 local.
- **O2 - Apprentissage robuste :** prévenir les sorties de faible énergie, préserver l'état causal et apparier les comparaisons.
- **O3 - Prototype :** produire une inférence native déterministe sans allocation dans la boucle audio.
- **O4 - Preuve traçable :** préenregistrer les gates, conserver tous les runs et publier résultats positifs, négatifs et invalides.

**H-AMPERE-1 - Valeur multi-cadence.** Le chemin lent améliore d'au moins 5 % l'ESR médiane appariée face au même graphe dont sa modulation est exactement nulle ; la borne inférieure de l'intervalle à 95 % est positive et MAE, log-mel et MR-STFT ne régressent pas de plus de 5 %.

**H-AMPERE-2 - Fidélité confirmatoire bornée.** Une unique baseline globale est choisie sur développement : la configuration ouverte minimisant la moyenne équipondérée des ESR médianes obtenues sur Fulltone et Big Muff ; une égalité exacte est départagée par le RTF p95. Cette même baseline est gelée pour Blackstar et UA1176. Le candidat améliore son ESR médiane d'au moins 10 % face à elle. La borne inférieure à 95 % du bootstrap apparié par dispositif, fichier puis seed est positive ; les métriques secondaires ne régressent pas de plus de 5 %. Le delta `median_known_reference_alias_residual_attenuation_db` candidat moins baseline doit rester au moins nul sur les sondes synthétiques gelées. Aucune réduction d'aliasing du matériel physique n'est inférée. Blackstar apporte un seul fichier test et UA1176 deux : cinq initialisations mesurent la variabilité d'apprentissage, pas une population de sources ni une généralisation large.

**H-AMPERE-3 - Déployabilité.** Sur une plateforme de référence gelée à M1 [CPU, OS, compilateur, ISA, threading et mode énergétique à renseigner], l'erreur blocs/fichier est au plus `2e-5`, la latence au plus 64 échantillons et le RTF p95 inférieur à 1 pour des blocs de 128, sur 30 répétitions intercalées après warm-up.

L'ESR rapporte l'énergie de l'erreur à celle de la cible ; MAE est l'erreur absolue moyenne ; log-mel et MR-STFT mesurent des écarts spectraux. Sur les sondes synthétiques à référence connue, `median_known_reference_alias_residual_attenuation_db = -10 log10(E_alias_residual / E_reference_harmonics)` : une valeur plus élevée est meilleure. Le RTF rapporte temps de calcul et durée audio. Les contributions attendues sont une architecture ablatée, un protocole source-disjoint multi-seed, un prototype natif et un résultat publiable même négatif. La suite visée est [AAP national/international, échéance et consortium à confirmer].

## 3. Méthodologie

**LT1 - Ancre long contexte (M1-M3).** Trois systèmes synthétiques ordonnés, trois seeds, frontières d'épisodes et pré-roll commun ; aucun audio physique. Livrables : protocole gelé et audit. Gate : gain `>-0,2`, corrélation `>0,9` à chaque garde préenregistrée et dérive ESR médiane au plus 5 %. Un échec arrête la chaîne.

**LT2 - Valeur des cadences (M3-M6).** Si LT1 passe, comparer exactement le coeur long seul au même graphe avec état lent, puis tester le x2 local face au même graphe sans x2. Le chemin lent est retenu uniquement si H1 passe. Le x2 exige une médiane `delta median_known_reference_alias_residual_attenuation_db >= 1,0 dB` face au contrôle sans x2, au plus 5 % de régression ESR/MAE/log-mel/MR-STFT, `RTF p95 x2 / contrôle <= 1,50`, un RTF p95 inférieur à 1 et une latence au plus 64 échantillons. Aucune architecture de remplacement n'est choisie après lecture.

**LT3 - Développement et natif (M6-M10).** Après audit des droits, entraîner le candidat, NAM A2 Full/Lite, Slimmable NAM, Wright LSTM64 et NablAFx TCN-TFiLM/S4-TFiLM `small` et `large` sur Fulltone et Big Muff, mêmes splits et trois seeds. Les familles, largeurs et versions sont gelées avant scores. Mesurer tous les coûts dans le même binaire. Livrable : frontière fidélité-coût avec parité, mémoire, latence et RTF ; candidat et unique baseline globale sélectionnée par la règle H-AMPERE-2 sont gelés avant confirmation.

**LT4 - Confirmation et transfert (M10-M12).** Une seule ouverture de Blackstar et UA1176, cinq initialisations, puis audit. Livrables : verdicts H2-H3, prototype, prépublication, paquet reproductible et dossier d'AAP suivant.

[Avant dépôt : associer à chaque lot responsable, ETP, GPU, stockage, droits, plateforme et dépendances.] Le plafond technique est 324 heures GPU et 50 Gio sur trois lignées au plus ; il mesure un budget de calcul, pas une consommation énergétique. L'accès institutionnel et le temps ingénieur doivent être confirmés.

Les épisodes synthétiques et fichiers physiques sont les unités d'observation ; aucune fenêtre n'est traitée comme indépendante. Les splits préservent les fichiers source. Les comparaisons partagent données, seeds et budget. L'incertitude utilise 10 000 réplications appariées : systèmes ou dispositifs sont pondérés également, puis fichiers et seeds sont rééchantillonnés dans chaque strate. Sur le corpus confirmatoire étroit, l'intervalle décrit seulement ces fichiers. Toute valeur non finie, fuite, défaut de provenance ou dérive du protocole invalide le run.

L'amélioration continue suit une séquence gelée : LT1 long seul ; LT2 état lent puis x2 ; LT3 développement et natif ; LT4 confirmation unique. Une étape ne s'ouvre que si ses prérequis passent. Un échec ferme la revendication correspondante et déclenche l'audit ; il n'autorise ni retry, ni choix entre plusieurs corrections. Toute autre hypothèse exige une nouvelle version prospective hors sélection AMPERE.

### Risques et gestion

- **Instabilité ou sortie effondrée :** gardes gain/corrélation et finitude ; arrêt et résultat négatif ou `INVALID`, sans retry.
- **État lent ou x2 redondant :** ablations appariées ; module supprimé et conclusion bornée publiée.
- **Temps réel manqué :** RTF p95 intercalé, parité et mémoire ; revendication prototype rejetée, sans changer de baseline a posteriori.
- **Fuite ou licence insuffisante :** audit avant lecture scientifique ; brut non redistribué, seuls manifestes et procédures autorisés le sont.
- **Confirmation négative ou corpus étroit :** une seule ouverture et limites explicites ; aucun réglage ni généralisation large.
- **Moyens insuffisants :** revue M1 puis à chaque gate ; réduire la portée des mécanismes, jamais les seeds ni la traçabilité.

### Science ouverte et participation

Protocoles, configurations, seeds, code, métriques et décisions seront versionnés ; chaque run, même échoué, restera immuable. Publications en accès ouvert selon la politique USMB. La licence du dépôt pérenne sera décidée avec l'établissement et les ayants droit ; les données brutes ne seront partagées qu'avec autorisation. AMPERE n'intègre pas de sciences participatives dans cette version [confirmer]. Un éventuel test d'écoute utilisera un protocole distinct après les gates objectifs et ne sélectionnera pas le modèle.

## 4. Pilotage et budget

| Unité | Nom | Prénom | Position | Rôle et responsabilités, deux lignes max. |
|---|---|---|---|---|
| [unité] | [nom] | [prénom] | [statut] | Coordination, gates et AAP suivant. |
| [unité] | [nom] | [prénom] | [statut] | Architecture et optimisation. |
| [unité] | [nom] | [prénom] | [statut] | DSP, natif et benchmark. |
| [unité] | [nom] | [prénom] | [statut] | Statistiques et science ouverte. |

Point mensuel ; revues de gate à M3, M6, M10 et M12. La coordination autorise l'ouverture des données ; un second membre vérifie splits, registre et métriques. Sans double validation, le gate reste fermé. [Renseigner titulaires, complémentarité, ETP, instance et décision inter-unités.]

| Type | Justification liée aux lots | Montant |
|---|---|---:|
| Fonctionnement - calcul/stockage | [besoin net après inventaire USMB ; LT1-LT4] | [EUR] |
| Missions | [destination et objectif scientifique] | [EUR] |
| Petits matériels | [besoin d'acquisition ou benchmark démontré] | [EUR] |
| Investissement | [seulement si nécessaire ; devis obligatoire] | [EUR] |
| **Dépenses A** |  | **[EUR]** |
| Cofinancement B | [cofinanceurs confirmés] | [EUR] |
| **Aide A-B** | Maximum 10 000 EUR en fonctionnement seul, 15 000 EUR sinon | **[EUR]** |

## 5. Références

1. Wright, A. et al. (2020), « Real-Time Guitar Amplifier Emulation with Deep Learning », *Applied Sciences* 10(3), 766. DOI: 10.3390/app10030766.
2. Comunità, M. et al. (2023), « Modelling Black-Box Audio Effects with Time-Varying Feature Modulation », *ICASSP*. DOI: 10.1109/ICASSP49357.2023.10097173.
3. Comunità, M. et al. (2025), « Differentiable black-box and gray-box modeling of nonlinear audio effects », *Frontiers in Signal Processing* 5:1580395.
4. Carson, A. et al. (2025), « Resampling Filter Design for Multirate Neural Audio Effect Processing », *IEEE/ACM TASLP* 33, 2163-2174. DOI: 10.1109/TASLPRO.2025.3574878.
5. Bilbao, S. et al. (2017), « Antiderivative Antialiasing for Memoryless Nonlinearities », *IEEE SPL*. DOI: 10.1109/LSP.2017.2675541.
6. Sato, R. et Smith, J. O. III (2025), « Aliasing Reduction in Neural Amp Modeling by Smoothing Activations », *DAFx-25*.
7. Yeh, Y.-T. et al. (2024), « DDSP Guitar Amp: Interpretable Guitar Amplifier Modeling », arXiv:2408.11405.
8. Atkinson, S. (2025), « Slimmable NAM: Neural Amp Models with Adjustable Runtime Computational Cost », arXiv:2511.07470.
9. Simionato, R. et Fasciani, S. (2025), « Comparative Study of State-Based Neural Networks for Virtual Analog Audio Effects Modeling », *J. Audio Speech Music Process.* 2025:30. DOI: 10.1186/s13636-025-00416-3.
10. Sato, R. et Silverstein, E. (2026), « WaveNet-Style Guitar Amplifier Model Pruning for Real-Time iOS Deployment », démonstration DAFx 2026, arXiv:2607.10086.
11. Dal Rí, F. A. et al. (2026), « Prism: Neural Modeling of Multiband Mixtures of Nonlinear Analog Effects », *JAES* 74(7/8), 544-554. DOI: 10.17743/jaes.2026.0278.
