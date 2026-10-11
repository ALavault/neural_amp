# Travaux liés — P3, transfert de fréquence d'échantillonnage des modèles à espace d'états

## Cadrage

- Question : la littérature contient-elle déjà (a) le transfert de fréquence d'un SSM diagonal par
  rééchelonnement de Δ pour un effet audio, (b) l'avance d'un demi-échantillon figée par le ZOH dans
  un modèle entraîné, (c) une correction d'entrée à deux coefficients ou une lecture FOH d'un modèle
  entraîné en ZOH ?
- Inclusion : indépendance à la fréquence des effets audio neuronaux ; SSM (S4, S4D, S5, Mamba,
  HiPPO) et leur discrétisation ; changement de résolution ou de fréquence d'un SSM, tous domaines ;
  SSM pour l'audio et les effets ; théorie classique des maintiens, invariance impulsionnelle, retard
  fractionnaire, antirepliement en VA. Toutes années, préprints compris.
- Exclusion : captures scellées et matériel à droits incertains du dépôt (non lus) ; Mamba et SSM
  sélectifs au-delà de la définition de leur discrétisation ; parole et vision sauf pour un résultat
  de changement de résolution d'un SSM.
- Type : vérification ciblée pour un article de 4 pages, pas une revue systématique.
- Convention : [V] = l'affirmation citée a été lue dans le texte primaire le 2026-10-11, avec sa
  localisation ; [NV] = non lue dans le texte primaire (seconde main ou texte inaccessible).
- Les revues internes du 2026-09-22 et du 2026-09-23 (`reports/neural_amp-research/`) ont servi à
  localiser des sources ; chaque affirmation marquée [V] ci-dessous a été relue dans le texte primaire.

## Références

| Clé | Statut | Référence | Ce qu'elle établit (lu) | Rôle pour le dossier |
|---|---|---|---|---|
| carson2024srirnn | [V] | Carson, Wright, Chowdhury, Välimäki, Bilbao, DAFx24 | RNN de la GuitarML Tone Library, base 44,1 kHz : à M fois la fréquence, le système correct remplace le retard d'un échantillon de la boucle d'état par M échantillons (§3.2, éq. 6-7) ; méthodes STN, LIDL, APDL, CIDL ; exact pour M entier ; sur-échantillonnage seulement, la descente est « left to further work » (§2) | Référence principale de l'indépendance à la fréquence des effets neuronaux ; aucun SSM ; le Δ d'un SSM joue le rôle de leur retard M |
| carson2025interp | [V] | Carson, Wright, Bilbao, ICASSP 2025 | Sous-échantillonnage 44,1 → 40,5 kHz par filtre d'extrapolation (Lagrange, minimax) dans la boucle d'état ; « no single filter for undersampling » ne convient à tous les modèles (§V) ; les échecs viennent d'une instabilité autour d'un point fixe, prédite par analyse linéarisée (résumé, §VI) | Seul critère a priori publié de réussite d'un changement de fréquence (RNN) ; à opposer au SSM diagonal stable dont les pôles restent stables quand Δ change (déduction) |
| carson2025resampling | [V] | Carson, Välimäki, Wright, Bilbao, IEEE TASLP 2025 | Compare le rééchantillonnage autour du modèle (FIR, IIR demi-bande, 44,1 ↔ 48 kHz) et l'ajustement du modèle ; conclut que le rééchantillonnage est préférable sauf si une latence et un retard de phase nuls sont requis, et qu'il évite l'instabilité car la boucle du RNN reste inchangée (conclusion) | Solution de référence « rééchantillonner autour du modèle » face au rééchelonnement de Δ sans filtre ni latence |
| chowdhury2022agnostic | [NV] | Chowdhury, billet Medium, 2022 | Non lu : la page renvoie 403 ; cité par carson2024srirnn (§3.2) comme origine de l'interpolation linéaire entre états retardés (LIDL) | Antériorité pratique du RNN agnostique à la fréquence ; à citer via Carson et al. si la page reste inaccessible |
| wilczek2022ode | [V] | Wilczek, Wright, Välimäki, Habets, DAFx20in22 | Neural ODE de circuits de distorsion : pas de sur-échantillonnage nécessaire, fréquence augmentable après entraînement (résumé) ; à 192 kHz le LSTM naïf se dégrade et les ODENet tiennent ; à 22,05 kHz les modèles à temps continu produisent une sortie repliée (§4, Tables 2 et 4) | Précédent « temps continu » pour les effets, montée jusqu'à 192 kHz, sans SSM ni analyse de décalage temporel |
| massi2026fno | [V] | Massi, Mezza, Bernardini, DAFx26 | Opérateur de Fourier neuronal par trames de durée fixe, évaluable à toute densité d'échantillonnage ; les modifications de RNN sont « tailored to upsampling » et ne permettent pas la descente (résumé) ; la vérité de test aux autres fréquences est le signal rééchantillonné (§5) | Travail le plus récent sur l'indépendance à la fréquence en VA ; sans état ni maintien, donc hors de la question du ZOH |
| saito2021sfi | [V] | Saito, Nakamura, Yatabe, Koizumi, Saruwatari, EUSIPCO 2021 | Noyaux convolutifs issus de filtres analogiques latents échantillonnés par invariance impulsionnelle à la fréquence voulue, recalculés une fois quand la fréquence change (§2-3) ; séparation de sources | Équivalent convolutif d'une paramétrisation continue ; l'invariance impulsionnelle n'utilise pas de maintien, donc pas d'avance de ZOH (déduction) |
| gu2020hippo | [V] | Gu, Dao, Ermon, Rudra, Ré, NeurIPS 2020 | HiPPO-LegS est équivariant par dilatation (Prop. 3) et sa récurrence discrète est invariante au pas (§3) ; les expériences discrétisent en bilinéaire et en ZOH (note 2, annexe B.3) | Origine de la lecture des SSM comme systèmes continus discrétisés par un pas Δ |
| gu2022s4 | [V] | Gu, Goel, Ré, ICLR 2022 | S4 se discrétise par la méthode bilinéaire (§2.3, éq. 3) ; sans réentraînement, 96,3 % à 0,5× la fréquence sur Speech Commands 10 en changeant Δ (§4.3, Table 5) ; la légende de la Table 5 demande de lire l'annexe D.5, qui renvoie à la Table 11 de S4D pour citer Speech Commands | Source de la propriété « changer Δ pour changer de fréquence » ; descente seulement, classification |
| gu2022s4d | [V] | Gu, Gupta, Goel, Ré, NeurIPS 2022 | Formules ZOH et bilinéaire d'un SSM diagonal, noyau K = (CB, CAB, …) (§3.1) ; les deux règles ne font pas de différence notable dans la distribution d'entraînement (§5.1, Table 2) | Forme de la couche ZOH dont T1 calcule le premier moment du noyau ; aucune comparaison des règles en transfert de fréquence |
| smith2023s5 | [V] | Smith, Warrington, Linderman, ICLR 2023 | Discrétisation ZOH (§3, éq. 6) ; transfert sans réentraînement en rééchelonnant globalement Δ par le rapport des fréquences (§6.2) ; Speech Commands 35, 16 → 8 kHz : S5 de 96,52 à 94,53 %, S4-LegS de 96,08 à 91,32 %, CNN sous 9 % (Table 2) ; Δt variable pour l'échantillonnage irrégulier (§6.3) | Énoncé canonique de la recette appliquée par le dossier ; descente seulement |
| nguyen2022s4nd | [V] | Nguyen, Goel, Gu et al., NeurIPS 2022 | Changement de résolution en divisant Δ par le rapport des fréquences ; repliement dans le noyau aux autres résolutions, corrigé par un masquage passe-bas des coefficients au-delà d'un seuil (§4.2) | Explication concurrente par repliement ; elle concerne surtout la descente, où la nouvelle Nyquist passe sous des pôles appris (déduction) |
| gu2024mamba | [V] | Gu, Dao, COLM 2024 | Discrétisation ZOH (§2, éq. 4) ; Δ, B et C deviennent des fonctions de l'entrée (sélection, §3) | Le pas dépend de l'entrée : le rééchelonnement par la fréquence ne s'y applique pas directement ; hors périmètre |
| lahoti2026mamba3 | [V] | Lahoti, Li, Chen et al., ICLR 2026 | Discrétisation « exponentielle-trapézoïdale » à deux termes d'entrée par pas pondérés par λt appris (Prop. 1) ; équivaut à une convolution de largeur deux sur l'entrée d'état dans la récurrence (§3.1.2) ; λ = 1/2 redonne le trapèze, λ = 1 l'Euler de Mamba-2 (Remarque 2) ; ne pas imposer λ ≈ 1/2 marche mieux (Remarque 3) ; aucune expérience de changement de fréquence (grep) | Structure la plus proche d'une correction d'entrée à deux coefficients par mode, mais apprise à pas fixe et non dérivée pour un transfert |
| ibrahim2026beyondzoh | [V] | Ibrahim, Liu, Wang, arXiv 2026 | Compare ZOH, FOH (entrée linéaire entre deux échantillons, §3.3, éq. 5-6), bilinéaire, polynomial, maintien d'ordre supérieur et RK4 dans Vision Mamba à résolution fixe ; le FOH y apporte peu ou dégrade (§4) | FOH entraîné dès le départ, sans transfert ni lecture FOH d'un modèle ZOH |
| zhu2025fssm | [V] | Zhu, Zhang, Lu et al., ICASSP 2025 | Maintien du premier ordre (interpolation linéaire entre xn et xn+1) dans le module SSM de MambaIR : récurrence à deux termes d'entrée B1 xn + B2 xn+1 (§III, Fig. 3), borne d'erreur cumulée (annexe A, éq. 12-17) ; super-résolution d'image à pas fixe | FOH appris dans un SSM, de même forme à deux coefficients d'entrée que la correction, mais non causal, entraîné tel quel et sans changement de pas |
| graf2026flowstate | [V] | Graf, Ortner, Woźniak, Pantazi, ICML 2026 | Encodeur S5 discrétisé en ZOH, Δ ajusté aux fréquences non vues (§3.3) ; Prop. B.3 : l'écart d'état entre les pas Δ et Δ′ est borné au premier ordre en max(Δ, Δ′) par l'écart des reconstructions ZOH (annexe B.1) | Analyse formelle la plus proche de T1 : borne d'ordre un, sans identification d'un retard pur ni correction |
| zubic2024event | [V] | Zubić, Gehrig, Scaramuzza, CVPR 2024 | SSM (S4D, S5) à pas appris déployés à des fréquences d'inférence plus hautes qu'à l'entraînement ; perte attribuée au repliement et traitée par masquage des Cn au-delà d'un seuil α, α = 0,5 retenu, ou par pénalité de norme H2 (résumé, §3.3) | Explication concurrente de l'erreur en montée (repliement) que le dossier doit discuter |
| boulanger2026pdm | [V] | Boulanger, Wood, arXiv 2026 | Encodeur SSM (LegT, FouT) appliqué à la parole PCM à 16 kHz et PDM à 1,024 MHz en changeant seulement Δ ; trajectoires de coefficients presque identiques au bruit de quantification près (§III, Fig. 6) | Montée d'un facteur 64 d'un SSM par Δ seul, hors effets audio, sans mesure du décalage temporel |
| bittner2025sedge | [V] | Bittner, Schnöll, Wess, Jantsch, Machine Learning 2025 | SSM diagonal complexe discrétisé en ZOH (éq. 14) ; sous-échantillonnage interne couche par couche à l'inférence ; régularisation RZOH contre l'erreur de discrétisation et de repliement (§3.2) ; tests à 16, 8 et 4 kHz | Changement de pas d'un SSM audio, en descente seulement |
| dallinger2025pianossm | [V] | Dallinger, Bittner, Schnöll, Wess, Jantsch, DAFx25 | SSM diagonal discrétisé en ZOH (§3.1) ; synthèse à une fréquence inférieure à celle d'entraînement ; le nombre de valeurs propres repliées corrèle avec l'erreur, jugée acceptable (§5.2, Table 3) | Seul SSM audio re-discrétisé à une autre fréquence ; descente seulement, explication par repliement |
| yin2024drc | [V] | Yin, Cheng, Steinmetz et al., arXiv 2024 | Compresseur modélisé avec des couches S4 ; une seule mention de la fréquence (44,1 kHz, grep) ; aucune expérience à une autre fréquence | SSM pour effet audio sans transfert de fréquence |
| simionato2025ssm | [V] | Simionato, Fasciani, EURASIP JASMP 2025 | Compare LSTM, ED, LRU, S4D et S6 pour des effets VA (§3) ; une seule occurrence de « sampling rate », pour les données à 48 kHz (grep) ; aucune expérience inter-fréquences | Confirme qu'aucun SSM d'effet audio n'a été testé à une autre fréquence dans ces travaux |
| comunita2025frontiers | [V] | Comunità, Steinmetz, Reiss, Frontiers in Signal Processing 2025 | Définit les bases S4 (matrices diagonales) et S4-TF (S4 avec conditionnement TFiLM) et leurs configurations (§3, tableau des modèles) ; lu dans le préprint arXiv 2502.14405 | Origine de l'architecture S4-TF sur laquelle la correction ne se transfère pas |
| esqueda2025lru | [V] | Esqueda, Murai, DAFx25 | Réseau de LRU réelles et d'étages non linéaires avec ADAA d'ordre un ; cette ADAA introduit un retard d'un demi-échantillon compensé dans la branche directe par la moyenne de deux points (§4) ; mesures à 96 kHz (§5) | Précédent audio d'une compensation de demi-échantillon par un filtre à deux coefficients dans un réseau à état |
| parker2016adaa | [V] | Parker, Zavalishin, Le Bivic, DAFx-16 | Convolution en temps continu du signal interpolé : à faible niveau, le noyau rectangulaire donne (xn + xn−1)/2, un retard d'un demi-échantillon, et le noyau triangulaire un retard d'un échantillon (§4, éq. 17-18) ; le point effectif d'un noyau est son premier moment normalisé (annexe A, éq. 37-39) | Argument « premier moment du noyau = retard » déjà publié en VA ; T1 l'applique au noyau ZOH d'un SSM entraîné |
| jackson2000impulse | [V] | Jackson, IEEE Signal Processing Letters 2000 | Résumé de l'auteur seulement : si la réponse impulsionnelle causale est discontinue en t = 0, le premier échantillon de l'invariance impulsionnelle doit valoir la moitié de la discontinuité | Correction classique d'un demi-poids à l'origine, voisine de l'écart entre ZOH et FOH (rapprochement, déduction) |
| laakso1996splitting | [NV] | Laakso, Välimäki, Karjalainen, Laine, IEEE Signal Processing Magazine 1996 | Non lu ; cité par carson2024srirnn et carson2025interp pour les filtres de retard fractionnaire (Lagrange, passe-tout) | Référence des filtres de retard fractionnaire auxquels comparer la correction à deux coefficients |
| franklin1998digital | [NV] | Franklin, Powell, Workman, Addison-Wesley 1998 | Non lu (aucun exemplaire consultable) ; cité pour les équivalents discrets par maintien (ZOH, maintien triangulaire non causal) | Source classique à lire avant d'affirmer « le ZOH équivaut à un retard d'un demi-pas » |
| astrom2011computer | [NV] | Åström, Wittenmark, Dover 2011 (3e éd.) | Non lu ; cité par graf2026flowstate pour l'exactitude du ZOH sur une entrée en escalier ; cité pour l'échantillonnage par maintiens d'ordre zéro et un | Source classique de la discrétisation par maintien, à lire avant citation |

## Synthèse par question

- (a) Rééchelonner Δ pour changer de fréquence : établi pour S4, S5, S4ND, FlowState, l'encodeur
  PDM et les SSM audio de TU Wien (gu2022s4, smith2023s5, nguyen2022s4nd, graf2026flowstate,
  boulanger2026pdm, bittner2025sedge, dallinger2025pianossm), le plus souvent en descente, et hors
  effets audio ; les SSM d'effets audio lus ne le testent pas (yin2024drc, simionato2025ssm).
- (b) Les effets neuronaux à fréquence variable sont traités par modification de RNN ou par
  rééchantillonnage (carson2024srirnn, carson2025interp, carson2025resampling), par neural ODE
  (wilczek2022ode) ou par opérateur (massi2026fno), jamais par un SSM.
- (c) Explication de l'erreur de transfert dans la littérature SSM : repliement du noyau ou des
  valeurs propres (nguyen2022s4nd, zubic2024event, dallinger2025pianossm) ou erreur de
  reconstruction ZOH bornée au premier ordre (graf2026flowstate) ; aucune source lue ne l'identifie
  à un retard d'un demi-échantillon.
- (d) Discrétisations au-delà du ZOH dans les SSM : bilinéaire (gu2022s4, gu2022s4d), trapèze
  généralisé à λ appris (lahoti2026mamba3), FOH appris (ibrahim2026beyondzoh, zhu2025fssm), toutes
  à pas fixe.
- (e) Le lien « noyau rectangulaire = demi-échantillon, premier moment = retard effectif » est
  publié en antirepliement VA (parker2016adaa, esqueda2025lru) ; la version contrôle classique
  (franklin1998digital, astrom2011computer) n'a pas été lue.

## Travaux les plus proches et risque de nouveauté

- Une recherche bornée ne prouve pas la nouveauté : les bases, requêtes et dates sont listées
  ci-dessous ; « non trouvé » signifie non trouvé dans ces recherches, pas inexistant.
- Risque élevé, mécanisme déjà publié hors SSM : parker2016adaa (§4, annexe A) dérive qu'un noyau
  rectangulaire équivaut à un retard d'un demi-échantillon et que le premier moment du noyau fixe
  l'instant effectif ; esqueda2025lru (§4) compense ce demi-échantillon par une moyenne de deux
  points dans un réseau à état. Le dossier ne peut pas revendiquer la propriété du noyau rectangulaire,
  seulement son application au ZOH figé d'un SSM entraîné puis rééchelonné.
- Risque élevé, propriété classique non lue : l'équivalence approchée « ZOH ≈ retard d'un demi-pas »
  et le maintien triangulaire (FOH non causal) relèvent des manuels de commande numérique
  (franklin1998digital, astrom2011computer, [NV]) ; il faut les lire avant de formuler T1 comme un
  résultat, et présenter T1 comme une application.
- Analyse formelle la plus proche : graf2026flowstate, Prop. B.3, borne au premier ordre l'écart
  entre deux pas d'un S5 discrétisé en ZOH ; pas d'interprétation en retard pur, pas de correction,
  pas d'audio. Un relecteur peut voir T1 comme un raffinement de cette borne.
- Structure la plus proche de la correction à deux coefficients : lahoti2026mamba3 (Prop. 1,
  §3.1.2), convolution de largeur deux sur l'entrée d'état dans la récurrence, à λ appris et à pas
  fixe ; à distinguer explicitement (correction a posteriori, dérivée de l'accord du gain continu et
  de la dérivée en 0 Hz, pour un transfert de fréquence).
- FOH dans des SSM : ibrahim2026beyondzoh et zhu2025fssm entraînent en FOH dès le départ, à
  résolution fixe ; zhu2025fssm écrit déjà la récurrence FOH avec deux coefficients d'entrée
  (B1 xn + B2 xn+1, forme non causale), donc l'équivalence « deux coefficients = FOH » n'est pas
  nouvelle en soi ; aucune lecture FOH d'un modèle entraîné en ZOH n'a été trouvée.
- Explication concurrente : zubic2024event attribue la perte en montée de fréquence au repliement
  et la traite par masquage ou norme H2 ; nguyen2022s4nd et dallinger2025pianossm relient la perte
  au repliement des valeurs propres, en descente. Le dossier doit montrer pourquoi, en montée, le
  repliement des pôles appris n'explique pas son erreur.
- Non trouvé (bases et requêtes ci-dessous) : un travail qui dérive l'avance d'un demi-échantillon
  figée par le ZOH dans un SSM entraîné et révélée par le rééchelonnement de Δ ; une lecture FOH
  d'un point de contrôle entraîné en ZOH ; une correction d'entrée à deux coefficients par mode pour
  un transfert de fréquence ; un SSM d'effet audio testé à une autre fréquence (yin2024drc et
  simionato2025ssm ne le font pas) ; un échec de transfert propre au conditionnement TFiLM.

## Recherches effectuées

- Date de toutes les recherches : 2026-10-11.
- API arXiv (`export.arxiv.org`, tri par date, nombre de résultats entre crochets) :
  `abs:"zero-order hold" AND abs:"state space"` [5] ; `abs:"first-order hold" AND abs:"state space"`
  [2] ; `abs:"sample rate" AND abs:"state space" AND abs:audio` [2] ;
  `abs:"sampling rate" AND abs:"state space model"` [17] ; `abs:"sample rate independent"` [3] ;
  `abs:"sample-rate" AND abs:"audio effect"` [5] ; `abs:"half-sample" AND abs:delay AND abs:neural`
  [0] ; `abs:discretization AND abs:"state space" AND abs:resolution AND abs:"zero-shot"` [2] ;
  `abs:"S4" AND abs:"sampling rate"` [3] ;
  `abs:"sampling frequency" AND abs:"state space" AND abs:neural` [1] ;
  `abs:"virtual analog" AND abs:"state space"` [1] ; `abs:"triangle hold"` [1] ;
  `abs:"first-order hold" AND (abs:neural OR abs:learning)` [1] ;
  `abs:"sample rate" AND (abs:amplifier OR abs:distortion) AND abs:neural` [7] ;
  `abs:"state space" AND abs:"audio effect"` [2] ;
  `abs:"state space" AND (abs:guitar OR abs:amplifier) AND abs:neural` [7] ;
  `abs:"sampling rate" AND abs:"virtual analog"` [1] ; `abs:"zero-order hold" AND abs:"sampling rate"`
  [7] ; `au:Chowdhury_J AND (abs:"sample rate" OR abs:"sample-rate")` [0] ;
  `au:Clarke AND au:Chowdhury` [29, aucun en audio] ;
  `abs:TFiLM AND (abs:"sample rate" OR abs:"sampling rate")` [0] ;
  `abs:"feature-wise" AND abs:"state space" AND abs:"sampling rate"` [0].
- Retenus depuis arXiv : 2508.05287 (FlowState), 2509.08458 (FSSM), 2604.20606, 2608.28472 et
  les travaux déjà connus ; écartés sur résumé : 2609.07294 (plasticité, ZOH piloté par l'incrément
  de déformation), 2410.21527 (regroupement de séries irrégulières), 2610.01172 (transfert de taux
  d'apprentissage), 2607.21000 (SSM discret natif).
- Semantic Scholar (`api.semanticscholar.org`) : trois requêtes, réponse 429 à chaque fois ;
  chaînage avant non fait par cette base.
- OpenAlex : recherches par titre des trois articles de Carson et al. ; chaînage avant des
  citations de l'article ICASSP 2025 (1 citant, la version TASLP), de l'article TASLP (3 citants,
  dont un article ICASSP 2026 de séparation de sources sous écart de fréquence, non retenu) et des
  versions arXiv (0).
- Crossref (`api.crossref.org`) : sept recherches bibliographiques pour les DOI de S4ND, HiPPO,
  Zubić et al., Laakso et al., Jackson, Bilbao et al. (ADAA) et Saito et al. ; enregistrements CSL
  de doi.org pour chaque DOI de `refs.bib`.
- Archive DAFx (`dafx.de/paper-archive/search`, première page de 10 titres) : « Piano-SSM »,
  « Neural Ordinary Differential Equations », « Fourier Neural Operators », « continuous-time
  convolution », « sample rate independent », « state space model », « sampling rate neural »,
  « zero-order hold », « first-order hold » ; trois PDF ouverts depuis « first-order hold »
  (DAFx20in21 articles 3 et 23, DAFx23 article 37) puis écartés, le mot « hold » n'y désignant pas
  un maintien.
- OpenReview, recherche par titre : Mamba-3 (ICLR 2026), FlowState (ICML 2026), S5 (ICLR 2023),
  Mamba (COLM 2024) ; la lecture directe des notes est refusée (défi 403).
- WebSearch, pour trouver seulement : SSM d'effet audio et rééchelonnement de Δ ; FOH ou maintien
  triangulaire dans S4 et S5 ; DAFx 2026 et SSM indépendant de la fréquence ; RZOH et SSM
  diagonaux audio ; retard d'un demi-pas du ZOH et maintien triangulaire chez Franklin et al. ;
  « Splitting the unit delay » ; ADAA de Bilbao et al. ; correction de Jackson ; ZOH, demi-échantillon
  et réseaux à temps continu ; FOH ou invariance de rampe dans S4D, S5, Mamba ; Chowdhury et Clarke.
- Manuels : archive.org ne prête que des exemplaires sans recherche plein texte accessible ;
  métadonnées prises dans les notices OpenLibrary par ISBN.
- Tri : environ 500 titres vus, doublons compris ; 31 textes lus au moins par recherche ciblée (19 PDF arXiv,
  9 PDF DAFx, l'article Springer de Bittner et al., la version acceptée de Bilbao et al. sur
  Aaltodoc, le résumé de Jackson) ; 30 entrées retenues, dont 26 [V] et 4 [NV].
- Lus puis non retenus pour tenir le dossier à 30 entrées : Bittner et al., DAFx26 (réverbération à
  plaque par SSM diagonal en ZOH, fréquence fixe) ; Bilbao, Esqueda, Parker, Välimäki, IEEE SPL 2017
  (ADAA d'ordre supérieur, l'ordre un équivaut à parker2016adaa).
- Non cherchés en détail : références de sur-échantillonnage en VA (par exemple Kahles et al.,
  JAES 2019) ; l'origine de TFiLM (Birnbaum et al., NeurIPS 2019) figure dans
  `paper/icassp2027/refs.bib`.
- « Chowdhury & Clarke » : aucun travail commun sur la fréquence d'échantillonnage trouvé (arXiv,
  WebSearch) ; seul le billet de Chowdhury (2022) est cité par Carson et al.
