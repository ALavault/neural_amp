# P4 : plan de l'article (sans prose)

- Titre de travail : Sondes d'identification comme métriques de fidélité pour la modélisation d'effets audio.
- Statut : dossier exploratoire, D-PRODUCT-002 ; chiffres dans `RESULTS.md` (rendu de `RESULTS.src.md`), revendications dans `CLAIMS.md`.
- L'auteur rédige l'article ; ce dossier ne contient que faits, chiffres, plan et figures.

## Cible

- Lieu et édition : DAFx27, 30e International Conference on Digital Audio Effects, Crémone (Politecnico di Milano, campus de Crémone), 24 au 27 août 2027 (site dafx27.deib.polimi.it).
- Date limite : fin mars 2027, à confirmer ; appel à communications DAFx27 non publié au 2026-10-11 ; pour mémoire, DAFx26 : 30 mars 2026, retouches jusqu'au 3 avril.
- Format : 8 pages maximum, références comprises selon la consigne du dossier ; le CfP DAFx26 dit « 8 pages maximum » sans préciser pour les références : à confirmer dans le CfP DAFx27 ; gabarit LaTeX DAFx, article soumis prêt à publier.
- Relecture : double aveugle à DAFx26 (option « blind » du gabarit, matériel supplémentaire anonymisé aussi), reconduction à confirmer ; système 1+1/2 de DAFx26 (rebuttal pour les articles limites).
- Politique IA : aucune trouvée dans le CfP DAFx26 ni sur le site DAFx27 : à vérifier dans le CfP DAFx27 ; la rédaction reste à l'auteur.
- Anonymisation : le dépôt est public (nom d'auteur dans l'historique git) ; pas d'URL du dépôt, de hash de commit ni d'identifiant research-memory dans la soumission ; code et chiffres via un dépôt anonymisé ; DAFx26 décourageait les prépublications pendant la relecture.
- Droits des données et de l'audio :
  - VS100R : capture propre du projet (envoi FX, séance du 26/09) ; aucun audio brut dans le dépôt.
  - JVM410H OD1 : jeu ToneTwist sous CC BY-NC (attribution, usage non commercial) ; vérifier les conditions avant tout extrait audio.
  - Nolly : rendus du plugin Neural DSP Archetype: Nolly ; licence du plugin à vérifier avant tout exemple audio ; rendus non encore produits.
  - DI du banc et de la VS100R : sources listées dans `diagnosis/probe_metrics/manifest.json` (dont IDMT, NAM, un enregistrement privé et un extrait de provenance web) ; droits à vérifier avant tout exemple audio.
- Alternative : IEEE WASPAA 2027, Levi (Finlande), 19 au 22 septembre 2027 selon la page de l'IEEE SPS ; date limite non annoncée, vers fin avril 2027 à confirmer ; 4 pages + 1 page de références à confirmer ; double aveugle à confirmer ; politique IEEE sur le texte généré par IA (divulgation) à vérifier.

## Thèse

- Une sonde d'identification (FIR régularisé, Hammerstein 3 et 5, Volterra sur base de Laguerre gelée), ajustée séparément à la sortie de référence et à la sortie évaluée sur une excitation A puis comparée par ses rendus sur une excitation B indépendante, donne un score d'écart comportemental (pseudo-distance, D-0010).
- Le rapport de détection seul ne départage pas les sondes ; leur part captée d_probe / d_audio et leur part non représentée disent de quel type de défaut il s'agit (linéaire, statique, interaction temporelle, hors de toute sonde) : lecture post hoc.
- Volterra–Laguerre voit la non-linéarité préfiltrée et les termes croisés que Hammerstein laisse hors de son espace ; Hammerstein garde la mémoire linéaire longue ; l'emboîtement est absent dans les deux sens : complémentarité, pas classement.
- Aucune sonde ne représente le bruit additif ni un écho hors mémoire : seule l'ESR les voit ; les sondes sont robustes au bruit de mesure.
- Sur captures réelles VS100R, à entrée commune, l'ESR sépare les changements réels aussi bien que les sondes ; les sondes apportent la robustesse aux défauts d'acquisition (post hoc) ; l'attente préinscrite de 10 planchers n'est tenue par aucune métrique.
- La sortie à fort gain de la JVM410H n'est représentable par aucune classe de sonde essayée : les sondes ne servent alors qu'à signaler leur insuffisance ; cause ouverte (Q-0014).
- Un score de dynamique d'enveloppe E2 remplit ses critères préinscrits au banc mais co-répond à l'écho ; pas de validation matérielle.
- Portée : détection et nature d'un écart, sans lien perceptif.

## Plan et budget

| Section | Contenu | Pages | Figures/tableaux |
|---|---|---|---|
| 1 Introduction | écart entre sortie de référence et sortie évaluée pour une même entrée ; limites de l'ESR et de la MR-STFT ; contributions ; portée sans perception | 0,75 | — |
| 2 Score d'écart par sondes | d_probe, d_audio, r_ref, r_eval ; ajustement séparé sur A, lecture sur B ; part captée, part non représentée ; borne triangulaire ; équivalence linéaire et norme de Gram ; emboîtement absent ; règles de refus | 1,25 | Tableau 3 (sondes et protocole gelé) |
| 3 Banc de défauts connus | 13 défauts × 3 sévérités, 6 familles × 3 niveaux, plancher à −80 dB, règle R ≥ 10 / R < 3, développement puis gel, écarts datés | 0,75 | — |
| 4 Résultats au banc | comptes de détection ; part captée et non représentée ; Volterra–Laguerre contre Hammerstein ; signature harmonique par ordre ; répétabilité ; Wiener–Hammerstein et MR-STFT écartés | 1,5 | Figure `figures/captured_share.pdf` ; Tableau 1 |
| 5 Dynamique E2 | sauts de niveau, trajectoires moyennées, quatre critères préinscrits, co-répondants | 0,5 | — |
| 6 Captures réelles | VS100R : contrôles négatifs, attente réfutée, AUC, robustesse, dérive, sinus −3 dBFS et garde ; JVM410H : monotonie, couverture, limite de capacité | 1,5 | Figures `figures/vs100r_separation.pdf` et `figures/jvm_residual.pdf` |
| 7 Coût et usages | coût par paire et par minute d'audio ; ensemble minimal ; usages | 0,5 | Tableau 2 |
| 8 Limites et conclusion | pas de perception ; une séance, un ampli par cas ; sinus de test du développement ; Nolly en attente | 0,25 | — |
| Références | — | 1 | — |

## Figures et tableaux

| Fichier | Générateur | Clés de numbers.json | Contenu |
|---|---|---|---|
| `figures/captured_share.pdf` | `figures.py`, `captured_share()` | `capt_<défaut>_<sonde>` pour défaut ∈ gain, color, prefilter, redistrib, delay, allpass, release, cubic, quad, cross, u7, echo, noise et sonde ∈ fir, h3, h5, vl2, vl3 | part captée par sonde et défaut, sévérité maximale, niveau médian, médiane sur 6 familles |
| `figures/vs100r_separation.pdf` | `figures.py`, `vs100r_separation()` | `vs_auc_h3_all`, `vs_auc_h3_common`, `vs_auc_esr_all`, `vs_auc_esr_common` ; points lus dans `diagnosis/probe_metrics/real_pairs_vs100r.csv` | score / plancher réel par paire pour Hammerstein 3 et ESR, `di_test`, répétitions, contrôles négatifs, changements réels, entrée non commune en creux ; gain 0 (faible signal) exclu, comme dans l'AUC |
| `figures/jvm_residual.pdf` | `figures.py`, `jvm_residual()` | `jvm_h5_res_g0`, `jvm_h5_res_g5`, `jvm_h5_res_g10`, `jvm_h5_1024_res_g5`, `jvm_tanh_res_g5`, `jvm_tanh_res_g10`, `jvm_par_read_min`, `jvm_par_read_max`, `jvm_wh_read` | résidu de lecture r / RMS(y) de chaque classe de sonde essayée sur la JVM410H |
| Tableau 1 (à composer) | — | `det_<métrique>_yes`, `det_<métrique>_lim`, `det_<métrique>_no` | comptes détecté / limite / non détecté à sévérité maximale |
| Tableau 2 (à composer) | — | `cost_<métrique>_s`, `scale_<métrique>_per_min_s`, `scale_wh_per_min_min` | coût par paire et par minute d'audio |
| Tableau 3 (à composer) | — | `dev_<sonde>_taps`, `dev_fir_ridge`, `dev_h_ridge`, `dev_vl2_pole`, `dev_vl3_pole`, `dev_<sonde>_capacity` | protocole gelé et capacité sur la référence statique |

## Théorie

- `THEORY.md` ne contient aucun énoncé pour cette campagne : pas de tableau de statuts ici ; énoncés à formaliser puis à inscrire dans `THEORY.md`.
- Équivalence linéaire en paramètres : fit(y_eval) − fit(y_ref) = fit(y_eval − y_ref) = Δθ, si normalisation, caractéristiques et régularisation ne dépendent que de x_A ; vérifiée par `tests/unit/test_probe_metrics.py` (test_linear_equivalence_and_gram_norm, test_volterra_fit_is_linear_in_the_output).
- Égalité de norme de Gram : d_probe² = Δθᵀ G_B Δθ, G_B Gram des caractéristiques de lecture sur B ; même test.
- Borne triangulaire : |D_audio − D_probe| ≤ R_ref + R_eval sur B ; contrôlée sur chaque ligne (`bound_holds` dans `diagnosis/probe_metrics/calibration_test_*.csv`, `diagnosis/probe_metrics/real_pairs_vs100r.csv`, `diagnosis/probe_metrics/real_pairs_jvm.csv`, compte dans `diagnosis/probe_metrics/report_numbers.json`) ; tests test_identity_scores_zero et test_difference_outside_capacity_is_reported_as_unrepresented.
- Écart des erreurs de fit insuffisant : contre-exemple F/G (même erreur de fit, score porté par la branche d'ordre 3) ; test_same_fit_error_but_different_systems_are_separated.
- Non-emboîtement Hammerstein / Volterra–Laguerre : mesuré numériquement (`diagnosis/probe_metrics/dev_select_v2.json`), sans énoncé général.
- Non-identifiabilité de Wiener–Hammerstein (reparamétrisations de même rendu) : test_wh_reparametrisations_render_identically ; justifie la comparaison par rendus seulement.
- Non-additivité de la part captée et de la part non représentée (aucune orthogonalité hors échantillon) : remarque à démontrer ou à illustrer.

## Travaux liés

- Voir `RELATED.md` et `refs.bib`, préparés à part dans ce dossier.

## Manques avant soumission

- Lecture Nolly (E-0051) : attendre les 42 rendus, puis lancer `diagnosis/probe_metrics/nolly_reading.py` (environ une heure CPU) ; règles déjà figées, pas de nouvelle expérience à concevoir.
- Thèse centrale en préliminaire : part captée, part non représentée, AUC réelles et contrôles JVM sont post hoc ; une confirmation préinscrite (règle écrite avant les chiffres, par exemple H2 de Nolly ou un nouveau banc) est nécessaire pour les présenter comme établies ; nouvelle expérience CPU, coût faible.
- Lien perceptif absent : à annoncer comme limite ; un test d'écoute serait une nouvelle expérience, coût élevé.
- Cause de la limite JVM410H (Q-0014) : prises répétées ou sonde à plusieurs étages ; nouvelle expérience ; captures matérielles impossibles pour l'instant (D-0013) : à traiter en limite.
- E2 sur matériel : excitation à sauts de niveau datés ; nouvelle capture.
- Plancher réel apparié dans le temps : répétitions du centre intercalées ; nouvelle capture.
- Sinus glissants de test distincts du développement : relancer la famille sinus glissant avec de nouvelles réalisations ; nouvelle expérience CPU.
- Réconcilier `diagnosis/probe_metrics/REPORT.md` et ce dossier, sans expérience : relâchement sur DI (plage du rapport plus étroite que celle des 5 sondes au niveau médian) ; « ESR ≥ » du rapport arrondi au-dessus du carré du meilleur résidu ; plage des 90e centiles du rapport sans l'IMD ni l'ordre harmonique 1 ; cubique à faible sévérité vu par H5 sur 5 familles sur 6 seulement ; chiffres de développement Wiener–Hammerstein de `NOTE.md` antérieurs au correctif 7064c38.
- Confronter chaque attente descriptive préinscrite de `NOTE.md` (FIR, Hammerstein, bruit, hors capacité) au résultat ; le rapport ne le fait pas explicitement ; sans expérience.
- Archiver dans un fichier les contrôles Codex cités sans trace (glissement en fin de balayage, faux positifs de la garde sous bruit ajouté) ou les retirer du texte ; petit calcul CPU.
- Intervalles de confiance des AUC réelles (effectifs de 5 contre 10) : calculables depuis les CSV existants ; sans nouvelle mesure.
- Droits audio : vérifier ToneTwist CC BY-NC, licence Nolly et sources DI avant tout exemple sonore.
- Énoncés théoriques à écrire dans `THEORY.md`.
- Confirmer le CfP DAFx27 : date, pages avec ou sans références, politique IA.
