# P1 — Plan : pièges d'alignement et de polarité dans l'évaluation des modèles neuronaux d'effets audio

<!-- Dossier sans prose (D-PRODUCT-002). L'auteur écrit le texte ; ce plan ne contient que des listes et des tableaux. Les nombres sont dans RESULTS.md et CLAIMS.md, régénérés depuis numbers.json. -->

## Cible

- Venue principale : EUSIPCO 2027 (European Signal Processing Conference, EURASIP), édition et lieu à confirmer.
- Date limite : début février 2027, à confirmer sur l'appel officiel.
- Format : 5 pages au maximum, tout compris (figures, références, annexes) ; gabarit EUSIPCO à confirmer.
- Relecture : simple aveugle (auteurs nommés), à confirmer.
- Politique IA : outils autorisés pour la correction de la langue seulement ; l'auteur rédige toutes les sections (politique à relire au moment de la soumission, à confirmer).
- Anonymisation : non requise en simple aveugle ; le dépôt public peut être cité.
- Droits des données :
  - ToneTwist AFx (Big Muff, F-Fuzz, Marshall) sous CC BY-NC : citer le jeu, ne pas redistribuer l'audio ; chiffres et figures dérivés seulement.
  - Marshall Valvestate VS100R : capture propre de l'auteur (exploratoire, 2026-09-26) ; audio brut hors dépôt ; diffusion éventuelle à décider par l'auteur.
  - Valeurs publiées NablAFx : annexe de Comunità, Steinmetz et Reiss (2025), citée, extraite dans `paper/icassp2027/data/published_bigmuff.json`.
  - Aucune capture scellée, aucune sortie réservée au rapport externe.
- Alternative : ICASSP 2028, 4 pages de contenu plus une 5e page réservée aux références et à l'éthique ; non aveugle ; date limite mi ou fin septembre 2027, à confirmer ; politique LLM de l'IEEE SPS (sections entièrement produites par un LLM interdites).
- Étiquette de tous les résultats : exploratoire, voie produit (D-PRODUCT-002).

## Thèse

- Les comparaisons en ESR brute entre modèles neuronaux d'effets ou d'amplis peuvent être dominées par des erreurs d'alignement que les modèles apprennent mal ou que les données contiennent : latence de rendu (F-Fuzz), retard sous-échantillon (VS100R), inversion de polarité (Big Muff).
- Une co-lecture à retard commun compensé, estimé hors de la section notée, et une garde de polarité rendent ces comparaisons lisibles ; elles ne les corrigent pas.
- Preuve la plus forte : le correctif préinscrit F-Fuzz (latence mesurée, cibles avancées, tous les critères tenus).
- Preuve méthodologique : sur une question préinscrite (longueur d'entraînement), la lecture brute et la co-lecture donnent des étiquettes opposées.
- Preuve théorique : T2, condition sous laquelle une sortie inversée est un minimum local strict le long du gain pour L1 + MR-STFT.
- Ce que la thèse ne dit pas : l'origine du retard VS100R, la nécessité de la garde, une supériorité d'architecture, un effet audible.

## Plan et budget

| Section | Contenu | Pages | Figures/tableaux |
|---|---|---|---|
| 1. Introduction | ESR comme métrique de classement ; trois pièges ; contributions (diagnostic, correctif préinscrit, co-lecture, T2) | 0,5 | — |
| 2. Pièges et métriques | ESR, MR-STFT ; coût d'un retard pur sur l'ESR ; insensibilité de la MR-STFT au signe ; énoncé T2 | 0,75 | — |
| 3. Protocole | Données (ToneTwist, VS100R), modèles (SSM-WaveNet-FiLM, S4-param, S4-TF), instruments : retard optimal par intercorrélation, retard oracle, co-lecture à τ commun sur `di_validation`, garde de polarité | 0,75 | Tableau 1 (runs, graines, n, préinscrit ou post hoc) |
| 4. Latence de rendu F-Fuzz | Retard +4 partout, plancher commun, contrôles Big Muff et Marshall, correctif préinscrit | 0,75 | `figures/ffuzz_lag.pdf` |
| 5. Retard sous-échantillon VS100R | Oracle par prise (post hoc) puis co-lecture préinscrite sur la longueur d'entraînement ; divergence des étiquettes | 1,0 | `figures/delay_curves.pdf`, `figures/train_length_readings.pdf` |
| 6. Polarité | Run inversée, garde, modèles publiés compatibles, T2 | 0,5 | `figures/polarity_published.pdf` ou tableau |
| 7. Limites et recommandations | Une graine pour F-Fuzz, une capture VS100R, origine inconnue, oracle absorbe la phase, pas d'écoute | 0,25 | — |
| Références | Voir `RELATED.md` | 0,5 | — |
| Total | | 5,0 | 3 ou 4 figures, 1 tableau |

## Figures et tableaux

| Fichier | Générateur | Clés et sources | Place |
|---|---|---|---|
| `figures/ffuzz_lag.pdf` | `figures.py:ffuzz_lag` | `ffuzz_thr_esr` ; séries par réglage depuis `diagnosis/ffuzz/segments.json` et `diagnosis/ffuzz/segments_adv4.json` (moyennes de `ffuzz_esr_interior_lag0_*`, `ffuzz_esr_realigned_*`, `ffuzz_adv4_interior_*`) | Retenue, section 4 ; légende : le seuil préinscrit porte sur la moyenne des réglages, pas sur chaque réglage |
| `figures/delay_curves.pdf` | `figures.py:delay_curves` | `trainlen_tau_c_min`, `trainlen_tau_l10_min`, `trainlen_tau_l20_min` (et max) ; courbes `median_esr_curve` de `demo/train_length/eval_train_length_aligned.json` | Retenue, section 5 |
| `figures/train_length_readings.pdf` | `figures.py:train_length_readings` | `trainlen_thr_verdict`, `trainlen_thr_accept_seed`, `trainlen_p_l10_d`, `trainlen_a_l10_d`, `trainlen_p_l20_d`, `trainlen_a_l20_d` | Retenue, section 5 (peut fusionner avec la précédente en deux panneaux) |
| `figures/polarity_published.pdf` | `figures.py:polarity_published` | `polar_two_mean_abs_y`, `polar_inv_esr`, `polar_inv_esr_negated`, `polar_pub_l1_ratio_min` ; `paper/icassp2027/data/published_bigmuff.json` | Réserve, section 6 ; remplaçable par une ligne de tableau |
| `figures/vs100r_oracle.pdf` | `figures.py:vs100r_oracle` | `vs100r_ratio_all_min`, `vs100r_ratio_all_max`, `vs100r_n_takes_ge2` ; `diagnosis/vs100r_192k/measures*.json` | Réserve (post hoc) ; à couper si la place manque |
| Tableau 1 (à composer par l'auteur) | — | `ffuzz_seed`, `ffuzz_split`, `trainlen_n_runs`, `trainlen_n_takes`, `vs100r_n_test_takes`, `polar_guard_runs`, `quiet_n_models` | Retenu, section 3 |

## Théorie

| Énoncé | Statut | Rôle dans l'article |
|---|---|---|
| T2 | D (2026-10-02) | Section 2 et 6 : condition w₁ E∣y∣ < w₂ m sous laquelle une sortie exactement inversée est un minimum local strict le long du gain de sortie pour L1 + MR-STFT ; explique pourquoi le poids relatif des deux termes n'est pas sans dimension ; rétrodiction compatible avec la run inversée, sans en donner la probabilité |

- T2, portée à rappeler dans le texte : une direction, une solution exacte, un lot ; corollaire non établi pour la tête tanh du banc ; ne dit pas si la garde de polarité est nécessaire.
- T2, à compléter (sans changer son statut) : marge numérique sur Big Muff écrite dans une fiche JSON par `theory/t2_check.py` ; marge sur les lots d'entraînement (non calculée) ; prédiction T2-P1 (non mesurée, environ 12 graines par bras).
- À développer, non inscrit au registre : coût en ESR d'un retard pur d, ESR(d) = 2(1 − ρ_y(d)) pour une cible normalisée sur la même région, et son développement pour d petit (proportionnel à d² et au moment spectral d'ordre 2 de la cible) ; expliquerait la croissance du plancher F-Fuzz avec le fuzz et l'ordre de grandeur des gains VS100R. Pour le statut D : énoncé au registre `THEORY.md`, preuve complète, contrôle symbolique, contrôle numérique sur les vraies cibles et les instruments (`diagnosis/ffuzz/measure.py`, `demo/train_length/coreading_delay.py`), relecture indépendante.
- À développer, spéculatif : ce que le retard commun τ estimé par minimisation de l'ESR absorbe quand le modèle a aussi une erreur de phase (moyenne pondérée de l'écart de pente de phase ?) ; formaliserait « l'oracle peut absorber une erreur de phase ». Même chemin vers le statut D ; risque que l'énoncé n'ait pas de forme close utile.
- À développer, spéculatif : un modèle causal de la classe SSM-WaveNet ne représente un retard pur qu'approximativement ; borne d'erreur à formuler pour une couche diagonale ; risque : une pile profonde peut l'approcher, l'énoncé pourrait être faux en général. Ne pas écrire « ne peut pas représenter » avant preuve.
- Ne pas relier T1 ni T4 au retard de la VS100R : le projet ne l'a pas établi.

## Travaux liés

- Voir `RELATED.md` (revue bibliographique tenue à part, clés dans `refs.bib`).

## Manques avant soumission

- Coût MR-STFT d'un retard pur et identité ESR = 2(1 − ρ) sur les cibles F-Fuzz : écrire une fiche JSON ; CPU, quelques minutes, audio ToneTwist requis ; pas de nouvelle expérience (re-mesure d'un résultat de E-0039).
- Marge numérique de T2 : faire écrire à `theory/t2_check.py` une fiche JSON ; CPU, audio requis ; pas de nouvelle expérience.
- Lecture Nolly (D-0015) : attendre les rendus de l'utilisateur (T-0013), puis la file préinscrite `scripts/product_nolly_queue.sh` ; coût GPU de la file à estimer ; expérience nouvelle mais déjà préinscrite.
- F-Fuzz sur plusieurs graines ou partitions : 2 architectures × au moins 2 graines de plus, de l'ordre de 4 à 5 h de GPU par run d'après `FIX_ffuzz.md` ; nouvelle expérience, demande une décision de l'utilisateur (D-0010).
- Origine du retard VS100R : test entre entrées de l'interface (T-0011), bloqué par D-0013 ; nouvelle capture matérielle ; sans elle, l'article garde « origine non établie ».
- Confirmation de l'inversion des modèles publiés : demander checkpoints ou sorties aux auteurs ; aucune expérience de notre côté.
- Taux d'inversion sans garde (T2-P1) : environ 12 graines par bras, environ 85 h de GPU d'après `THEORY.md` ; nouvelle expérience, facultative pour cet article.
- Audibilité d'un retard sous-échantillon commun : aucune écoute n'existe (F-0009) ; à mentionner comme limite, pas comme résultat.
- Venue : confirmer dates, gabarit, nombre de pages et texte exact de la politique IA d'EUSIPCO 2027.
- Droits : vérifier les conditions d'attribution de ToneTwist AFx pour des figures dérivées ; décider si la capture VS100R est publiée.
- Tableau 1 : à composer par l'auteur à partir des clés listées.
