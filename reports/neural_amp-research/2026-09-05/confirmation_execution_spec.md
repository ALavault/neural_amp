# Spécification d'exécution post-développement v9

Ce document prépare l'implémentation sans autoriser une étape ni modifier le
snapshot v9 actuellement en préflight. Les chemins `test` ci-dessous sont des
concepts de protocole; aucun membre ou fichier test n'a été résolu.

## 1. Revue développement et locks

Préconditions : exactement onze run IDs v9, onze statuts `completed`, onze
entrées uniques dans `RUN_LEDGER.jsonl`, checkpoints 500–7 500 et métriques de
validation sous le contrat de contexte commun.

1. Vérifier les deux candidats Fulltone/Big Muff et les neuf comparateurs.
2. Construire neuf lignes comparateur `{device,family,seed,file_id,metrics}`.
3. Appeler la sélection gelée : moyenne également pondérée des ESR médians par
   dispositif; gap strictement inférieur à 1 % départagé par `L1+MR-STFT`, puis
   ordre déterministe gelé.
4. Persister atomiquement `DEVELOPMENT_REVIEW.json`, `COMPARATOR_LOCK.json`,
   `CANDIDATE_LOCK.json` et `RECIPES_LOCK.json` avec compteurs test à zéro.
5. Figer pour chaque famille : constructeur, optimisation, traitement de
   latence, contexte 48k/14,4k, règle de checkpoint et commit d'exécution.

Le failure mining est généré sur validation par le runner mais reste
`selection_eligible=false`; il ne peut modifier aucun lock.

## 2. Téléchargement et audit confirmation

Seulement après les quatre locks ci-dessus :

- Télécharger `HarleyBenton-Rodent.zip` (2 913 794 564 octets,
  `md5:7ab43d078cc195f7595c8b1e1fd09723`) et
  `HarleyBenton-FuzzyLogic.zip` (1 272 358 120 octets,
  `md5:a702262de22616643a47df5d090c68e1`).
- Confirmer l'identifiant exact `CC-BY-NC-4.0` via métadonnée machine ou README.
- Ouvrir seulement les membres `/trainval/` des deux archives wet et les
  membres `/trainval/` déjà déclarés de `DRY-with-markers.zip`.
- Pour chaque dispositif, préparer cinq sources internes attendues : quatre
  train (`idmt-gtr2`, `nam`, `prvt-gtr`, `yt-bass`) et une validation
  (`idmt-gtr4-sg`).
- Auditer chaque paire : mono, 48 kHz, longueurs identiques, valeurs finies,
  énergie positive, aucun dépassement numérique, deux marqueurs ≥0,05 et
  alignement dry/wet à ±1 échantillon.
- Interdire corrélation, recalage, gain, DC ou normalisation.
- Écrire en quarantaine; promouvoir atomiquement seulement après 10/10 succès.
- Persister `test_member_paths_resolved=0` et
  `test_waveform_samples_read=0`.

Toute archive/checksum/licence/politique incompatible arrête la campagne; aucun
fallback de données n'est autorisé.

## 3. Matrice confirmation

Créer exactement douze nouveaux runs :

`2 dispositifs × {candidat, comparateur global} × seeds {0,1,2}`.

Format :
`quality_teacher_v9_confirmation_train_<device>_<family>_seed<n>_v1`.

Chaque run repart de l'initialisation gelée; aucun checkpoint développement
n'est utilisé comme initialisation. L'entraînement, le contexte commun, la
latence et le watchdog sont identiques aux recettes verrouillées. Le checkpoint
est choisi uniquement par perte totale validation; les 15 valeurs de validation
et le checkpoint sélectionné sont persistés. Le plafond global de 300 GPU-h
continue depuis le développement et borne chaque nouvelle alarme par
`min(48 h, budget restant)`.

La gate `CONFIRMATION_CHECKPOINTS_LOCK.json` exige 12/12 runs complets et douze
sélections déterministes. Une matrice incomplète n'ouvre pas les tests.

## 4. Ouverture test simultanée

Préconditions littérales passées : `candidate_lock`, `comparator_lock`,
`recipes_lock`, `confirmation_data_audit`,
`confirmation_checkpoints_lock`.

Une seule commande obtient l'autorisation puis résout les cinq dispositifs
ensemble : Fulltone, Big Muff, Ampeg, Rodent, Fuzzy Logic. Les trois premiers
sont descriptifs; seuls Rodent et Fuzzy Logic entrent dans l'estimand
confirmation. Aucune sélection de modèle, checkpoint, métrique ou seuil n'est
possible après cette ouverture.

## 5. Lignes statistiques et verdict

Pour chaque fichier confirmation et seed, construire une ligne appariée avec
les métriques candidat et comparateur : ESR, MAE, MR-STFT, log-mel,
enveloppe/transitoire, corrélation et gain_error. Les deux prédictions partagent
exactement les blocs, resets, prérolls et coordonnées source.

Le bootstrap gelé :

- 10 000 réplications, seed 20 260 831;
- resampling apparié fichiers puis seeds dans chaque dispositif;
- dispositifs Rodent/Fuzzy Logic comme strates fixes de poids égal;
- aucune fenêtre indépendante utilisée comme pseudo-réplication.

`GO-OBJECTIVE-SOTA` exige simultanément : amélioration ESR agrégée ≥10 %,
borne basse 95 % >0, amélioration ESR médiane >0 par dispositif, amélioration
`L1+MR-STFT` et borne basse >0, régressions secondaires ≤5 %, corrélation >0,9
et `gain_error > -0,2`. Une exécution complète qui échoue à au moins un seuil
donne `NO-GO-OBJECTIVE-SOTA`; fuite, provenance ou matrice incomplète donnent
`INVALID`.

## 6. Audit final

Recalculer le verdict depuis les JSON source, vérifier 10 000 réplications,
l'unicité des 25 trajectoires historiques autorisées (2 héritées v5 + 11 v9 +
12 confirmation), le ledger de budget, l'ordre temporel des locks et le compteur
test nul avant ouverture. Synchroniser ensuite STATE, MATURITY, DECISIONS,
FAILURES, CLAIMS, HANDOFF et VERDICT sans transcription manuelle des nombres.
