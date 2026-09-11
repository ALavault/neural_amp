# Audit technique initial — 2026-08-28

## Faits observés

- Hôte de référence : Intel Xeon E5-2630 v3, AVX2, 32 CPU logiques.
- Entraînement : NVIDIA RTX PRO 4000 Blackwell, 24 467 MiB.
- Espace disponible au démarrage : environ 720 GiB.
- Environnement : Python 3.12, PyTorch 2.13.0, NumPy 2.5.2, SciPy 1.18.1.
- `make data-audit` passe sur `datasets/manifests/catalog.yaml`.
- Les paires locales couvrent Fulltone, Big Muff, Blackstar et UA1176 à 48 kHz.
- Les identités Fulltone sont disjointes. Big Muff ne publie pas l'identité de
  session/interprète au-delà des fichiers train/validation/test.
- Les sorties test Fulltone et Big Muff ont été utilisées historiquement à M4;
  elles ne sont plus prospectives. Aucun run enregistré n'a ouvert les sorties
  test Blackstar ou UA1176.
- `full_island_x2` est qualifié uniquement sur fixtures synthétiques. Son profil
  natif de référence a une latence de 32 échantillons et un p95 RTF bloc 64 de
  0,79176; cette valeur ne prouve pas le coût d'une future architecture.
- Le teacher x4 historique dépasse le budget temps réel (p95 RTF 2,01685), mais
  reste admissible comme teacher non déployé.
- La validation CUDA du LSTM Wright échoue sur le stack courant pour des séquences
  de 65 536/100 000 échantillons et passe jusqu'à 32 768. Toute validation
  récurrente utilisera des chunks gelés de 32 768 au maximum et un test de parité
  par rapport au rendu chunké CPU.
- L'import direct de NablAFx échoue avec `ModuleNotFoundError: rational`. Ce
  défaut est fermé sans installer cette activation inutilisée : les seuls
  sous-ensembles TCN–TFiLM et S4–TFiLM requis sont réimplémentés localement et
  comparés directement aux classes du commit épinglé. La parité est exacte.
- Les configs NablAFx TCN ont des champs réceptifs réels de 133 333 (small) et
  118 097 (large) échantillons. La mention RF2047 antérieure était erronée et
  ne doit pas être utilisée pour budgéter ces comparateurs.
- TFiLM max-pool le bloc courant de 128 échantillons. Empilé, il impose une
  anticipation bornée : 635/1 270 échantillons pour TCN et 508/1 016 pour S4.
  Les adapters ajoutent ce délai avant toute loss ou métrique et passent
  causalité/reset/parité en blocs; ces comparateurs ne passent pas la gate de
  latence déployable de 48 échantillons.

## Diagnostic des faibles résultats historiques

Les résultats M4 n'invalident pas une recherche architecturale large : les FSSR
testés avaient environ dix fois moins de paramètres qu'A2, un RF rapide de 31
échantillons contre 6 347 pour A2, et leur sortie Big Muff s'effondrait vers une
faible énergie. Élargir seulement les canaux a fermé 0,4 % du gap Fulltone et
25,2 % du gap Big Muff; le résidu est resté presque inactif. Cela favorise trois
hypothèses discriminantes : horizon intermédiaire insuffisant, modulation lente
placée trop tard, et objectif d'entraînement favorisant le bassin silencieux.

## Risques qui ferment la campagne

1. **Contamination.** Toute lecture de sorties test Blackstar/UA1176 avant le gel
   architecture/loss/schedule/export rend la confirmation `INVALID`.
2. **Fausse parité.** Un forward offline ne suffit pas; chaque modèle doit passer
   causalité, reset et parité sur blocs irréguliers.
3. **Fausse efficacité.** Les FLOP/s publiés ne remplacent pas le même binaire C++
   sur l'hôte gelé.
4. **Faux gain ASR.** Une sortie silencieuse ou tonalement régressée invalide la
   mesure même si son énergie d'alias paraît faible.
5. **Budget x2.** Le profil x2 actuel consomme presque toute la gate RTF. Les
   candidats devront limiter l'île non linéaire, fusionner leurs kernels ou
   démontrer une autre route AA passant les mêmes gates; ajouter naïvement x2 à
   un gros réseau n'est pas admissible.
6. **Données.** Aucun dataset physique 192 kHz n'existe; aucune revendication
   matérielle d'aliasing ne peut être construite par upsampling.

## Contrôles minimaux avant le screening

- test de compétence A2 et LSTM sur un extrait `debug` sans réserver de run;
- parité full/chunk/stream, reset, causalité, formes et gradients pour chaque
  famille;
- preuve que les chunks de validation bornés ne changent pas les métriques;
- budget analytique paramètres/MAC/état/scratch et microbenchmark Python indicatif;
- vérification que seuls train et validation sont adressables par le launcher;
- vérification fail-closed des caps familles, variantes, seeds, checkpoints et
  reprises.

## Addendum de faisabilité — 28 août 2026

Le benchmark natif aveugle valide les 24 couples famille/profil et retient les
six profils `max`. Un second préflight a mesuré forward+backward CUDA sur 2 048
échantillons, un warmup puis deux répétitions, sans lire de fichier audio. Les
cinq familles TCN se situent entre 0,0295 et 0,1681 s/update. La récurrence S6
pure Python atteint 14,5130 s/update, soit 20,157 h projetées à 5 000 updates;
elle dépasse le cap de 6 h et est rejetée avant gel. Une tentative
`torch.compile` sur seulement 512 échantillons n'a pas achevé sa première
itération en deux minutes. Le comparateur S6 officiel reste requis; seul ce
squelette candidat non optimisé est écarté.

L'audit exécutable ultérieur du comparateur S6 a aussi fermé cette voie, pour une
raison distincte : au commit primaire épinglé, la factory appelle `S6` sans
l'importer, tandis que `Mamba.py` présente une indentation invalide, deux
définitions incohérentes de `MambaLay` et des variables non définies. Le dépôt
requiert TensorFlow 2.15 alors que la campagne est PyTorch. Réparer et traduire
ces ambiguïtés créerait une nouvelle architecture, pas une reproduction loyale;
le comparateur est donc retiré avant gel et sans consultation de résultats audio.

Le préflight de forme d'entraînement réelle utilise 8 192 échantillons scorés
et des warm-ups suffisants : 6 346 pour A2, 49 152 pour Wright et les candidats,
RF plus latence pour les TCN NablAFx, et un segment total de 144 000 pour S4.
Avec la loss exacte auraloss 0.4.0 (commit
`1576b0cd6e927abc002b23cf3bfc455b660f663c`), les onze charges passent. Les
projections à 5 000 updates vont de 0,060 h à 4,598 h; la mémoire maximale est
7,39 Go pour S4 large. Cette mesure n'a lu aucun échantillon physique.
