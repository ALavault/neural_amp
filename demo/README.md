# Démo : d'un appareil à un plugin temps réel

## Construire

```
make demo
```

Construit le plugin JUCE (VST3 + standalone + rendu hors ligne), vérifie que le
rendu du plugin est identique au moteur natif mesuré, puis régénère `REPORT.md`.

Les modèles doivent exister sous `demo/runs/` ; sinon les réentraîner avec
`uv run python scripts/product_train.py --device fulltone_full_drive_2` (idem
pour `electro_harmonix_big_muff`).

## Jouer la démo

1. Lancer `build/demo_plugin/FssrAmpDemo_artefacts/Release/Standalone/FSSR Amp Demo`.
   Vérifier que le périphérique audio est à **48 kHz** : le moteur NAM ne
   rééchantillonne pas, et le plugin affiche un avertissement en rouge sinon.
2. « Charger un modèle .nam » →
   `demo/runs/product_a2_fulltone_full_drive_2_seed0_v1/model_full.nam`.
3. « Charger DI + capture réelle » → sélectionner les deux fichiers de test de
   l'appareil (l'entrée DI et la capture cible ; l'ordre de sélection n'importe pas).
4. Basculer le sélecteur **A : modèle** / **B : réel**. Les deux flux sont
   alignés à l'échantillon près, donc la bascule est une comparaison directe.
5. Entrée guitare : le modèle est mono et lit le canal 1 de l'interface ; sa
   sortie est dupliquée sur les deux canaux.
6. `model_lite.nam` montre le compromis coût/fidélité (voir `REPORT.md`).

Le Big Muff est présenté comme le cas difficile : chiffres et repères de
littérature dans `REPORT.md`.

## Écoute aveugle A/B/X

`make demo` génère aussi `demo/listening/index.html` (ouvrir avec `file://`).
Trois extraits par appareil, pris à des positions fixes, réel contre modèle :
A et B sont tirés au sort à chaque chargement, X à chaque essai, et la page
affiche le score et la p-valeur binomiale. Les extraits audio restent locaux
(sources CC-BY-NC) et ne sont pas versionnés.

## Chaîne de capture

`make demo-capture-selftest` prouve la chaîne de bout en bout sans interface
audio : un modèle `.nam` déjà entraîné joue l'appareil inconnu, une latence et
un gain sont tirés au hasard à chaque exécution, et le contrôle qualité doit
reconstituer un jeu d'entraînement à partir de la seule capture.

Sur du vrai matériel, la séquence est la même :

```
uv run python scripts/product_capture_selftest.py   # génère demo/capture/reamp_reference.wav
# jouer ce fichier dans l'appareil, enregistrer le retour
uv run python scripts/product_capture_qc.py demo/capture/reamp_reference.wav capture.wav sortie/
uv run python scripts/product_train.py --device <appareil>
```

Le QC ne reçoit que le signal envoyé et la capture reçue : il retrouve la
latence à l'échantillon près depuis les blips de calibration, rapporte le
niveau du programme en dBFS, et refuse une prise écrêtée ou silencieuse.

## Dossier de démonstration

`make demo` génère aussi `demo/deck/index.html` : dix volets pour un éditeur de
plugins. Chaque chiffre est lu dans `demo/report.json` et `demo/RUNS.jsonl`, donc
une nouvelle mesure met le dossier à jour et aucun chiffre ne peut dériver de sa
source.

## Index des dossiers (2026-10-10)

Voie produit (D-PRODUCT-001) : aucun de ces dossiers ne porte de revendication scientifique. Les
verdicts sont dans les documents cités, pas ici. Le registre des runs est `RUNS.jsonl`.

**Captures et fichiers sources**
- `capture/valvestate/` : séance VS100R du 2026-09-26 (envoi d'effets et sortie ligne) : `SESSION.md`,
  `layout.json`, `settings.csv` ; prévisions et analyses dans les `predictions_*.md` du dossier.
- `capture/valvestate_mic/` : séance VS100R au micro : `SESSION.md`, `AUDIT_20261008.md`.
- `capture/v2/` : fichier source et feuille de séance de la prochaine capture, logicielle
  (Nolly, ampli Rhythm, 192 kHz) : `SESSION.md`, `layout.json`. Générés par
  `scripts/product_capture_v2.py`.
- `capture/v2_44k1/` : variante 44,1 kHz séparée, pour le banc KIPR seulement : `PROTOCOL.md`.
  Générée par `scripts/product_capture_v2_44k1.py`.

**Évaluations préinscrites** (le document contient règles, résultats et lecture)
- `train_length/predictions.md` : longueur d'entraînement par prise (30, 60 ou 150 s), co-lecture
  à retard compensé, décision de l'utilisateur.
- `odd_gate/predictions.md` : porte impaire (structure KIPR) contre la porte actuelle.
- `greybox/predictions.md` : boîte grise façon KIPR contre SSM-WaveNet.
- `kipr_known_answer/predictions.md` : banc à réponse connue (plancher de 2ᵉ harmonique).
- `fft_size/` : taille de FFT et TF32 : `NOTE.md`, `predictions_fast_notf32.md`.
- `sr_transfer/` : transfert de fréquence d'échantillonnage : `RESULTS.md` et ses `predictions_*.md`.
- `param/` : modèles paramétriques FiLM : `RESULTS.md` et ses `predictions_*.md`.
- `chain_ultrasonic/` : test D sur systèmes synthétiques : `RESULTS.md`.

**Sorties et outils**
- `nablafx_bench/` : une sortie JSON par run du banc ; `butterfly/` : sorties des pilotes
  papillon (analyse dans `diagnosis/butterfly/`).
- `listening/` : pages d'écoute (`seeds.html`, `butterfly.html`) ; `plugin/` : plugin JUCE ;
  `deck/` : dossier généré par `make demo` ; `runs/` : journaux des files.
- `resplit/`, `resplit_fulltone/` : manifestes des découpages intra-prise.
- `REPORT.md`, `DIAGNOSTICS.md`, `DATA_DIAGNOSIS.md`, `SOTA_COMPARISON.md`,
  `RESPLIT_COMPARISON.md` : fiches de septembre ; lire l'erratum en tête quand il y en a un.

Hors de `demo/` : la mission sondes de métriques est dans `diagnosis/probe_metrics/REPORT.md`.
