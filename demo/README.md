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
