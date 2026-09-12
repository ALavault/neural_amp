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
