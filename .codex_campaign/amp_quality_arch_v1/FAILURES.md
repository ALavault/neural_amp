# Échecs

Aucun run scientifique n'a été lancé dans cette lignée.

- `ARCH-F-001` — Le candidat `selective_s6_x2` est fini, causal, conforme en
  blocs et temps réel en C++, mais échoue au préflight d'entraînement : médiane
  14,5130264 s/update pour 2 048 échantillons, soit 20,15698 h projetées à
  5 000 updates contre un cap de 6 h. `torch.compile` n'a pas produit une
  première itération en deux minutes sur 512 échantillons et a été interrompu.
  Rejet technique avant gel, sans audio/ESR et sans famille de remplacement.
- `ARCH-F-002` — Le comparateur `selective_s6_glu` est retiré avant gel. Au
  commit primaire épinglé, `Models.py` appelle une classe `S6` non importée,
  `Mamba.py` contient une indentation/syntaxe invalide et des noms de classe
  incohérents, et son environnement TensorFlow 2.15 n'est pas celui du projet.
  Une réécriture ne serait plus une reproduction loyale. Les quatre autres
  comparateurs ouverts sont conservés; aucun score audio n'a motivé ce retrait.
- `ARCH-F-003` — La gate mécaniste gelée rejette les cinq candidats. Le contrôle
  micro-TCN échoue lui-même aux gardes d'amplitude/corrélation après 500 updates;
  les bus physique déterministe et S6 régressent respectivement de 0,21 % et
  0,32 % sur le dynamique. La cascade améliore deux-clippers de 24,65 %, sous
  le minimum de 50 %, et reste sous-amplifiée. Les dix runs sont finis et
  valides : verdict scientifique `NO-GO-ARCH`, pas `INVALID`.

Risques encore actifs : limite cuDNN du LSTM lors des validations trop longues
(chunks ≤32 768 requis), anticipations TFiLM des comparateurs non déployables,
et enveloppe CPU x2 très serrée. La dépendance NablAFx `rational` est évitée par
des adapters locaux dont la parité au source épinglé est testée.
