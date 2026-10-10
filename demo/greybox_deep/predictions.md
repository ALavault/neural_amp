# Boîte grise approfondie façon KIPR : préinscription

Voie produit (D-PRODUCT-001), aucune revendication scientifique. Décision D-0014 (research-memory,
2026-10-10). **Écrit le 2026-10-10, avant tout run de cette piste.** Les règles ci-dessous ne
changent plus après les chiffres.

## Question

Demande de l'utilisateur, le 2026-10-10 :
- tester une version « dopée » de la boîte grise façon KIPR, avec plus de niveaux et de couches
  qu'une seule non-linéarité ;
- mieux mesurer les effets mémoire ;
- essayer sur les jeux DI + ampli ;
- comparer aussi aux modèles neuronaux existants.

**Décisions que la mesure change :**
- faire ou non d'une boîte grise approfondie la ligne de base DSP des comparaisons ;
- savoir si l'effort doit porter sur la profondeur (étages), sur la mémoire ou sur la portée
  linéaire.

## Ce que le diagnostic établit déjà (`DIAGNOSIS_greybox.md`, mesures en cours)

Oracles par prise : la structure de GREY est ré-ajustée sur la moitié A d'une prise et lue
sur la moitié B.

**État au moment d'écrire :**
- section de **validation**, 3 prises sur 8 ;
- témoin d'optimisation (V0c), 2 prises sur 8.

**Résultats :**
- **Portée linéaire.** Porter les FIR de 32 à 512 coefficients (V1) divise l'ESR_B par 2,2 à
  2,8 face à V0. Contre le témoin qui reçoit le même nombre de pas sans ajout, elle divise
  encore par 2,0 à 2,6.
- **Profondeur.** Un second étage (V4) divise l'ESR_B par 1,07 à 1,21 face à V0. Le témoin
  seul donne déjà 1,05 à 1,07.
- **Sur les prises de test** (4 sur 8), le tableau est le même, et le biais comme les
  enveloppes lentes n'y apportent presque rien.

**Biais de sélection, déclaré.** Ces oracles sont ajustés sur des prises de test. L'ajout des
FIR longs à toutes les variantes est donc éclairé par les prises sur lesquelles on lira
ensuite les runs. C'est pour cela que la conception est vérifiée sur la section de
validation, et que la lecture ci-dessous garde GREY et C comme références fixées d'avance.

## Modèle

`src/fssr_nam/models/deep_greybox.py`, `DeepGreyBoxAmp`, option `--model deepgreybox` du banc.
- Il contient GREY à l'identique quand ses ajouts sont neutres (test, 1e-12).
- Chaque ajout démarre neutre :
  - **étages supplémentaires** : chemin direct de poids 1 et branches de poids 0 ;
  - **biais lisse** avant le coupe-pic, dont on retranche la sortie statique ;
  - **enveloppes lentes** de 50 et 300 ms, qui agissent sur le gain d'attaque et sur le biais ;
  - **FIR d'entrée et de sortie** plus longs.
- Les boutons pilotent tous les coefficients par une tête dont la dernière couche part de
  zéro, comme GREY.

## Bras

**VS100R** (envoi FX, 48 kHz, partition 925, graines 42 à 44) :
- mêmes drapeaux que C et GREY, y compris les pas de division du bras C de même graine ;
- contrôle de la validation contre C après chaque run.

| Bras | Run | Configuration |
|---|---|---|
| GREY (référence, existant) | `vs100r48grey_s*` | 1 étage, FIR 32 |
| L | `vs100r48gbL_s*` | 1 étage, FIR 512 |
| F1 | `vs100r48gbF1_s*` | 1 étage, FIR 512, biais, enveloppes 50 et 300 ms |
| F3 | `vs100r48gbF3_s*` | **3 étages**, FIR 512, biais, enveloppes 50 et 300 ms |
| C (référence, existant) | `vs100r48fast_film_s*` | SSM-WaveNet, FiLM joint |

**JVM410H OD1** (sortie haut-parleur, graine 42, partitions 925, 736, 688, 744 et 369) :
- références appariées par partition : les runs du bras A (`param_ssm_wavenet_marshall`
  pour 925, `g2_film_joint_s*` pour les autres) ;
- protocole du bras A : divisions sur plateau, arrêt anticipé permis, 15 000 pas.

| Bras | Run |
|---|---|
| GREY | `jvm_grey_s*` |
| L | `jvm_gbL_s*` |
| F3 | `jvm_gbF3_s*` |

Le banc traite le JVM comme du 48 kHz nominal alors que l'audio est à 44,1 kHz. Le bras A
l'a été de la même manière. Les constantes de temps de la boîte grise sont donc nominales,
environ 8 % plus longues en réalité.

**File** : `scripts/product_greybox_deep_queue.sh`, depuis un runner épinglé.
1. Porte au bit du chemin par défaut du banc, modifié depuis la dernière porte.
2. Essai de 100 pas pour chaque configuration.
3. Runs dans cet ordre : VS100R L, VS100R F3, JVM GREY, JVM L, VS100R F1, JVM F3.

Chaque run, échecs compris, va dans `demo/RUNS.jsonl`.

## Lectures (`demo/greybox_deep/evaluate.py`, commité avec ce texte)

**VS100R.** Lecture P de `demo/train_length` (ln ESR par prise sur `di_test`, moyenne par couple
gain-contour, couple non vu exclu), puis d_i = s(bras) − s(référence) par graine. Les
étiquettes « meilleur », « moins bon », « équivalent » et « non tranché » suivent la règle P.

Lectures principales :
1. **F3 contre GREY** : l'approfondissement améliore-t-il la boîte grise ? Verdict
   « boîte grise approfondie meilleure que GREY » si l'étiquette est « meilleur » ; sinon,
   l'étiquette est rapportée telle quelle.
2. **F3 contre C** : verdict « boîte grise approfondie équivalente à C (ESR) » si l'ESR est
   acceptable (d ≤ +0,10 et chaque d_i ≤ +0,25) ; sinon « moins bonne que C (ESR) ».
   L'étiquette est rapportée à côté.

Lectures secondaires, descriptives :
- L contre GREY : portée linéaire ;
- F1 contre L : biais et mémoire ;
- F3 contre F1 : profondeur ;
- L contre C.

**Co-lecture à retard compensé** : la règle de `coreading_delay.py`, inchangée, est appliquée à
tous les bras. Les mêmes différences sont rapportées, à titre descriptif.

**Mémoire** (`demo/greybox_deep/HISTORY_MEASURE.md`, définition corrigée) :
- Δγ_T = γ_T(modèle) − γ_T(appareil) pour T = 100 et 300 ms ;
- médiane sur les graines et les 8 prises de test, par bras ;
- rapporté à côté de l'ordre de grandeur d'un affaissement de 3 dB (0,19) et du plancher de S0.

**JVM.** ESR de test du banc pour chaque fiche, puis d_s = ln ESR(bras) − ln ESR(référence) par
partition. Mêmes étiquettes et même règle d'acceptabilité, appliquées aux 5 valeurs.
- Lectures principales : F3 contre GREY, et F3 contre A, avec les verdicts ci-dessus en
  remplaçant C par A.
- Lectures secondaires : L contre GREY, GREY contre A, L contre A.
- Mémoire : Δγ sur les 9 réglages de test, entrée commune de 95 s.

**Aussi rapportés** : le nombre de paramètres (traitement et tête séparés), le retard compensé
de chaque run, et chaque étiquette même si elle contredit les attentes.

**Avant de lire** :
- chaque run VS100R compte 15 000 pas, et son contrôle contre C est `ok` ;
- chaque run JVM a sa fiche ;
- les valeurs lues sont finies.

Un bras incomplet n'est pas lu. Ses runs présents sont rapportés tels quels.

## Attentes (repères, pas des seuils)

- **L contre GREY** (VS100R) : d ≤ −0,7, soit une ESR au moins divisée par 2, d'après les
  oracles.
- **F1 contre L et F3 contre F1** : |d| ≤ 0,2. Les oracles ne trouvent presque rien à la
  mémoire ni à la profondeur.
- **F3 contre C** (VS100R) : « moins bon ». L'écart à retard compensé est plus grand, car C
  y gagne environ 1,6 en ln et GREY presque rien.
- **JVM** : pas d'attente chiffrée. C'est le cas qu'aucune sonde ne représentait (Q-0014).
- **Mémoire** : aucun affaissement de 3 dB détecté sur ces appareils, donc |Δγ| petit pour
  tous les bras, C compris.

## Ce qui ne pourra pas être conclu

- Rien sur la forme optimale d'une boîte grise : trois configurations seulement.
- Rien à l'écoute.
- Pour chaque appareil, une seule capture. Trois graines pour la VS100R, cinq partitions à
  une graine pour le JVM.
- La conception des bras dépend d'oracles en partie lus sur les prises de test (voir plus
  haut).
- Rien sur le routage du moteur KIPR, dont aucun code ni coefficient n'est repris.

## Budget

Estimation, à corriger au premier run :

| Bras | Durée par run |
|---|---|
| GREY | 51 min (mesuré, VS100R) |
| L | environ 1 h |
| F1 | environ 1 h 30 |
| F3 | jusqu'à 4 h (trois étages) |

Total : 24 runs, de 35 à 50 h de GPU. L'ordre de la file permet de s'arrêter en gardant des
comparaisons complètes.

## Corrections avant tout run (2026-10-10, relecture Codex et relance)

Aucun run de la piste n'a abouti au moment de ces corrections. Les règles de verdict ne
changent pas.

**`evaluate.py` applique maintenant ce que le texte annonçait :**
- **Contrôles avant lecture.** Un run ne compte que s'il a 15 000 pas, un contrôle contre C
  `ok` (nouveaux bras VS100R) et des valeurs finies. Sinon, son bras est incomplet et n'est
  pas lu.
- **Co-lecture compensée.** On reprend la règle de `coreading_delay.py` en entier :
  - une graine est retirée si son run ou celui de la référence touche le bord de la grille ;
  - une lecture privée d'une graine ne reçoit pas d'étiquette ;
  - le contrôle d'effet de bord conditionne la lecture ;
  - le résultat est confronté à la lecture brute (« robuste au calage » ou « divergence »).
- **Paramètres.** La tête FiLM de SSM-WaveNet (`film_net`) est comptée comme tête, et non
  plus comme traitement.
- **Sorties ajoutées** : les phrases de verdict des lectures principales, ainsi que le
  plancher de S0 et l'ordre de grandeur d'un affaissement de 3 dB, à côté des Δγ.

**Précisions de texte :**
- Le retard compensé ne concerne que la VS100R : aucun protocole de calage n'est défini pour
  le JVM.
- Les essais de 100 pas sont notés dans le journal de la file. Seuls leurs échecs vont dans
  `demo/RUNS.jsonl`, comme pour les runs principaux, réussis ou non.

**Relance `_r2`.**
- La première porte (`repro_gate_vs100r48_greybox_deep`) a échoué avant tout pas : PyTorch
  plante dans `PeerToPeerAccess.cpp`, faute de NVML. L'échec est inscrit dans
  `demo/RUNS.jsonl`.
- **Cause, établie par un essai minimal** : avec `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`,
  posé par `product_queue_lib.sh`, PyTorch a besoin de NVML. Or le pilote et sa bibliothèque
  NVML sont désaccordés depuis le 2026-10-10, jusqu'au prochain redémarrage. Sans ce réglage,
  CUDA fonctionne.
- **Correctif** : la bibliothèque n'active ces segments que si `nvidia-smi` répond.
- La porte `_r2` doit rester identique au bit à la référence, qui avait été calculée avec ces
  segments actifs. Si elle échoue, la file s'arrête.
