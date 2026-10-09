# Boîte grise façon KIPR contre SSM-WaveNet : préinscription

Voie produit (D-PRODUCT-001), aucune revendication scientifique. **Écrit le 2026-10-06, avant tout
entraînement de cette piste.** Les règles ci-dessous ne changent plus après les chiffres.

## Question

Que rapportent les 28 000 paramètres de SSM-WaveNet sur la VS100R, face à un modèle structuré de
quelques centaines de paramètres de traitement ? Ce modèle reprend la structure du moteur DSP
KIPR : filtres, gain d'enveloppe, courbe impaire entre gains selon le signe. Et ce modèle
structuré reproduit-il les cibles KIPR, issues de sa propre famille ?

**Décision que la mesure change** : faire ou non de la boîte grise la ligne de base des prochaines
comparaisons. Si elle est équivalente, un modèle de cette taille suffit, et la boîte noire doit
justifier son coût.

## Modèle (`src/fssr_nam/models/greybox.py`, `GreyBoxAmp`)

Écrit pour ce dépôt, sans code de l'archive KIPR (GPL-3).
- Chaîne : FIR d'entrée (32 coefficients), puis quatre branches. Chaque branche : filtre du
  premier ordre, gain d'attaque compressé par une enveloppe de |u|, coupe-pic algébrique impair
  x/(1+|x|^q)^(1/q) entre des gains e^±β selon le signe, filtre du premier ordre. Ensuite, mélange
  des branches et FIR de sortie.
- Les boutons pilotent les coefficients par une tête (2 → 32 → 40) dont la dernière couche part
  de zéro.
- Taille : 1 524 paramètres avec boutons, dont 1 416 dans la tête ; 108 sans boutons.
- Aucun biais : le modèle est impair quand les β sont nuls. Contrôles :
  `tests/numerical/test_greybox.py` (fini, gradients, imparité, zéro → zéro, asymétrie,
  causalité, boutons, entier/pas à pas, remise à zéro).
- Coefficients structurels sans décroissance des poids (`_optim`, `--honor-optim`).

## Bras

- **VS100R** : `vs100r48grey_s42`, `_s43`, `_s44`, `--model greybox`, mêmes drapeaux que le
  bras C. Le lr reste 0,01 comme pour C, avec les pas de division du jumeau A. Référence : le bras C
  `vs100r48fast_film_s*`, par graine.
- **KIPR** : `kipr_{sym,asym}_s{42,43,44}_grey`, `known_answer.py --tag grey`, protocole r2.
  Référence : `kipr_*_r2`.

Avant les runs VS100R : la même porte au bit du chemin par défaut que la piste impaire (même
file), puis un smoke de 100 pas de la boîte grise. Après chaque run : validation identique à
celle du bras C.

## Lectures (script `demo/odd_gate/evaluate.py --arm grey`, commité avec ce texte)

**ESR VS100R** : lecture P (voir `demo/odd_gate/predictions.md`), d_i = s(GREY) − s(C).
« Acceptable » si d ≤ +0,10 et chaque d_i ≤ +0,25.

**Verdict de la piste** : « boîte grise équivalente au bras C (ESR) » si l'ESR est acceptable ;
sinon « boîte grise moins bonne que le bras C (ESR) ». On rapporte avec : exp(d), le rapport moyen
d'ESR, et l'étiquette meilleur / moins bon / équivalent / non tranché.

**2ᵉ harmonique VS100R**, même lecture R1 que la piste impaire (« réduit », « inchangé »,
« partiel »). Rapportée à côté du verdict, sans le modifier ; |D3| est descriptif.

**KIPR** (`known_answer.py measure --tag grey`) : mêmes portes et même lecture que r2. Comparaison
à r2 : « réduit » si E_sym(grey) ≤ E_sym(r2) − 10 dB, seulement si les deux lectures sont valides
(sinon, comme pour la piste impaire). Rapportée à côté du verdict.

**Descriptif, sans seuil** : les mêmes que la piste impaire (|D3|, prises 16 et 36, nombre de
paramètres) ; exp(d) est le champ `esr_ratio`.

**Attente, non seuil** : la boîte grise moins bonne en ESR d'au moins 0,5 en ln (ESR × 1,6).

## Ce qui ne pourra pas être conclu

- Rien sur la forme optimale de la boîte grise : une seule structure est testée, sans recherche
  d'architecture.
- Rien à l'écoute.
- Avec 3 graines, seuls des effets nets ressortent.

## Amendement avant tout entraînement (2026-10-06, après la lecture de r2)

La reprise r2 est sans lecture : deux runs sur six sont de polarité inversée
(`demo/kipr_known_answer/predictions.md`, résultats r2). Pour les runs KIPR de cette piste, la
mesure (`known_answer.py measure`) prend pour la sortie le signe de sa corrélation avec la cible
sur les segments `train_noise`, jamais sur `test_noise`. Ensuite seulement viennent la porte
d'alignement et les lectures, inchangées. L'entraînement ne change pas. r2 n'ayant pas de lecture
valide, la comparaison KIPR à r2 sera descriptive (règle déjà écrite). Les lectures VS100R ne
changent pas : le banc VS100R a sa garde de polarité.

## Résultats KIPR (2026-10-06, six runs, exit 0 ; lecture intermédiaire, VS100R en attente)

Lecture `known_answer.py measure --tag grey`, sortie `known_answer_grey.json`.

**Verdict KIPR préinscrit : aucune lecture (alignement).** Comparaison à r2 : descriptive (r2 sans lecture valide).

- `sym` : excès de H2/H1 (médiane des graines) +15.5 dB à -24 dBFS, +28.3 dB à -12 dBFS, +34.9 dB à -6 dBFS ; écart de H3/H1 -3.1, -3.1, -3.1 dB.
- `asym` : excès de H2/H1 (médiane des graines) +3.8 dB à -24 dBFS, +5.0 dB à -12 dBFS, +4.7 dB à -6 dBFS ; écart de H3/H1 -1.1, -0.6, -0.2 dB.
- Alignement à −12 dBFS (décalage, ESR, polarité appliquée) : `kipr_sym_s42_grey` 6, 1.69, +1 ; `kipr_sym_s43_grey` 6, 1.69, +1 ; `kipr_sym_s44_grey` 6, 1.69, +1 ; `kipr_asym_s42_grey` 0, 0.24, +1 ; `kipr_asym_s43_grey` 0, 0.24, +1 ; `kipr_asym_s44_grey` 0, 0.24, +1.
- **Descriptif** : les trois graines donnent des runs identiques au bit près. Sans bouton, la boîte
  grise n'a aucun tirage aléatoire (paramètres initialisés à des constantes) et les données sont les
  mêmes : la préinscription ne l'avait pas prévu ; les « trois graines » sont un seul run. La cible
  symétrique n'est pas suivie (décalage 6 à 7, ESR 1,7 à 1,8), la cible asymétrique l'est (décalage
  0, ESR 0,17 à 0,25).

## Note de provenance (2026-10-06, correction de routage transmise par Codex ; règles inchangées)

La place des filtres du premier ordre avant la courbe, dans cette boîte grise, reprend la
proposition d'apprentissage de l'archive KIPR V2. Ce n'est pas un routage établi du moteur. La
trace d'exécution transmise le 2026-10-06 place les filtres du premier ordre observés
(descripteurs 7193, 7226, 727f) **après** la saturation, sur un tampon temporaire ; une adresse
réutilisée ne prouve pas une connexion. Seuls sont validés, chacun à part : l'enveloppe avec les
conversions R/G ×4 puis ×16, et le fragment courbe → G inverse → deux biquads. La boîte grise
préinscrite n'est pas modifiée ; ses résultats ne diront rien du routage du moteur.

## Précisions avant l'évaluation (2026-10-08, relecture Codex de la piste ; règles et script inchangés)

Écrites avant toute lecture VS100R du bras GREY ; `evaluate.py` est inchangé depuis `f6cf3a2`.

- **Comparaison KIPR à r2.** La règle préinscrite donne la priorité au bras : la lecture KIPR de
  GREY n'étant pas valide (« aucune lecture (alignement) »), la comparaison vaut **« aucune
  lecture (bras) »**. L'amendement et la lecture intermédiaire du 2026-10-06 écrivaient à tort
  « descriptive ». Le JSON KIPR de GREY porte aussi des contrôles hors seuil (H3 de la cible
  symétrique, H2 de la cible asymétrique à −12 dBFS).
- **Deux sens d'« équivalent ».** Le verdict « boîte grise équivalente au bras C (ESR) » suit
  l'acceptabilité (d ≤ +0,10 et chaque d_i ≤ +0,25). L'étiquette descriptive « équivalent » de la
  lecture P est plus étroite (|d| ≤ 0,05 et chaque |d_i| ≤ 0,15). Les deux sont rapportés tels
  quels.
- **Bandes de la 2ᵉ harmonique.** Comme pour la piste impaire : centres de 250 Hz à 8 kHz, le
  centre 198,4 Hz étant exclu par le script préinscrit.
- **Tests.** La parité par blocs testée compare l'appel entier et le pas à pas (`step`), seule API
  de flux du modèle ; `forward` repart toujours de l'état nul. Un test ajouté vérifie que la remise
  à zéro efface aussi les historiques FIR : FIR aléatoires, 100 échantillons avant la remise à zéro,
  et contre-épreuve sans remise à zéro.
- **Contrôles avant lecture**, affichés avec les résultats :
  - chaque run du bras compte 15 000 pas ;
  - le JSON Hammerstein régénéré contient les 24 cas attendus, identifiés par run et par prise ;
  - les valeurs lues sont finies.

## Résultats (2026-10-09, trois runs VS100R et six runs KIPR complets)

**Runs.**
- VS100R : `vs100r48grey_s{42,43,44}`, 15 000 pas sur 15 000, contrôle face au bras C
  `ok` (`demo/RUNS.jsonl`, fiches `demo/nablafx_bench/`), sortie 0 (`demo/runs/kipr_tracks_queue.log`).
- La porte au bit du chemin par défaut (`repro_gate_vs100r48_kiprtracks_r3`, 372 valeurs,
  0 écart) est passée avant le premier run VS100R (2026-10-08, 03:44).
- KIPR : `kipr_{sym,asym}_s{42,43,44}_grey`, 3 000 pas chacun.

**Commandes**, sur CPU, après le dernier run :
`diagnosis/hammerstein/vs100r_hammerstein.py`, puis
`demo/kipr_known_answer/known_answer.py measure --tag grey`, puis
`demo/odd_gate/evaluate.py --arm grey`. Sortie : `demo/greybox/evaluation.json`.

**Contrôles avant lecture** (précisions du 2026-10-08) :
- 24 cas Hammerstein par niveau pour GREY et pour C : 3 runs attendus × 8 prises de test,
  identifiants exacts, aucune valeur non finie dans les bandes de 250 Hz à 8 kHz.
- Les 216 cas déjà mesurés (C, A48, B192 : 9 modèles × 8 prises × 3 niveaux) sont reproduits à l'identique.
- `known_answer_grey.json` est identique à celui du 2026-10-06.
- Toutes les valeurs lues sont finies.

**Verdict préinscrit de la piste : boîte grise moins bonne que le bras C (ESR).**

**ESR VS100R : non acceptable, étiquette « moins bon ».** s = moyenne des ln ESR par couple.

| Graine | s(C) | s(GREY) | d_i |
|---|---|---|---|
| 42 | −3,952 | −1,552 | +2,400 |
| 43 | −3,967 | −0,832 | +3,135 |
| 44 | −3,929 | −1,717 | +2,212 |

- d = +2,582, soit un rapport géométrique moyen d'ESR exp(d) = 13,2.
- L'attente préinscrite (au moins +0,5 en ln, ESR × 1,6) est largement dépassée.

**2ᵉ harmonique VS100R : « partiel »**, rapportée sans modifier le verdict. Écart signé de D2 à
l'appareil, en dB.

| Niveau | C | GREY | Δ = GREY − C |
|---|---|---|---|
| −20 dBFS | +12,72 | +20,76 | +8,05 |
| −10 dBFS | +2,77 | +2,97 | +0,20 |
| −3 dBFS | +4,83 | +3,37 | −1,45 |

**KIPR : comparaison à r2 « aucune lecture (bras) ».** La lecture du bras reste « aucune lecture
(alignement) », de même que celle de r2.

**Descriptif, sans seuil :**
- |D3| médian, C puis GREY : 0,17 et 0,87 dB à −20 dBFS, 0,22 et 0,80 à −10, 0,31 et 0,95 à −3.
- Prises au gain 2,5, écart de ln ESR GREY − C par graine 42, 43, 44 :
  - prise 16 : +0,88, +2,05, −0,32 ;
  - prise 36 : +2,51, +2,89, +2,06.
- Écart de D2 à −10 dBFS, médiane des graines, C puis GREY : prise 16, +28,1 et +32,3 dB ;
  prise 36, +17,7 et +24,2 dB.
- Paramètres : C 28 321, GREY 1 524, soit 18,6 fois moins.

**Ce que cela ne dit pas** (liste préinscrite) :
- rien sur la forme optimale d'une boîte grise : une seule structure a été testée ;
- rien à l'écoute ;
- avec 3 graines, seuls des effets nets ressortent ; l'écart d'ESR l'est.

S'y ajoute, d'après la note de provenance : rien sur le routage du moteur KIPR.

Résultat de la voie produit, sans portée scientifique.
