# Lecture des rendus Nolly (capture v2) : règles figées

Voie produit (D-PRODUCT-001), mission sondes de métriques (research-memory D-0010), aucune
revendication scientifique. **Écrit le 2026-10-10, avant tout rendu.** Ce texte et le script
`nolly_reading.py`, commités ensemble, ne changent plus après les chiffres. Un écart éventuel sera
écrit dans une section datée « Écarts », sans modifier les règles. Le script calcule toutes les
portes, toutes les lectures et tous les verdicts ci-dessous ; rien n'est choisi après les données.

## Objet

Les rendus de Neural DSP Archetype: Nolly, ampli Rhythm (`demo/capture/v2/SESSION.md`), donnent
des écarts réels et connus : des changements de boutons, et un baffle activé. On y lit ce que
chaque métrique voit de ces écarts : ordre des sévérités, nature de l'écart, capacité des sondes,
spécificité. On ne juge pas un modèle. Aucun modèle neuronal, aucun entraînement, aucun
classement des sondes comme générateurs.

## Données attendues

- **Dossier de rendus** : les 42 prises nommées comme dans `demo/capture/v2/settings.csv`, plus
  `loopback_software.wav` et `center_stereo.wav`. WAV 32 ou 64 bits flottants, 192 kHz, mono
  (stéréo pour `center_stereo.wav`).
- **Entrée** : x est le fichier source v2 lui-même (`reamp_v2_192k.wav`, sha256 `9ebc26a3…`),
  commun à toutes les prises. Le rendu est hors ligne, et Reaper compense la latence déclarée par
  le plugin : aucun décalage n'est appliqué, ni à x ni aux sorties.

## Portes, dans l'ordre

1. **Bouclage logiciel** : `loopback_software.wav` est au format des rendus, fini, au moins aussi long
   que la source, égal à la source à 2⁻²³ près (un pas de 24 bits), et nul au-delà. Échec : aucune
   lecture.
2. **Stéréo** : `center_stereo.wav` est au bon format et ses deux canaux sont égaux à 2⁻²³ près.
   Échec : aucune lecture, car le rendu mono mélangerait deux canaux différents.
3. **Prises manquantes** : listées ; les paires qui en dépendent ne sont pas lues.
4. **Par prise** :
   - **format** : conteneur WAV, échantillons flottants, 192 kHz, mono, non vide. Un fichier
     illisible échoue à cette porte ;
   - **valeurs** finies, et longueur au moins égale à la source ;
   - **réponse aux marqueurs** : le RMS des deux sections de marqueurs dépasse d'au moins 20 dB
     celui de la section de silence ;
   - **retard** : identique aux marqueurs de début et de fin, à l'échantillon près à 192 kHz, et
     pas au bord de la recherche (±4 096 échantillons) ;
   - **paliers faibles non muets** : RMS au-dessus de −100 dBFS sur chaque segment faible des trois
     sections de paliers.

   Une prise qui échoue à l'un des quatre premiers points est exclue. Un seul palier muet arrête
   toute la lecture : le noise gate est probablement resté actif, et il touche aussi la DI.
5. **Ancres** : les prises 01, 20 et 39 doivent toutes passer la porte 4. Sinon, aucune lecture.
6. **Latence** : la médiane des retards aux marqueurs des prises sans baffle doit valoir au plus
   1 024 échantillons à 192 kHz (5,3 ms, la mémoire des sondes). Au-delà, la latence n'a pas été
   compensée : aucune lecture. Les retards et la polarité de chaque prise sont rapportés, jamais
   appliqués. Ils varient avec les filtres de tonalité, d'où l'absence de porte sur une latence
   commune.
7. **Déterminisme** : les trois ancres, fichiers entiers, sont-elles identiques au bit ?
   - **Oui** : le plancher est nul. Il n'y a ni rapport au plancher ni affirmation de détection ;
     les scores sont rapportés bruts.
   - **Non** : pour chaque score scalaire, le plancher est la médiane des trois paires d'ancres,
     et chaque score de chaque paire est aussi rapporté en multiple de ce plancher. Sont compris :
     chaque sonde et chaque segment, ESR, magnitude, phase, enveloppe, signatures harmoniques
     (totaux et ordres, aux deux fréquences), E2, et les écarts de multisinus (en valeur
     absolue).

   Dans les deux cas, chaque prise est comparée à l'ancre la plus proche dans l'ordre de rendu
   (égalité vers la plus petite), comme pour la VS100R.

## Paires, figées

- **Répétitions** : 01–20, 20–39, 01–39.
- **Prises « un bouton » et « combinaison »** : chacune contre l'ancre la plus proche.
- **Baffle** : 40 contre 32, 41 contre 38, 42 contre 39 (même réglage, baffle coupé).
- **Échelles de H1** (côté bas 2,5 → 0, côté haut 7,5 → 10) : gain 16 → 30 et 6 → 7 ; grave
  22 → 10 et 21 → 2 ; médium 26 → 3 et 34 → 35 ; aigu 11 → 33 et 31 → 4 ; présence 5 → 36 et
  28 → 23. Gains de H2 : 30, 16, 20, 6, 7. Paires gain de H4 : 39 → 30 et 1 → 7.

## Lectures

- **À 48 kHz** (chaque section décimée par 4, `resample_poly`) :
  - sondes gelées : FIR, Hammerstein 3 et 5 (`dev_select.json`), Volterra–Laguerre 2 et 3
    (`dev_select_v2.json`). Wiener–Hammerstein est exclu, comme dans `REPORT.md` ;
  - estimation A = `di_train` ; lecture B = `di_test` et bruit rose. `di_validation` n'est jamais
    utilisée : elle reprend les enregistrements de `di_train` ;
  - par sonde et segment : `d_norm`, part captée d_probe/d_audio, part non représentée, capacité
    r/RMS de chaque sortie rapportée à **son propre** RMS, indicateur « signature partielle »
    (capacité de l'une ou l'autre sortie strictement au-dessus de 0,3), référence quasi
    silencieuse, borne triangulaire. Les avertissements de capacité de la bibliothèque, qui
    rapportent aussi la sortie évaluée au RMS de la référence, sont retirés au profit de ces
    champs ;
  - ESR, magnitude, phase et enveloppe sur B, sans les 0,05 premières secondes ;
  - E2 (relâchement et attaque) sur les trois sections de paliers, aux instants de saut du layout.
- **Signature harmonique**, sur les trois sinus glissants synchronisés. La bande porte sur la
  fréquence de l'harmonique, en sortie :
  - **à 192 kHz** : ordres 1 à 9, nfft 32 768, bande 180–8 000 Hz, plancher −80 dB. C'est une
    **nouvelle version de protocole**. La version gelée (nfft 8 192) tronquerait à 192 kHz la
    fenêtre de 0,1 s (19 200 échantillons) à 43 ms ; la borne basse de 180 Hz garde la
    fondamentale de l'ordre 9 au-dessus des 20 Hz du balayage ;
  - **à 48 kHz, protocole gelé** (ordres 1 à 5, bande 100–8 000 Hz), pour comparaison avec la
    VS100R.

  Les deux sont rapportés par ordre et au total. Une référence sans fondamental donne « non
  mesurable » ; ce statut est conservé.
- **Multisinus**, à 192 kHz :
  - par prise : meilleure approximation linéaire aux raies excitées (fichier `.npz` à côté du
    JSON) ; niveaux impair et pair aux raies de détection, en dB par rapport aux raies excitées ;
    écart entre la première et la dernière période lue ;
  - par paire : écart relatif de cette approximation linéaire, et variation des niveaux impair et
    pair.
- **Retard commun par paire**, rapporté seulement : la valeur de la grille −3 à +3 échantillons à
  48 kHz (pas de 0,05) qui minimise l'ESR sur `di_test`. Il mélange un retard pur et la phase des
  filtres de tonalité. Une valeur au bord de la grille est signalée et ne se lit pas. Rien n'est
  recalé.

## Lectures préinscrites

**H1, monotonie le long des échelles.**
- **Mesure** : 5 boutons × 2 côtés × 5 métriques, soit 50 comparaisons. Les métriques sont l'ESR,
  la magnitude, la phase et le `d_norm` de Hammerstein 3 sur `di_test`, plus la signature
  harmonique totale à 192 kHz sur le sinus synchronisé à −10 dBFS.
- **Comparaison** : un signe compte si le score à |Δ| = 5 dépasse strictement celui à
  |Δ| = 2,5, du même côté. Une référence quasi silencieuse sur le segment (pour les quatre
  métriques lues sur `di_test`), un sinus sans fondamental (pour la signature harmonique) ou un
  rendu absent rendent la comparaison « non mesurée » ; ce n'est pas un échec.
- **Verdict** : « confirmée » à 45 signes ou plus ; « non confirmée » si plus de 5 comparaisons
  mesurées échouent ; « non concluant » sinon. Sur la JVM410H, toutes les échelles étaient
  monotones.
- **Portée** : un échec peut venir de la loi du bouton dans le plugin, pas seulement de la
  métrique. Aucune cause n'est attribuée.

**H2, capacité des sondes (question Q-0014).**
- **Mesure** : r/RMS de Hammerstein 5 et de Volterra–Laguerre 3 sur `di_test`, aux gains 0, 2,5,
  5 (prise 20), 7,5 et 10, chaque fois rapporté au RMS de la sortie concernée. Le seuil gelé est
  0,3.
- **(a) Refus** : un réglage strictement au-dessus de 0,3 déclenche la règle de refus. Les paires
  qui le touchent portent l'indicateur « signature partielle » pour cette sonde ; ESR, magnitude
  et phase restent les lectures globales.
- **(b) Croissance** : « oui » si au moins 3 des 4 pas croissent pour les deux sondes ; « non »
  sinon ; « non mesurable » s'il manque un point.

**H3, linéarité du baffle.**
- **Mesure** : un filtre est estimé sur A, de la sortie baffle coupé vers la sortie baffle actif
  (Welch, segments de 2¹⁷ à 48 kHz, soit 2,7 s ; division spectrale avec une garde numérique de
  10⁻¹⁰). Il est gardé bilatéral (retards de −1,37 s à +1,37 s, pour admettre une avance relative)
  et lu sur B, sans 2¹⁶ échantillons à chaque bord. ρ est le résidu relatif, le pire des trois
  paires et des deux segments.
- **Plancher de la méthode**, sur le substitut synthétique (baffle FIR connu de 150 ms) : ρ de
  0,005 à 0,006 sur `di_test`, de 0,004 à 0,005 sur le bruit rose (section « Essai à blanc »).
- **Verdict** : « baffle linéaire au sens de la méthode » si ρ ≤ 0,02, environ trois fois ce
  plancher ; sinon « linéarité non établie ». Un grand résidu ne se lit pas comme une
  non-linéarité : le biais de fenêtrage de Welch croît avec la longueur de la réponse, et un écho
  long, exactement linéaire, laisse lui aussi un grand résidu. Les deux causes ne sont pas
  séparées.
- **Rapporté à côté** : la part de l'énergie du filtre estimé au-delà de 0,25 s, indice d'une
  réponse longue.
- **Descriptif, à côté** : la part captée par le FIR gelé (256 coefficients, 5,3 ms) sur ces
  paires. Elle mesure son angle mort de mémoire longue.

**H4, spécificité de E2.**
- **Mesure** : à chaque niveau de paliers, E2 « relâchement » de chaque paire baffle est comparé
  au plus petit E2 « relâchement » des deux paires gain 0 et gain 10. Les 9 comparaisons sont
  requises ; s'il manque une paire, le verdict est « non mesurable ».
- **Verdict** : « spécifique » si le baffle est plus bas dans les 9 cas ; « co-réponse au
  baffle » si au moins un échec a un E2 baffle au-dessus de 10⁻³ dB (le plancher du banc est
  d'environ 6·10⁻⁵ dB) ; « non concluant » sinon (tous nuls, par exemple).
- **Attendu : « co-réponse au baffle ».** Sur le substitut, le baffle de 150 ms donne un E2 de 0,36
  à 1,7 dB, contre 0,002 à 0,056 dB pour le plus petit des deux gains (le gain 0 ; le gain 10 donne
  0,21 à 0,57 dB). La compression du substitut est faible. Le score E2 suit le gain à court terme
  y/x par fenêtres de 5 ms : une réponse longue étale chaque saut.
- **Lecture** : une co-réponse confirme, sur un baffle réaliste, celle déclarée pour l'écho au banc
  (0,66 à 0,81 dB). E2 n'isole alors pas la dynamique quand un filtre linéaire long est en jeu, et
  il se lit baffle coupé. « Spécifique » dirait que la compression de Nolly domine son baffle dans
  cet indice.

## Descriptif, sans seuil

- **Tableau principal** : métrique × bouton × |Δ| × côté × excitation (`di_test`, bruit rose,
  niveaux de multisinus, de sinus synchronisés et de paliers).
- Les prises « combinaison », les multisinus, les retards, et l'écart entre les deux protocoles
  harmoniques.

## Ce qui ne pourra pas être conclu

- **Le matériel** : c'est un logiciel. Rien sur un ampli réel, ni sur une chaîne d'acquisition.
- **Un modèle** : aucune comparaison neuronale, aucune qualité de modèle.
- **La détection**, si le rendu est déterministe : le plancher étant nul, tout écart non nul se
  voit. Seules les relations sont lisibles (ordre, nature, capacité, spécificité).
- **La généralité** : un seul ampli, un seul canal, une seule séance.
- **Les grandeurs physiques** : |Δ| est en graduations ; la loi des boutons du plugin est inconnue.

## Essai à blanc (substitut synthétique, avant tout rendu)

`nolly_standin.py` rend 16 prises avec une chaîne DSP fixe : filtres de tonalité, saturation avec
un léger tassement, filtre de présence, et un FIR connu de 150 ms comme baffle. Ce n'est pas un
modèle de Nolly. Les prises rendues sont les ancres 01, 20 et 39, les échelles de gain et
d'aigu, les prises 32 et 38, et les trois prises baffle. Le script définitif tourne en entier
dessus : 12 processus, environ 35 minutes. Il faut compter une heure pour les 42 rendus.

- **Portes** :
  - bouclage et stéréo exacts (écart 0) ; ancres identiques au bit ;
  - aucune prise muette ;
  - marqueurs largement au-dessus du silence (la sortie du substitut y est nulle, ou de l'ordre
    de 10⁻¹⁵ avec le baffle) ;
  - médiane des retards : 0.
- **Retards aux marqueurs** : de −8 à +17 échantillons selon le réglage, et environ 1 030, de
  polarité négative, avec le baffle. Ils suivent les filtres, pas une latence.
- **H1** : 20 signes sur les 20 comparaisons mesurées, 30 non mesurées (rendus absents). Verdict
  « non concluant », comme le veut la règle.
- **H2** : la capacité croît avec le gain (« oui ») et dépasse 0,3 dès le gain 5 ; les paires
  baffle portent l'indicateur « signature partielle » pour Hammerstein 5.
- **H3** : ρ de 0,0051 à 0,0055 sur `di_test` et de 0,0035 à 0,0048 sur le bruit rose ; énergie au
  delà de 0,25 s inférieure à 2·10⁻⁵. Verdict « baffle linéaire au sens de la méthode ».
  - Avec des segments de 2¹⁵, le même essai laissait 0,036 à 0,042 ; avec une estimation sur un
    seul bloc, 0,015 à 0,14.
  - Contrôle ponctuel, non archivé, sur des systèmes exactement linéaires (bruit blanc) : un écho
    de 0,2 ou de 0,5 à 1 s laisse ρ = 0,12 ou 0,28, donc « linéarité non établie », et non une
    non-linéarité ; une avance relative de 17 échantillons laisse ρ ≈ 1·10⁻⁴.
- **H4** : « co-réponse au baffle » sur les 9 comparaisons (E2 baffle de 0,36 à 1,7 dB, contre
  0,002 à 0,056 dB pour le plus petit des gains).
- **Part captée par le FIR gelé sur les paires baffle** : 0,45 à 0,61 sur `di_test`, 0,17 à 0,31
  sur le bruit rose.
- **Retard commun des paires baffle** : au bord de la grille, signalé.
- **Signature harmonique** : les deux protocoles sont mesurables sur toutes les paires ; totaux
  proches (0,795 contre 0,786 pour la prise 4).

Ces chiffres valident l'instrument, pas une attente sur Nolly. Le JSON de l'essai n'est pas commité.
Après cet essai, seuls ont changé le calcul des planchers de la branche non déterministe (un ordre
harmonique peut manquer à une répétition), qui ne s'exécute pas quand les ancres sont identiques,
et une docstring. Relectures Codex en lecture seule : trois passes, chaque point vérifié et corrigé.

## Écarts

### 2026-10-11, rendus réels et lecture

Les règles ci-dessus n'ont pas changé ; le script a tourné tel quel sur les 42 rendus. Résultats
et chiffres : `REPORT.md`, section « Sorties réelles Nolly ».

**Séance** (valeurs : `demo/capture/v2/SESSION_VALUES.md`) :
- **Déterminisme** : non atteint. Deux rendus du même réglage diffèrent, même d'une seule
  instance rendue deux fois. `SESSION.md` demandait de s'arrêter dans ce cas. La séance a
  continué sur décision de l'utilisateur, après diagnostic du worker. La lecture a pris la branche
  « non » de la porte 7, prévue pour ce cas : planchers tirés des trois paires d'ancres.
- **Suréchantillonnage du plugin** : NORMAL, choisi par l'utilisateur après ce diagnostic, au
  lieu du réglage HIGH de l'état de base proposé. `SESSION.md` demande seulement un mode fixe et
  noté.
- **Presets** : enregistrés comme chaînes d'effets REAPER (état complet de l'instance), le preset
  interne du plugin ne pouvant pas être enregistré par script.
- **Préférences REAPER** modifiées pendant la séance puis restaurées (moteur audio actif, médias
  gardés en ligne) : sans elles, des réglages n'atteignaient pas le plugin, ou une source hors
  ligne se rendait en silence. Aucun rendu gardé n'a été fait dans ces conditions.

**Lecture** :
- **8 prises exclues par la porte 4**, comme le prévoit la règle : 09, 12, 13, 18, 23, 24, 38 et
  41. Toutes échouent au seul critère « même retard aux deux marqueurs ». Les paires qui en
  dépendent ne sont pas lues, d'où :
  - H1 : les 5 comparaisons du côté haut de la présence (prise 23) sont non mesurées ;
  - H3 et H4 : « non mesurable », faute de la prise 41.
- **Contrôle post hoc, non préinscrit** (`nolly_marker_drift.py`, `nolly_marker_drift.json`) :
  la sortie aux marqueurs de fin, corrélée à la sortie aux marqueurs de début (même entrée), ne
  montre aucun décalage entier dans ±512 échantillons, sur les 42 prises :
  - 12 et 18 : deux échantillons adjacents presque ex æquo au sommet du pic de corrélation ;
    sortie aux deux marqueurs aussi proche que pour les ancres ;
  - 09, 13, 23, 24, 38 et 41 (présence 10, gain d'au moins 5) : le plugin répond autrement à la
    même entrée, précédée d'une histoire différente, avec un pic de la différence vers 1,2 kHz.
    La cause n'est pas identifiée.
  Le critère de retard est appliqué tel que préinscrit, sans que son échec identifie la cause de
  l'écart. La règle n'est pas modifiée.
- Ce contrôle sert à la condition de l'entraînement (`demo/nolly_models/predictions.md`,
  « Écarts »), pas à cette lecture.
