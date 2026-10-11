# Consigne pour l'agent local qui pilote REAPER (séance de capture v2, Nolly)

Tu es un agent qui travaille sur la machine de l'utilisateur, Windows ou macOS, avec REAPER et
le plugin Neural DSP Archetype: Nolly (ou Nolly X, selon ce qui est installé). Ta tâche :
produire les rendus de la séance de capture v2, ampli **Rhythm**, exactement comme le décrit
`SESSION.md`, et prouver qu'ils sont conformes.

Un serveur lira ces rendus avec des règles figées avant toute donnée. Un écart de format, de
gain, de durée ou de réglage invalide toute la séance, sans possibilité de le corriger après
coup. Tu ne dois donc rien improviser : en cas de doute, arrête-toi et demande à l'utilisateur.

## Fichiers fournis (dans ce dossier)

- `reamp_v2_192k.wav` : la source, 86 713 203 échantillons (451,631 s), 192 kHz, 24 bits,
  mono. Son sha256 est dans `SHA256SUMS`.
- `SESSION.md` : la feuille de séance. Elle donne l'ordre, l'état fixe du plugin, les
  pré-contrôles et le tableau des 42 prises. **C'est la référence ; ce texte-ci l'outille.**
- `settings.csv` : les 42 prises (`order, block, G, B, M, T, P, cab, split, setting, file`).
  Les boutons vont de 0 à 10.
- `layout.json` : les sections de la source, en échantillons à 192 kHz. Les contrôles
  ci-dessous en ont besoin.

## Interdits

- Ne jamais modifier, rééchantillonner, normaliser ni recopier autrement la source.
- Pour faire passer un contrôle, ne jamais toucher au gain d'item, au fader, au pan, au master ni
  aux gains du plugin. Seuls les **réglages de rendu et de routage** peuvent changer, et ils sont
  ensuite identiques pour tous les rendus.
- Aucun effet hors du plugin, aucun effet sur le master, aucune enveloppe d'automation, aucune
  modulation ni liaison de paramètres (MIDI, apprentissage, liens), aucune télécommande active
  pendant les rendus.
- Dans le plugin, couper le noise gate, toutes les pédales, l'EQ graphique, le doubler, le delay,
  la reverb, la transposition et l'accordeur. Le baffle (cab) reste coupé, sauf pour les prises
  40, 41 et 42.
- Aucun décalage temporel manuel : ni décalage de piste, ni décalage d'item, ni « ignorer la
  PDC ».
- Ne rien installer sans l'accord de l'utilisateur. Ne rien envoyer hors de la machine : c'est
  l'utilisateur qui transfère les fichiers.
- Ne rien deviner. Si une porte échoue, s'il manque une information (nom d'un paramètre, plugin
  qui refuse 192 kHz…) ou si une valeur ne peut pas être réglée exactement : arrête-toi, montre la
  preuve et demande.

## Méthode

Pilote REAPER par ReaScript (Lua, ou Python si REAPER est configuré pour), plutôt qu'à la souris.
Tu peux ainsi lire et écrire les paramètres du plugin et tout journaliser. Conserve chaque script
et chaque journal dans `nolly_logs/`. Écris les vérifications en Python avec numpy et soundfile si
l'utilisateur l'accepte, sinon avec sox.

Une comparaison « identique » porte toujours sur les **échantillons décodés et leur nombre**,
jamais sur les fichiers WAV ou leurs empreintes, qui contiennent des métadonnées.

**Étape 0 — Source.** Calcule le sha256 de `reamp_v2_192k.wav` et compare-le à `SHA256SUMS`. S'ils
diffèrent, arrête-toi.

**Étape 1 — Projet.**
- Projet neuf. Fréquence du projet fixée **et forcée** à 192 000 Hz : par ReaScript,
  `PROJECT_SRATE = 192000` **et** `PROJECT_SRATE_USE = 1`. Relis les deux valeurs.
- Aucun suréchantillonnage ajouté par REAPER, ni par instance ni par chaîne d'effets. Celui du
  plugin est un réglage du plugin, à noter.
- Compensation de latence (PDC) active, pour le projet et pour chaque chaîne d'effets.
- Auto-bypass des effets sur silence **désactivé**, dans les préférences et pour le plugin.

**Étape 2 — Pistes et routage.**
- Piste « source » : la source en item à la position 0, volume d'item 0 dB, sans fondu, sans
  changement de vitesse ni de hauteur. Piste à 0 dB, pan centré ; master à 0 dB, sans effet.
- Pour chaque prise, une piste qui porte **sa propre copie de l'item source**, à la position 0,
  et **sa propre instance** du plugin. Pas d'envoi, pas de dossier, pas de double alimentation.
- Nombre de canaux des pistes, broches d'entrée et de sortie du plugin, et loi de panoramique du
  projet : fixés une fois, puis identiques partout. Plugin actif, jamais contourné ; son mélange
  sec/traité, s'il existe, à 100 % traité.

**Étape 3 — Réglages de rendu communs**, fixés une fois pour toutes :
- WAV, 32 bits flottants, 192 000 Hz ;
- mono, sauf pour les contrôles stéréo ;
- ni dither, ni noise shaping, ni normalisation, ni limiteur ;
- ni fondu, ni retrait du silence de début ou de fin, ni ajout de remplissage, ni seconde passe ;
- bornes : tout le projet ;
- rendu hors ligne pleine vitesse ;
- motif de nom de fichier `$track`, sans suffixe automatique, sortie dans le dossier `nolly/`.

Vérifie la longueur en échantillons, jamais en secondes arrondies : chaque rendu fait au moins
86 713 203 échantillons, et la source finit par 6 s de silence pour la queue.

**Étape 4 — Contrôle 0, bouclage (condition d'arrêt).**
- Rends la piste source **sans le plugin**, avec les réglages de l'étape 3, sous le nom
  `nolly/loopback_software.wav`.
- Vérifie par script :
  - format WAV flottant, mono, 192 000 Hz ;
  - au moins 86 713 203 échantillons, et uniquement des zéros au-delà ;
  - écart maximal |rendu − source| au plus 2⁻²³ (un pas de 24 bits) sur toute la source.
- S'il échoue, cherche la cause **dans les réglages de rendu et de routage seulement** : nombre de
  canaux de la piste, réduction stéréo vers mono au rendu, loi de panoramique du projet, dither.
  Par exemple, une réduction en mono qui additionne deux canaux identiques double le signal.
  Refais le contrôle. Si tu ne trouves pas, arrête-toi.
- **Rien d'autre n'est rendu avant que ce contrôle passe.** Note la configuration retenue dans
  `nolly_logs/render_log.json`.

**Étape 5 — État de base du plugin.**
- Inventaire complet des paramètres : nombre, nom, valeur normalisée et valeur affichée de chaque
  paramètre, dans `nolly_logs/params_inventory.json`.
- Établis l'**état de base complet** avant tout pré-contrôle :
  - ampli Rhythm sélectionné ;
  - G, B, M, T et P de l'ampli Rhythm à 5, et **seulement eux** ;
  - toutes les autres commandes de l'ampli fixées et notées, par exemple, selon la version,
    RESONANCE, MASTER, OUTPUT, POWER, INPUT MODE (le mode d'entrée peut ne pas être restauré par
    un preset : vérifie-le après chaque chargement) ;
  - gains d'entrée et de sortie, mode de qualité ou de suréchantillonnage du plugin ;
  - modules coupés (liste des Interdits) ;
  - baffle coupé ; pour le bloc cab on, baffle, micros, niveaux, panoramiques et polarités des
    micros fixés une fois, et envoi vers la réverbération de pièce (ROOM SEND ou équivalent) **à
    zéro**.
- Fais valider par l'utilisateur la correspondance des commandes et l'état de base avant de
  continuer. Toutes les instances partent ensuite de cet état, cloné, puis reçoivent les réglages
  de leur prise.

**Étape 6 — Réglage exact des boutons.**
- Table unique `(paramètre, cible) → valeur normalisée`, établie une fois et réutilisée pour
  toutes les prises, ancres comprises. Cibles : 0, 2,5, 5, 7,5 et 10. Si l'affichage est en
  pourcentage, 10 = 100 %.
- Pour chaque entrée, vérifie que l'affichage croît avec la valeur normalisée. Teste les bornes
  0 et 1 directement. Cherche ensuite par dichotomie, en un nombre d'itérations borné, la valeur
  normalisée dont l'affichage égale **exactement** la cible, et retiens le milieu de l'intervalle
  qui donne cet affichage.
- Si une cible n'est pas atteignable exactement (pas d'affichage trop grossier, loi du bouton),
  arrête-toi et demande.

**Étape 7 — Pré-contrôles du plugin** (SESSION.md), sur l'état de base :
- Le plugin se charge et rend à 192 kHz. S'il refuse, arrête-toi : il faudra une variante à
  48 kHz préparée côté serveur.
- **Latence** : relève la PDC du plugin telle que REAPER l'indique, et vérifie qu'elle est
  compensée (contrôle des retards à l'étape 9).
- **Déterminisme** : rends le centre deux fois, dans deux pistes qui ont chacune leur propre
  instance, puis compare les échantillons. On attend une identité exacte. Sinon, vérifie les
  réglages ; si l'écart persiste, arrête-toi et garde les deux rendus pour l'utilisateur. Ces
  deux rendus vont dans `nolly_logs/`, pas dans `nolly/`.
- **Stéréo, baffle coupé** : rends le centre une fois en **stéréo**, sans réduction préalable, sous
  le nom `nolly/center_stereo.wav`. Vérifie que les canaux gauche et droit sont égaux à 2⁻²³ près
  et que le centre mono vaut exactement ce canal.
- **Stéréo, baffle actif** : rends une fois en stéréo le centre avec le baffle du bloc cab on,
  dans `nolly_logs/center_cab_stereo.wav` (contrôle local, pas une porte du serveur). Vérifie
  L = R de la même façon.
- Si un contrôle stéréo échoue, un étage stéréo est resté actif (doubler, panoramique de micros,
  réverbération de pièce…) : arrête-toi.
- Repasse ensuite le rendu en **mono**, puis relis les réglages de rendu.

**Étape 8 — Les 42 prises.**
- Une piste par prise, dans l'ordre de `settings.csv`. Chaque piste porte exactement le nom de la
  colonne `file`, sans `.wav`.
- Baffle coupé partout, sauf les prises 40 à 42, qui reprennent le bloc cab on de l'étape 5.
- Pour chaque prise, **obligatoirement** :
  - le preset enregistré sous le nom du fichier ;
  - une capture d'écran du plugin ;
  - un relevé complet des paramètres (nom, normalisée, affichage) dans
    `nolly_logs/params/<file sans .wav>.json`.
  
  Si l'une de ces preuves est impossible, signale-le explicitement.
- Vérifie qu'aucune enveloppe ni modulation n'est active sur les pistes ou le plugin.
- Sélectionne **seulement** les 42 pistes de prises. Lis la liste des chemins qui seront produits
  (`RENDER_TARGETS`) : 42 chemins `nolly/<file>`, sans suffixe ni double extension. Rends ensuite
  en une passe.

**Étape 9 — Vérifications après rendu**, par script, résultat dans `nolly_logs/checks.json`, une
ligne par fichier. Elles reprennent les portes figées du serveur.

Pour chaque prise de `nolly/`, avec s le début d'une section dans `layout.json` et L sa longueur :
- **Fichiers** : les 42 noms exacts de `settings.csv`, plus `loopback_software.wav` et
  `center_stereo.wav`.
- **Format** : WAV flottant, 192 000 Hz, mono (stéréo pour `center_stereo.wav`), valeurs finies,
  au moins 86 713 203 échantillons.
- **Retard aux marqueurs** : pour `markers_start` puis `markers_end`, prends la fenêtre de sortie
  de s − 4 096 à s + L + 4 096 et calcule sa corrélation croisée (mode « valid ») avec la source
  de s à s + L. Le retard vaut l'indice du maximum de |corrélation| moins 4 096. Il doit être :
  - le même aux deux marqueurs ;
  - strictement à l'intérieur de la recherche (indice ni 0 ni le dernier).
  
  Note aussi le signe de la corrélation (polarité).
- **Marqueurs au-dessus du silence** : le RMS de chaque section de marqueurs dépasse d'au moins
  20 dB celui de la section `silence`.
- **Paliers non muets** : les sections `steps_-30dBFS_rms`, `steps_-24dBFS_rms` et
  `steps_-18dBFS_rms` sont découpées par leurs `onsets` en segments alternés, dont les **segments
  d'indice impair** sont les paliers faibles, 66 en tout. Le RMS de **chacun** doit être
  strictement au-dessus de −100 dBFS. Un palier muet signifie presque toujours un noise gate
  resté actif : arrête-toi.
- **Latence** : la médiane des retards des prises sans baffle vaut au plus 1 024 échantillons en
  valeur absolue. Au-delà, la latence n'a pas été compensée : arrête-toi.
- **Ancres** : les prises 01, 20 et 39 passent toutes les portes ci-dessus. Leur identité
  échantillon par échantillon est rapportée ; ce n'est pas une porte.
- **Empreintes** : sha256 de tous les rendus dans `nolly_logs/renders_sha256.txt`.

Une prise qui échoue à une porte doit être expliquée et rendue de nouveau avant tout envoi.

**Étape 10 — Valeurs de la séance.** Remplis le tableau « Valeurs fixes » de `SESSION.md` dans
`nolly_logs/session_values.md` :
- version et format du plugin, système, version de REAPER ;
- commande de l'ampli pour G, B, M, T et P, et autres commandes avec leur valeur fixe ;
- gains d'entrée et de sortie, mode de qualité, PDC ;
- résultat du test de déterminisme ;
- baffle, micros, niveaux, panoramiques et polarités du bloc cab on.

**Étape 11 — Rapport à l'utilisateur** :
- ce qui a été rendu ;
- le résultat de chaque contrôle, avec les chiffres ;
- tout écart à `SESSION.md` et sa raison ;
- les fichiers à envoyer : d'abord `loopback_software.wav` et `center_stereo.wav`, puis les 42
  prises et `nolly_logs/`, selon les commandes du LISEZMOI.

Tu ne transfères rien toi-même.

## Place et durée

Il faut compter :
- 347 Mo environ par prise mono, soit environ 14,6 Go pour les 42 ;
- environ 1,04 Go pour le bouclage et le contrôle stéréo ;
- environ 1,4 Go de plus dans `nolly_logs/`, pour les deux centres du test de déterminisme et le
  contrôle stéréo baffle actif.

Le rendu est hors ligne ; sa durée dépend du mode de qualité du plugin.
