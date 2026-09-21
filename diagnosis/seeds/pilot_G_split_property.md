# Pilote G — une propriété de la partition de validation prédit-elle l'ESR final ?

**Pré-enregistré le 2026-09-22, avant toute mesure de corrélation.** Attaque la seule question que
`DIAGNOSIS_seeds.md` section 6 déclare ouverte sans mécanisme : « H2 sans mécanisme ».

## Question

Le pilote E a établi que la chaîne de données porte l'effet de graine : fixer la partition et
l'ordre des lots fait tomber l'écart-type de l'effet run de 0,375 à 0,191, soit le plancher du
non-déterminisme (`pilot_E_split.md`). Mais **on ne sait pas pourquoi une partition est meilleure
qu'une autre**. La seule piste existante est une corrélation sans mécanisme, choisie après coup :
le RMS minimal des douze segments de validation ordonne l'ESR de test 6 fois sur 6
(`DIAGNOSIS_seeds.md`, H2, six runs).

Deux choses ont changé depuis, qui rendent la question testable proprement. La partition est
**reproductible et vérifiée** — la perte de validation recalculée sur la partition dérivée d'une
graine reproduit celle du journal à 0,15 % près (`validation_noise.md`, `pilot_E_split.md`). Et le
pilote C fournit **huit graines** sous conditions identiques, en mode déterministe, là où H2 n'en
avait que trois.

**Question posée :** une propriété mesurable de la partition de validation prédit-elle l'ESR de
test final ?

## Dispositif

- **Mesure avancée.** Les propriétés d'une partition ne dépendent que de la graine, pas du run.
  Elles sont donc calculées et commitées **avant** que les ESR des graines 47, 48 et 49 n'existent,
  ce que l'ordre des commits atteste. Le prédicteur est figé avant l'issue pour trois graines sur
  huit.
- **Partitions** : reproduites en répétant l'ordre de construction du pilote — `seed_everything(s)`,
  puis construction du processeur, puis module de données — pour les graines 42 à 49. Contrôle :
  pour la graine 42, la partition obtenue doit être celle déjà vérifiée dans
  `validation_noise.py`.
- **Issue** : ESR de test du bras « décide » du pilote C, une run par graine, mode déterministe.
- **Propriété primaire, nommée d'avance** : le **RMS minimal** des douze segments de validation,
  celle que H2 désignait. Toute autre est exploratoire et sera rapportée comme telle.
- **Propriétés exploratoires** : RMS moyen, énergie totale, facteur de crête moyen, centroïde
  spectral moyen, proportion de passages sous −40 dB, et le RMS minimal des 118 segments
  d'entraînement — ce qui est retiré de l'entraînement compte peut-être autant que ce qui sert à
  valider.

## Prédictions, et elles partitionnent l'espace

Soit ρ la corrélation de rang de Spearman entre la propriété primaire et le log de l'ESR de test,
sur les huit graines.

- **H2 soutenue** : |ρ| ≥ 0,74, seuil auquel p < 0,05 à n = 8 en bilatéral.
- **H2 réfutée** : |ρ| < 0,50. La corrélation observée sur six runs ne survit pas à huit graines et
  à une partition reproduite, et doit être retirée du diagnostic.
- **Indécidable** : 0,50 ≤ |ρ| < 0,74 — la direction est là, la puissance ne suffit pas à n = 8, et
  le chiffre est rapporté tel quel.

Sur les propriétés exploratoires, aucune conclusion ne sera tirée d'un ρ isolé : avec sept
propriétés testées, la plus forte a une chance sur deux de dépasser 0,7 par hasard seul. Elles ne
servent qu'à désigner une piste pour un dispositif ultérieur.

## Ce que ce pilote ne pourra pas dire

- Huit graines, une architecture, un appareil, un enregistrement. Une corrélation sur huit points ne
  devient un mécanisme qu'en étant confirmée sur un autre matériel.
- Il ne sépare pas la partition de l'**ordre des lots**, que la graine tire aussi : une propriété de
  la partition qui prédirait l'ESR pourrait n'être qu'un marqueur de l'ordre des lots.
- Il ne dit rien de ce qu'une **bonne** partition serait : prédire n'est pas prescrire, et le
  pilote E a déjà montré que figer une partition fige aussi sa difficulté.
- La propriété primaire est définie sur la cible, pas sur l'entrée sèche. Le RMS minimal d'un
  segment de validation dépend donc de l'appareil modélisé autant que du signal de test.

## Relecture adverse (2026-09-22, avant tout calcul de corrélation)

Soumis à une relecture adverse par Codex, via le CLI. Six critiques, dont trois que je retiens
sans réserve. **Les prédictions et les seuils ci-dessus ne sont pas modifiés** : ce qui suit les
commente, le verdict les appliquera tels quels et rapportera ces réserves en regard.

**Retenu, et corrigé sur-le-champ.**

1. *Le script du verdict corrélait des champs dépendant du run.* Il parcourait toutes les clés
   communes du JSON, donc aussi `logged_val`, `recomputed_val` et `relative_gap`, qui viennent du
   run dont on veut prédire l'ESR. C'était circulaire, et c'est un vrai défaut, pas une question de
   style. Corrigé : seules les sept propriétés déclarées ici sont corrélées.
2. *Mon affirmation « trois graines partagent exactement le même RMS minimal » est fausse.* Deux
   graines seulement sont exactement égales (46 et 49, à 2,0671791571658105·10⁻⁴) ; la graine 43
   vaut 2,1135467977728695·10⁻⁴, une valeur différente que mon format d'affichage à cinq décimales
   arrondissait à la même chaîne. Sept valeurs distinctes sur huit, pas six. L'erreur venait de mon
   propre formatage et je l'avais propagée dans un message de commit.
3. *La taille réellement confirmatoire est trois, pas huit.* Le RMS minimal a été choisi parce que
   H2 le nommait, et H2 vient de six runs dont trois graines ici présentes ; cinq des huit issues
   existaient déjà au moment du pré-enregistrement. Le texte le disait — « figé avant l'issue pour
   trois graines sur huit » — mais le reste de la formulation laissait entendre davantage. À lire
   comme une confirmation prospective à n = 3, ce qui ne soutient rien à soi seul.

**Retenu, appliqué au verdict et non au pré-enregistrement.**

4. *« H2 réfutée » est trop fort à n = 8.* Un |ρ| sous 0,50 n'exclut pas une association réelle ;
   il constate une absence de détection. Le verdict emploiera le libellé pré-enregistré et lui
   adjoindra la lecture correcte : **non soutenue, non concluante**. Changer le libellé après coup
   reviendrait à réécrire une prédiction.
5. *L'écart du contrôle de reproduction atteint 1,19 %, quand le texte cite 0,15 %.* Le 0,15 %
   venait d'une vérification antérieure sur d'autres runs ; ici la plage observée est 0,2 à 1,2 %.
   Aucun seuil d'échec n'avait été fixé pour ce contrôle, ce qui est une négligence : une partition
   fausse donnerait ~30 %, donc 1,2 % reste concluant, mais le seuil aurait dû être écrit d'avance.

**Retenu partiellement.**

6. *La portée de l'hypothèse glisse.* L'échec du seul RMS minimal ne réfute pas « une propriété
   mesurable prédit l'ESR » ; et un seuil en valeur absolue autorise les deux directions alors que
   H2 permettait une prédiction signée. Juste sur les deux points. Le verdict ne conclura donc que
   sur la propriété primaire, et signalera que la direction n'avait pas été contrainte.
7. *La protection contre les tests multiples est rhétorique.* Exact : aucune correction n'est
   appliquée, seule une mise en garde est écrite. Elle suffit tant qu'aucune inférence n'en est
   tirée, ce que le verdict respectera en ne rapportant les exploratoires que comme descriptives.

**Non retenu.** Le contrôle par les poids finaux n'est pas circulaire pour la propriété primaire,
calculée uniquement depuis les cibles du découpage ; la relecture en convient.
