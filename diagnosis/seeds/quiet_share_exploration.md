# Exploration post hoc — la part de passages calmes de la partition de validation

**Statut : post hoc, non pré-enregistré.** Le pilote G a désigné une piste sans conclure
(`pilot_G_split_property.md`) : la proportion de trames sous −40 dB de la partition de validation
corrèle à +0,81 avec le log de l'ESR de test (p exact 0,022 ; corrigé de Šidák, 0,124). Ce fichier
consigne ce que la piste a donné. Environ quinze associations ont été examinées sur huit points ;
aucun seuil n'avait été fixé, donc **aucun libellé de verdict n'est employé ici**. Le fichier existe
pour deux raisons : garder trace des routes réfutées, qui empêchent un correctif évident et faux,
et fixer une prédiction avant que le pilote C ne réponde.

## L'association la plus forte du lot, et sa limite

La part calme prédit **la date de la première division du pas d'apprentissage** à ρ = −0,905
(p = 0,002, n = 8) : plus la validation contient de passages calmes, plus tôt `ReduceLROnPlateau`
divise. Et cette date ordonne l'ESR à ρ = −0,810.

| graine | part calme | 1ʳᵉ division | ESR |
|---|---|---|---|
| 42 | 0,109 | 285 | 0,0358 |
| 43 | 0,073 | 252 | 0,0744 |
| 44 | 0,163 | 161 | 0,0979 |
| 45 | 0,111 | 234 | 0,0832 |
| 46 | 0,028 | 255 | 0,0540 |
| 47 | 0,120 | 164 | 0,0628 |
| 48 | 0,152 | 167 | 0,1580 |
| 49 | 0,241 | 141 | 0,1781 |

Deux limites, immédiates. La date de division est **une issue du run**, pas une propriété de la
partition : elle n'était pas dans les sept propriétés déclarées, et la corréler relève de la même
exploration libre que le reste. Surtout, les trois quantités forment **un seul axe à n = 8** : les
corrélations partielles sont symétriques (part calme → ESR à date constante, +0,308 ; date → ESR à
part calme constante, −0,308). Huit points ne peuvent pas dire laquelle agit sur l'autre.

## Quatre routes candidates, toutes réfutées

Mesurées sur la fenêtre **fixe** 20–140 époques, commune aux huit runs — la division la plus précoce
tombe à 141, donc aucune fenêtre ne dépend de l'issue qu'elle sert à expliquer.

| route | part calme → route | route → 1ʳᵉ division | lecture |
|---|---|---|---|
| fluctuation de `loss/val/tot` | +0,31 (p 0,46) | −0,41 (p 0,32) | non détectée, direction compatible |
| dérive par fenêtre de patience sur bruit | −0,21 (p 0,61) | +0,41 (p 0,32) | non détectée, direction compatible |
| niveau de la courbe de validation | −0,45 (p 0,26) | +0,52 (p 0,18) | non détectée, direction compatible |
| part du terme MR-STFT dans la perte | +0,48 (p 0,23) | −0,45 (p 0,26) | écartée par la taille de l'effet |

La dernière ligne se traite à part : la part du terme MR-STFT ne varie qu'entre 0,086 et 0,091 sur
les huit graines. Sa corrélation de rang à +0,48 ordonne donc un intervalle de 6 % en relatif, ce qui
ne peut pas porter un écart d'ESR de 5 à 1. Écartée par la taille de l'effet, pas par le rang.

**Correction de méthode, et elle change la lecture.** Les trois premières lignes portaient d'abord
+0,05 et −0,10 — des nuls plats — et j'avais écrit « quatre routes réfutées ». La fluctuation était
estimée par le résidu d'un ajustement **linéaire**, ce qui sur une décroissance courbée compte la
courbure comme du bruit : 0,225 annoncé contre 0,080 réel, un facteur 2,8. Avec l'estimateur par
différences successives, insensible à toute tendance lisse et confirmé à 15 % près par une médiane
glissante, les trois routes remontent à |ρ| de 0,21 à 0,52, **toutes dans la direction attendue**,
aucune significative. La conclusion correcte n'est donc pas que les routes sont réfutées : c'est que
**huit graines n'en départagent aucune**. Une réfutation obtenue avec un mauvais estimateur était un
faux négatif.

Une cinquième mesure a été **écartée avant usage** : la « pente relative avant la première division »
que j'avais d'abord calculée était un artefact de fenêtre à longueur variable — une fenêtre courte
est dominée par la descente initiale raide, donc toute run divisant tôt paraît mécaniquement plus
pentue. Ce n'était pas une observation.

Conclusion de l'exploration : l'association tient, et **aucune route mesurable n'est départagée**. Le
pilote G quitte donc H2 pour retomber sur son statut épistémique — une corrélation sans mécanisme
identifié — d'une propriété vers la gauche. C'est un résultat modeste et il faut l'appeler ainsi.

## Ce qui, en revanche, n'est pas une corrélation

Sur cette même fenêtre fixe, la fluctuation relative de `loss/val/tot` vaut **0,080** en moyenne
(0,031 à 0,248 selon la graine), face au seuil relatif de `ReduceLROnPlateau` de **1e-4** : un
facteur 800. C'est la mesure de `plateau_margin.md` — le seuil est très au-dessous du bruit qu'il est
censé départager — étendue des enfants papillon d'une seule graine à **huit graines indépendantes**,
en mode déterministe. Ce fait ne dépend d'aucune sélection post hoc et vaut par lui-même.

**Une affirmation retirée.** J'avais d'abord lu les fenêtres de patience comme montrant que l'ESR de
validation progresse encore de 16 % au moment où la division se déclenche. Retirée : c'est une
différence d'extrémités sur une série dont la fluctuation sur la fenêtre fixe vaut 0,83 — dix fois
celle de la perte — avec un écart-type inter-graines de 0,36 sur la quantité elle-même.
Ininterprétable. Consigné ici pour que l'erreur, la même que celle déjà commise sur la marge de
déclenchement, ne soit pas refaite une troisième fois.

## La prédiction inscrite, vérifiée

`plateau_simulation.py` rejoue fidèlement `ReduceLROnPlateau`. **Contrôle d'abord** : rejouée sur
`loss/val/tot`, elle reproduit les huit vraies dates de première division **à l'époque près**, huit
fois sur huit. L'outil est donc exact, et ses verdicts sur d'autres quantités surveillées valent.

| quantité surveillée | premières divisions, graines 42 à 49 | médiane | écart-type |
|---|---|---|---|
| `loss/val/tot` (le protocole) | 285 252 161 234 255 164 167 141 | 200 | 55 |
| `loss/val/mrstft` | 359 252 161 234 255 164 167 141 | 200 | 73 |
| `loss/val/l1` | 23 88 161 21 21 139 23 22 | 23 | 59 |
| `metric/val/esr` | 23 61 348 21 95 25 23 47 | 36 | 111 |

**La prédiction se vérifie** : surveiller l'ESR de validation déclenche plus tôt (médiane 36 contre
200) et plus erratiquement (écart-type 111 contre 55, étendue 21 à 348). « Surveiller l'ESR » n'est
donc pas un correctif, et la sortie la moins chère en apparence est fermée.

Une observation non prédite, et le raisonnement faux qu'elle a d'abord provoqué. Rejouée sur le seul
terme MR-STFT, la simulation reproduit 7 des 8 dates de la perte totale ; sur le seul terme L1,
aucune. J'en ai d'abord conclu que L1 stagnait. **C'est faux** : L1 passe de 0,09 à 0,001, un facteur
90, son minimum tombant entre les époques 391 et 1094. Ce que la date de déclenchement mesure n'est
donc pas la progression mais la **régularité** — elle tombe à la première fenêtre de 21 époques sans
nouveau record, et L1, dont la fluctuation vaut 0,155 contre 0,076 pour MR-STFT, en rencontre une dès
l'époque 23. Le calendrier du protocole se trouve ainsi gouverné par le terme qui pèse 9 % de la
perte surveillée, parce qu'il est le plus lisse des deux. Observation consignée, mécanisme non
établi.

## Levier ou témoin ? L'ordre n'est pas déterminé

Si le sort de la run est déjà fixé quand le pas se divise, la date de division est un **témoin** et
non un **levier**, et la corriger ne servirait à rien. Les données existantes ne tranchent pas :
chez les enfants de la fourche 100 (`decided_when.json`), les paires aux plus grands écarts se
figent aux époques 98, 115, 176 et 191, tandis que leurs premières divisions tombent à 35 et 113 ;
au bras « décide » du pilote C, la première division tombe entre 141 et 285. Deux familles de runs
différentes, aucun ordre propre.

## Prédiction fixée avant la réponse

Le bras « fixe » du pilote C impose les mêmes huit époques de division aux huit graines **tout en
laissant à chacune sa propre partition**. Il retire donc le levier supposé sans toucher à la cause
supposée, ce qui en fait l'expérience discriminante de cette exploration — elle est déjà en cours,
et aucune run supplémentaire n'est nécessaire.

- **Si la date de division est un levier** : s(fixe) ≪ s(décide), et l'issue « soutenue » du
  pré-enregistrement du pilote C (rapport ≤ 0,50) se réalise.
- **Si elle n'est qu'un témoin** : s(fixe) ≈ s(décide), et la partition agit ailleurs — sur la
  trajectoire précoce, ou par le confondant ci-dessous.

Écrit avant la fin de la seizième run ; la date du commit en atteste. s(décide) = 0,535 est déjà
connu et n'est pas la quantité discriminante.

## Le confondant à nommer, et pourquoi aucune expérience n'est proposée

Une partition de validation plus riche en passages calmes en laisse **moins à l'entraînement** :
12 segments sur 130. Construire délibérément la partition — les 12 segments les plus calmes contre
les 12 les plus forts — changerait donc du même coup le signal surveillé et la composition de
l'entraînement, sans les séparer. Cette expérience n'est pas proposée : elle coûterait du GPU pour
un résultat inassignable, et le pilote C répond d'abord.

Une seule mesure à coût nul mérite d'être faite ensuite, et sa prédiction est posée d'avance :
rejouer `ReduceLROnPlateau` sur `metric/val/esr` au lieu de `loss/val/tot`, en réutilisant
`halvings()` de `plateau_simulation.py`. **Prédiction : le déclenchement sera plus précoce et plus
erratique**, le CV de l'ESR de validation valant 1,32 contre 0,23 pour la perte. Si elle se vérifie,
« surveiller l'ESR » n'est pas un correctif, et la sortie la moins chère en apparence est fermée.
