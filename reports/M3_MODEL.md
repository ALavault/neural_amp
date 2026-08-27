# M3 — Implémentation et validation de FSSR-NAM

## Architecture implémentée

Les variantes ont été construites dans l’ordre imposé:

- **S0**: FIR pré/post causaux de 17 taps et spline Hermite cubique C1 à 17 nœuds;
- **S1**: S0 plus GRU lent de dimension 8, mis à jour après chaque intervalle causal complet de 64 échantillons;
- **S2**: S0 plus TCN résiduel 8 canaux, noyau 3, dilatations `[1,2,4,8]` et champ réceptif de 31 échantillons;
- **S3**: cœur, état lent et résidu rapide;
- **S4**: S3 avec suréchantillonnage local x2 du seul résidu non linéaire `phi(x)-x`.

La spline utilise une extrapolation linéaire, une dérivée première et une primitive analytiques, une initialisation identité et aucune contrainte de monotonie. La sortie du résidu est initialisée à zéro, son échelle est bornée à 0,5 et sa pénalisation énergétique est explicite.

S0–S3 ont une latence algorithmique nulle. S4 utilise deux FIR Kaiser-sinc de 33 taps à 96 kHz et déclare 16 échantillons de latence à 48 kHz. Toutes ses voies sont alignées sur ce délai sans anticipation.

## Validation numérique

La suite de 64 tests vérifie notamment:

- causalité, reset et parité fichier/blocs irréguliers;
- identité et overfit d’un segment court;
- apprentissage de `tanh` et réapprentissage de l’identité;
- gradients et sorties finis;
- borne BIBO des FIR finis;
- continuité C1, dérivée et primitive de la spline;
- mise à jour lente indépendante des limites de blocs;
- délai et filtres x2;
- rechargement exact des états entraînés.

Les huit entraînements M3 utilisent des entrées train/validation issues de seeds génératrices distinctes. Ils sont diagnostiques, sur CPU, une seed, et ne soutiennent aucune affirmation physique.

| Cas | Variante | ESR validation | Ratio énergie résiduelle |
| --- | --- | ---: | ---: |
| `tanh` | S0 | 5,99e-5 | 0 |
| `tanh` | S3 | 4,43e-5 | 4,96e-5 |
| `tanh` | S4 | 6,71e-5 | 3,75e-5 |
| `slow_sag` | S0 | 4,97e-5 | 0 |
| `slow_sag` | S1 | 4,08e-5 | 0 |
| `slow_sag` | S3 | 4,00e-5 | 3,95e-5 |
| mémoire non linéaire courte | S0 | 5,35e-4 | 0 |
| mémoire non linéaire courte | S2 | 3,63e-4 | 1,09e-3 |

S1 améliore S0 de 18,0 % sur `slow_sag`; S2 améliore S0 de 32,0 % sur la mémoire courte. L’erreur maximale de rechargement, streaming, reset ou causalité est `4,77e-7`. Le résidu reste inférieur à 0,11 % de l’énergie de sortie dans tous les runs.

## Antialiasing local contrôlé

Sur le diagnostic 9 kHz à référence 192 kHz, la spline naïve produit `-19,56 dB` d’énergie parasite connue contre `-36,71 dB` pour S4, soit une réduction de `17,15 dB`. L’erreur complexe du fondamental est `1,64e-7`. Ce résultat démontre le fonctionnement numérique du bloc local; il ne valide pas H3 sur matériel ou musique.

## Budget théorique

| Variante | Paramètres | MAC linéaires/échantillon approximatives |
| --- | ---: | ---: |
| S0 | 71 | 34 |
| S1 | 410 | 38,5 |
| S2 | 905 | 826 |
| S3 | 1 244 | 830,5 |
| S4 | 1 244 | 962,5 |
| A2 Full | 12 145 | 11 777 |

Ces comptes excluent activations, additions, trafic mémoire et surcoûts d’appel. Ils autorisent M4 mais ne remplacent pas le futur benchmark C++ réel.

## Décision de gate

M3 passe: S3 apprend plusieurs systèmes synthétiques, S1 et S2 améliorent leurs cas discriminants, le résidu ne domine pas, S4 est causal et aligné, et le budget théorique reste inférieur à A2. M4 doit maintenant vérifier ces propriétés sur deux dispositifs internes et trois seeds. Les résultats externes restent inaccessibles.
