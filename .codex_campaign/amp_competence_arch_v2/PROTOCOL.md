# Protocole prospectif AMP-COMPETENCE-ARCH-v2

## Question

Le contrôle `micro_tcn_x2` peut-il d'abord démontrer une compétence minimale
sur les trois systèmes synthétiques, avec un budget de convergence choisi par
une règle gelée, avant qu'une différence relative entre architectures soit
interprétée ?

## Séquence fail-closed

Le contrat exécutable est `configs/amp_competence_arch_v2/protocol.yaml`. Le
preflight le copie exactement dans `PROTOCOL_LOCK.yaml` avant le premier run.
Le contrôle est entraîné de zéro pour trois seeds et observé aux six checkpoints
gelés. Le plus petit checkpoint qui passe strictement gain et corrélation sur
chaque système/seed n'est retenu que si le checkpoint suivant repasse les gardes
et confirme un plateau ESR médian entre 0 et 5 % d'amélioration.

Sans budget confirmable, la campagne se ferme en
`NO-GO-COMPETENCE-v2` et aucun run candidat n'est autorisé. Après passage
seulement, le contrôle et les quatre candidats gelés sont réentraînés de zéro
sur un second split synthétique disjoint avec exactement le même budget, les
mêmes exemples, seeds, loss et optimiseur.

## Frontières

Les artefacts et le verdict `NO-GO-ARCH` de v1 restent immuables et ne sont ni
repris, ni retunés, ni utilisés comme observations de sélection v2. Aucun audio
physique n'est adressable. Blackstar, UA1176 et `EXTERNAL_REPORT_ONLY` restent
verrouillés. Aucun dataset 192 kHz ni FM9 n'est admis.

## Décision

Une architecture n'est promue que si ses gardes passent partout, si son
amélioration ESR appariée médiane atteint 10 %, si la borne basse bootstrap à
95 % est strictement positive et si aucune médiane par système ne régresse de
plus de 5 %. Toute issue scientifique valide manquant une gate est un no-go ;
une corruption d'instrumentation ou de protocole est `INVALID`.
