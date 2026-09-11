# Piste d'architecture — conditionnement par états physiques

## Statut

Note prospective destinée au steering du prochain objectif d'architecture.
Elle ne constitue ni une décision de campagne, ni une modification d'un protocole
gelé, ni un résultat expérimental.

## Intuition

Un simulateur neuronal d'amplificateur pourrait bénéficier d'un espace de
coordonnées physiques explicite, de façon analogue à l'emploi de descripteurs
timbraux dans StyleWaveGAN. Dans StyleWaveGAN, les descripteurs conditionnent le
mapping et le style, et une loss différentiable impose leur cohérence dans le son
généré. Pour un ampli, l'analogue pertinent ne serait pas un vecteur timbral global,
mais une trajectoire causale d'états physiques à plusieurs échelles temporelles.

Source de l'analogie : [StyleWaveGAN, sections 2.1 et 2.5](https://arxiv.org/pdf/2204.00907).

## Architecture candidate

```text
entrée audio ───────────────────────► cœur audio rapide ─────► sortie
      │                                   ▲
      └► observateur physique causal ─────┘
          états lents, cadence réduite
```

Un petit observateur causal calculerait ou prédirait un `physical-state bus`, par
exemple :

- enveloppes rapides et lentes, énergie et crête par bandes ;
- niveau de drive effectif et compression dynamique ;
- déplacement de bias/DC et proxy de blocking distortion ;
- charge, récupération et sag d'alimentation ;
- réglages normalisés et, si justifié, famille de circuit.

Ces états moduleraient le cœur audio par FiLM, gates ou paramètres de splines, à
cadence réduite. Une branche résiduelle resterait libre de corriger les erreurs des
approximations physiques. L'antialiasing sélectionné resterait un mécanisme
orthogonal du chemin audio rapide.

## Séparation nécessaire des informations

1. Les features calculables causalement depuis l'entrée et les réglages peuvent
   être utilisées à l'inférence.
2. Les états internes non observables doivent être prédits par l'observateur ou
   fournis comme supervision par un simulateur/teacher pendant l'entraînement.
3. Les descripteurs dérivés de la sortie matérielle peuvent servir de cibles
   auxiliaires, mais jamais d'entrées à l'inférence afin d'éviter toute fuite de la
   cible.

## Hypothèse falsifiable

> À budget CPU, latence et capacité comparables, le conditionnement d'un cœur
> audio par des états physiques causaux multi-échelles améliore la fidélité des
> comportements dynamiques et l'extrapolation en niveau par rapport à un modèle
> audio-only et à un état lent entièrement latent.

## Ablation minimale proposée

- cœur audio seul ;
- cœur plus état lent appris sans supervision physique ;
- cœur plus features physiques déterministes ;
- cœur plus observateur appris, supervisé par des features physiques.

La quatrième variante est la candidate privilégiée : elle combine un biais
inductif interprétable avec la liberté nécessaire pour compenser une physique
incomplète. Les comparaisons devront égaliser paramètres, CPU et latence, et inclure
des transitions de niveau et conditions hors distribution plutôt que le seul ESR
in-distribution.

## Risques à tuer expérimentalement

- proxies physiques faux ou trop spécifiques à un circuit ;
- états non identifiables à partir des seules entrées/sorties ;
- features redondantes que le réseau ignore ;
- amélioration due uniquement à une capacité ou un coût supplémentaires ;
- fuite de cible par des descripteurs indisponibles en production.
