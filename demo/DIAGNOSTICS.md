# Diagnostics SSM-WaveNet

## Verdicts

**Bandes :** le residu SSM-WaveNet est concentre dans la meme bande qu'A2 (1000-4000 Hz, 50.4% du residu). Le gain est uniforme a travers les bandes (2.3x a 3.7x).

**Robustesse :** a lr 0,01, 2 graines sur 3 convergent (ESR test moyen des convergees 0.02466) ; la graine 2 diverge (ESR test 1.4947). Sur les 3 graines, ESR moyen 0.5147 +/- 0.6930. Le 5x ne tient pas : le modele n'est pas stable a ce lr. Relancee a lr 0,005, la graine 2 atteint 0.02452, mais ce n'est plus la meme configuration.

**Splits M4 :** aucune des 2 tentatives de SSM-WaveNet ne converge (voir tableau), contre 0.18825 (A2) et 0.18655 (S4-TFiLM) sur les memes fichiers. Le 5x ne tient pas hors de la prise d'entrainement dans cette boucle. Ces fichiers sont tronques a 120/30/30 s : la comparaison au protocole publie est dans `demo/nablafx_bench/`.

## 1. Residus par bande (Big Muff resplit, test)

| Bande | A2 part | A2 ESR | SSM part | SSM ESR | Gain |
| --- | ---: | ---: | ---: | ---: | ---: |
| 20-200 Hz | 1.0% | 0.00723 | 0.9% | 0.00227 | 3.2x |
| 200-1000 Hz | 16.3% | 0.02959 | 13.5% | 0.00797 | 3.7x |
| 1000-4000 Hz | 54.7% | 0.07911 | 50.4% | 0.02369 | 3.3x |
| 4000-12000 Hz | 23.1% | 0.14397 | 30.6% | 0.06197 | 2.3x |
| 12000-24000 Hz | 4.4% | 0.44997 | 4.5% | 0.14881 | 3.0x |

## 2. Robustesse inter-graines (Big Muff resplit, lr 0,01, 15k pas)

| Graine | Run | Statut | ESR test |
| ---: | --- | --- | ---: |
| 0 | `ssm_wavenet_b8_c16_s4` | converged | 0.02690 |
| 1 | `ssm_bigmuff_resplit_seed1` | converged | 0.02243 |
| 2 | `ssm_bigmuff_resplit_seed2` | diverged | 1.49470 |
| **moyenne** | | | **0.51468 +/- 0.69298** |

Hors configuration : graine 2 relancee a lr 0,005, ESR test 0.02452.

## 3. Splits M4 (fichiers publies tronques a 120/30/30 s)

| Modele | lr | Statut | Pas | ESR test | Meilleure ESR val |
| --- | ---: | --- | ---: | ---: | ---: |
| SSM-WaveNet | 0.01 | diverged | ? | 1.94506 | ? |
| SSM-WaveNet | 0.005 | killed | 4800 | non teste | 1.47276 |
| S4-TFiLM large | 0.01 | converged | 15000 | 0.18655 | |
| NAM A2 Full | | converged | | 0.18825 | |
