# Porte de reproductibilité du 2026-09-24

**Question.** Le banc (`scripts/product_nablafx_bench.py`) et `src/fssr_nam/models/ssm_wavenet.py`
ont changé depuis les runs de référence : contrôles FiLM, appareils paramétriques, `val_indices`.
En mode déterministe, une run relancée à graine égale est identique au bit
(`diagnosis/butterfly/determinism.json`). Une relance courte au code actuel doit donc reproduire
chaque valeur journalisée par sa référence sur les mêmes pas. Sinon, le changement de code est un
facteur confondant, et aucune comparaison appariée avec les runs anciennes ne tient.

**Dispositif.** Trois relances, une par chemin, avec les drapeaux relevés dans les
enregistrements de référence (`scripts/product_repro_gate_queue.sh`), `--no-record` :

| relance | référence (commit) | pas |
|---|---|---|
| `repro_gate_ssm_bigmuff` | `pilotC_decide_seed42` (189474d) | 200 |
| `repro_gate_s4tf_bigmuff` | `paired_s4_tf_l_16_s925` (6112d6f) | 200 |
| `repro_gate_ssm_marshall` | `param_ssm_wavenet_marshall` (29a869c) | 300, soit deux points de validation |

Les relances ont tourné du 2026-09-24 à 10:55 au même jour à 11:22, sur un code identique à celui
du commit a950cb0 pour le banc et le modèle. Aucun commit de code n'a eu lieu entre-temps.

Comparaison par `diagnosis/seeds/repro_gate.py` : les lignes de `metrics.csv` sont comparées
valeur par valeur, sous forme de chaînes. Ce sont des repr exactes de flottants, donc l'égalité de
chaînes est une égalité au bit. Sont comparés : la perte d'entraînement, les normes de gradient
par paramètre, le learning rate et toutes les métriques de validation.

**Résultat : porte franchie, les trois chemins sont identiques au bit.**

| chemin | lignes | valeurs comparées | écarts |
|---|---|---|---|
| SSM-WaveNet, Big Muff | 87 | 865 | 0 |
| S4-TF-L-16, Big Muff | 87 | 881 | 0 |
| SSM-WaveNet-FiLM, Marshall | 10 | 348 | 0 |

Les bascules de la garde de polarité tombent dans la fenêtre du chemin SSM Big Muff (époques 7 à
9) ; les pertes de validation qui les suivent sont identiques.

**Une ligne par relance est écartée, et rapportée.** Arrêtée par `--max-steps`, une run clôt
son époque en cours et journalise la perte d'entraînement moyennée sur ses seuls premiers pas.
La référence journalise la même ligne à la fin de l'époque complète, au-delà de la fenêtre.
Ces lignes ont été repérées à la première comparaison ; elles sont listées dans
`repro_gate.json`.

**Environnement** (`repro_gate.json`) : torch 2.13.0+cu130, CUDA 13.0, cuDNN 9.2, lightning 2.6.1,
pilote 595.84, RTX PRO 4000 Blackwell. Lightning et ses voisins sont installés hors de `uv.lock`.

**Conséquences.**
- Les runs nouvelles de ces trois chemins se comparent aux anciennes sans facteur de code : c'est la
  condition du descriptif sur seize graines du pilote C
  (`diagnosis/butterfly/pilot_C_extension.md`).
- Le bras jointe-activations du Marshall (`param_ssm_wavenet_marshall`) peut servir de bras de
  G2.1 sans être relancé.
- Ce que la porte ne couvre pas : les pas au-delà de 300, le chemin S4-param, et les changements
  de code postérieurs à a950cb0. Ceux-ci sont couverts pour le CPU par
  `tests/regression/test_ssm_wavenet_bitwise.py`, pas pour le GPU.
