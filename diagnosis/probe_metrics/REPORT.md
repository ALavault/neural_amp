# Sondes d'analyse comme métriques de fidélité : rapport de validation

Voie produit, aucune revendication scientifique ni perceptive. Protocole, écarts et règles de
lecture : `NOTE.md` (tout ce qui est préinscrit y est daté avant les chiffres). Tableau principal :
`table_test.csv` (**métrique × défaut × sévérité × condition d'excitation**) ; synthèse :
`summary_test.json` ; figures : `figures/`. **BROUILLON : sections DI, Volterra–Laguerre,
Wiener–Hammerstein et sorties réelles en attente des passes en cours.**

## Ce que recouvre chaque type de résultat

| Type | Contenu | Où |
|---|---|---|
| Exécuté, synthétique | 13 défauts DSP connus × 3 sévérités × 6 familles d'excitation × 3 niveaux × 3 graines ; sondes FIR, Hammerstein 3 et 5 (v1), Volterra–Laguerre 2 et 3, Wiener–Hammerstein (v2) ; ESR, MR-STFT, IMD, signature harmonique | `calibration_test_*.csv` |
| Exécuté, réel | VS100R (envoi FX) : 3 répétitions, 12 contrôles négatifs, 8 changements réels ; JVM410H : échelles d'un bouton | `real_pairs_*.csv` |
| Hypothèses | Lectures marquées « post hoc » ou « exploratoire » ci-dessous | — |
| Non mesurable sans nouvelles captures | BLA stricte sur le réel, phase harmonique synchronisée sur la VS100R, niveaux hors des 3 sinus glissants, répétabilité d'un jour à l'autre, > 20/24 kHz, sortie après l'étage de tonalité | `manifest.json` |

## Contrôles mathématiques

- Borne |d_audio − d_probe| ≤ r_ref + r_eval : **0 violation** sur 1 755 lignes par sonde v1.
- Sondes linéaires en paramètres : fit(Δy) = Δθ et d_probe² = Δθᵀ G_B Δθ vérifiés numériquement
  (tests `test_linear_equivalence_and_gram_norm`, `test_volterra_fit_is_linear_in_the_output`).
- Contre-exemple F = x + 0,1 x³ / G = x + 0,2 x³ : écart d'erreur de fit < 1e-6 relatif, score
  d_probe = 0,1 RMS(x_B³) à 1e-4 près, porté par la seule branche d'ordre 3.
- Wiener–Hammerstein : deux reparamétrisations exactes (signe, gain) donnent un rendu identique
  au bit pour des paramètres éloignés ; entre départs, les paramètres diffèrent ~30 fois plus que
  les rendus. Aucune métrique ne compare donc des paramètres.
- Emboîtement Hammerstein / Volterra–Laguerre mesuré : absent dans les deux sens (NOTE, v2).

## Résultats sur défauts connus (v1 ; v2 à compléter)

Lecture préinscrite : rapport de détection R = score / plancher de répétition (même chaîne, deux
bruits à −80 dB) ; « détecté » si R ≥ 10, « non détecté » si R < 3. ESR et IMD lus en amplitude.
Ajouts post hoc signalés : part captée d_probe / d_audio et profils relatifs à d_audio.

### Quels défauts chaque métrique détecte-t-elle ?

À sévérité maximale et niveau médian, toutes familles confondues (défauts × familles) :
FIR 63 détectés / 2 limites ; Hammerstein 3 et 5 : 63 / 1 / 1 ; ESR et d_audio 59 / 1 / 5 non
détectés ; MR-STFT 7 / 6 / 52 non détectés ; IMD (deux tons) 13 / 13 ; signature harmonique (sinus
glissant) 12 / 13.

**Le seul seuil de détection ne départage pas les sondes.** Leur plancher est très bas parce
qu'elles ne représentent pas le bruit blanc : elles « détectent » ainsi presque tout, y compris un
écho hors de leur mémoire. Ce qu'elles voient vraiment se lit dans la **part captée**
(d_probe / d_audio, médiane sur les familles, niveau médian, sévérité maximale) :

| Défaut | FIR | H3 | H5 |
|---|---|---|---|
| gain, coloration, retard, passe-tout, préfiltre, redistribution à THD égal | 0,97–0,99 | 1,00 | 1,00 |
| relâchement d'enveloppe | 0,96 | 0,98 | 0,98 |
| terme cubique | 0,79 | 0,99 | 0,99 |
| terme quadratique | 0,67 | 0,92 | 0,92 |
| interaction x[n]·x[n−5] | 0,47 | 0,83 | 0,83 |
| ordre au-delà du degré (u⁷) | 0,36 | 0,77 | 0,97 |
| écho à 30 ms (hors mémoire) | 0,10 | 0,14 | 0,16 |
| bruit additif | 0,03 | 0,06 | 0,08 |

- **Signature harmonique D_k** : spécifique par ordre. Le défaut quadratique ne touche que
  l'ordre 2, le cubique les ordres 1 et 3 ; elle sépare défauts pairs et impairs, ce qu'aucune
  métrique globale ne fait.
- **FIR** : voit les changements non linéaires seulement par leur effet sur la réponse
  linéaire équivalente au niveau de l'excitation (part captée 0,47 à 0,79).

### Quels défauts manque-t-elle ?

- **Bruit additif et écho hors mémoire** : non représentés par les sondes (part captée ≤ 0,16 sur
  bruit coloré). Content-dépendant : sur plucks (notes périodiques), l'écho est capté à ~0,5,
  vu comme un filtre en peigne.
- **Non-linéarité préfiltrée** (terme sur u = passe-haut(x)) : sur bruit coloré, Hammerstein
  laisse ~0,47 de l'écart non représenté ; (PH ∗ x)³ n'est pas une somme de xᵏ filtrés.
- **MR-STFT (`metrics/spectral.py`)** : aveugle au passe-tout et au retard (magnitude seule) et
  dominée par un bruit à −80 dB dans les bandes où la référence est faible (log-magnitude) ; 52
  angles morts sur 65.
- **Ordres harmoniques individuels** : chacun manque les défauts des autres ordres, par
  construction ; c'est le total ou l'ensemble des ordres qui se lit.

### L'extraction est-elle plus stable que les écarts recherchés ?

Oui, sur ce banc : écart-type du log du score entre graines (indépendantes) médian ≤ 0,007 pour
toutes les métriques (90e centile 0,14–0,31) ; la signature harmonique est quasi déterministe
(D_k est une propriété du système). Wiener–Hammerstein fait exception : dispersion entre
départs 0,09 du RMS de la sortie, comparable aux petits défauts (voir v2).

### Que révèlent les sondes au-delà des métriques existantes ?

- **Robustesse au bruit de mesure** : le bruit additif n'entre presque pas dans le score
  (part captée 0,03–0,08), alors que l'ESR et d_audio le comptent entièrement.
- **Sensibilité aux écarts structurés sous le bruit** : un terme u⁷ dont la différence brute est à
  peine au-dessus du plancher (ESR ×1,75 sur deux tons) est vu à ×100–400 par les sondes, qui
  moyennent sur toute l'estimation et rejettent ce qui est hors de leur espace (comme une
  détection synchrone). Ce gain dépend de la capacité de la sonde.
- **Nature de l'écart** : la décomposition par branche (H5) range les défauts linéaires à
  l'ordre 1, le quadratique à l'ordre 2, le cubique aux ordres 1 et 3, u⁷ à l'ordre 5. **Mais**
  l'interaction x[n]·x[n−5] est attribuée à l'ordre 2 sur bruit coloré comme sur plucks : une
  excitation corrélée sur 5 échantillons ne sépare pas « carré » et « produit retardé ». La
  décomposition est conditionnée par la base et par l'excitation.
- **Complémentarité mesurée** (profils relatifs à d_audio, post hoc) : H3 ≈ H5 (ρ = 0,96) ;
  ESR et H5 non corrélés (ρ ≈ 0), l'ESR insistant sur bruit et écho hors mémoire ; MR-STFT
  séparée par le passe-tout et le retard.

### Quand refuser de conclure ?

- Part non représentée ou r / RMS(y) élevés (avertissement de capacité) : un faible score n'est
  pas une garantie de fidélité.
- Contributions par branche qui se compensent (somme > 2,5 × le score ; seuil exploratoire,
  choisi après l'étalonnage) : décomposition non identifiable (bruit : 3,7–4,6 ; écho : 3,7–8,7).
- Dispersion entre initialisations comparable à l'écart (Wiener–Hammerstein).
- Référence quasi silencieuse.
- Excitation absente pour la métrique (« non mesurable »), ou lecture hors du domaine excité
  (deux tons d'estimation et de lecture différents : corrigé, voir NOTE).
- Entrées non communes entre deux prises réelles (comparaisons directes d'enregistrements).

### Ensemble minimal (provisoire)

À confirmer après v2 et le réel : FIR (repère linéaire), Hammerstein 3 (H5 redondant ici),
signature harmonique par ordre sur sinus glissant, ESR (ou d_audio) pour ce que les sondes ne
représentent pas, et la part non représentée de la sonde comme indicateur de couverture.

## Commandes de reproduction

```sh
uv run pytest tests/unit/test_probe_metrics.py tests/unit/test_defect_bench.py -q
CUDA_VISIBLE_DEVICES="" .venv/bin/python diagnosis/probe_metrics/dev_select.py
CUDA_VISIBLE_DEVICES="" .venv/bin/python diagnosis/probe_metrics/dev_select_v2.py
CUDA_VISIBLE_DEVICES="" .venv/bin/python diagnosis/probe_metrics/calibrate.py --phase test --families noise multisine
CUDA_VISIBLE_DEVICES="" .venv/bin/python diagnosis/probe_metrics/calibrate.py --phase test --probes v2 --families noise multisine
CUDA_VISIBLE_DEVICES="" .venv/bin/python diagnosis/probe_metrics/calibrate.py --phase test --probes wh --families noise --levels -20 --seconds 1
CUDA_VISIBLE_DEVICES="" .venv/bin/python diagnosis/probe_metrics/decomposition.py noise
CUDA_VISIBLE_DEVICES="" .venv/bin/python diagnosis/probe_metrics/real_pairs.py vs100r
.venv/bin/python diagnosis/probe_metrics/summarize.py && .venv/bin/python diagnosis/probe_metrics/figures.py
.venv/bin/python diagnosis/probe_metrics/manifest.py
```
(les autres familles : `--families sweep two_tone`, `plucks`, `di`.)
