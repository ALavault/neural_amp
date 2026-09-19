# S4-TF-L-16 : la graine ou le non-déterminisme ? (T-0004)

Trois répétitions à graine égale de l'entraînement publié réexécuté, terminées le 2026-09-20 à
01:15. Enregistrements dans `demo/nablafx_bench/s4tfl16_seed*.json` et `*_repeat.json`.

## Comparabilité, vérifiée avant toute lecture

Chaque paire enjambe des commits différents du dépôt, ce qui pourrait confondre non-déterminisme et
dérive du code. Vérification faite :

| graine | commits | lignes changées dans le script d'entraînement | verdict |
|---|---|---|---|
| 42 | 35b468f1 → 1fc118b4 | 156 + 12 | **non vérifiable** : au commit d'origine, la garde de polarité et l'exemption de weight decay n'existaient pas dans le script |
| 43 | 15336c90 → 24076688 | 54 + 8 | **vérifiée comparable** |
| 44 | 0ad9b395 → 24076688 | 54 + 8 | **vérifiée comparable** |

Pour les graines 43 et 44, le diff se réduit à l'ajout de l'option `--deterministic` et du padding
par découpage, « installé seulement avec `--deterministic` », que les répétitions n'utilisent pas.
Sans cette option, le commit des répétitions appelle `use_deterministic_algorithms(False,
warn_only=True)`, passe `benchmark=True` et `deterministic=None` au `Trainer`, et emploie un lot de
16 — identique au commit d'origine. Ces deux paires mesurent donc le non-déterminisme seul.

## Observations

| graine | initial | répétition | Δ en log |
|---|---|---|---|
| 42 | 0,1373 | 0,0999 | −0,318 (non vérifiable) |
| 43 | 0,1569 | 0,1111 | **−0,345** |
| 44 | 0,1171 | 0,1145 | **−0,023** |

- Sur les deux paires vérifiées : écart-type intra-graine **0,173**, écart entre les moyennes 0,093.
- Sur les trois paires : intra-graine 0,192, écart des moyennes 0,073, **F(2,3) = 0,29, p = 0,77**.

Comparaison avec SSM-WaveNet (`diagnosis/seeds/hypotheses_nested.md`) : intra-graine 0,18,
composante graine 0,37, F(2,3) = 8,8, p ≈ 0,06.

## Ce que cela permet de conclure

- **Chez S4-TF-L-16, aucun effet de graine n'est détectable**, et la variation d'un run à l'autre à
  graine égale est du même ordre, voire plus grande, que la variation entre graines. C'est l'inverse
  de SSM-WaveNet, où la graine porte environ 80 % de la variance estimée.
- Le non-déterminisme GPU seul déplace un run de 0,345 en log, soit un facteur 1,4 sur l'ESR, à
  graine, données et code identiques.
- Conséquence directe pour le banc : l'écart entre SSM-WaveNet et S4-TF-L-16 se compare à une
  dispersion qui, pour S4, est celle d'un modèle avec lui-même.

## Observation non conclusive, à signaler

La moyenne géométrique des trois répétitions vaut environ 0,108, soit exactement la valeur publiée
pour S4-TF-L-16, quand celle des trois runs initiaux valait 0,136. Il serait tentant d'y voir une
reproduction réussie de la valeur publiée. Avec un non-déterminisme de 0,17 à 0,35 en log et trois
runs par groupe, cette coïncidence n'est pas distinguable du hasard.

## Ce que cela ne permet pas de conclure

- Deux paires vérifiées seulement ; la troisième reste inutilisable tant que le run initial de la
  graine 42 n'a pas été refait au commit courant.
- F(2,3) sur trois paires n'a aucune puissance : ne pas lire « p = 0,77 » comme une preuve d'absence
  d'effet de graine, seulement comme une absence de détection.
- L'entraînement publié réexécuté n'est pas l'entraînement modifié : rien ici ne porte sur les runs
  avec les deux changements.

## Prochaine expérience discriminante

Refaire le run initial de la graine 42 au commit courant, ce qui rend la troisième paire utilisable
et porte à trois le nombre de paires comparables. Un run d'environ 1 h 30. Sans cela, toute
statistique emboîtée pour S4 repose sur deux paires.
