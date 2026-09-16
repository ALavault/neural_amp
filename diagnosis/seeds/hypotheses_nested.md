# Graine ou non-déterminisme ? Prédictions écrites avant les répétitions (2026-09-16 10:30)

Constat de départ (`global_factor.txt`) : la graine multiplie l'erreur de tous les
segments de test par un même facteur. On mesure ce facteur par l'effet run, la
moyenne sur les douze segments du log de l'ESR. Son écart-type entre graines vaut
0,53 pour SSM-WaveNet avec les deux changements, 0,27 pour S4-TF-L-16 avec les
mêmes changements et 0,16 pour S4-TF-L-16 entraîné comme publié.
L'entraînement n'est pas déterministe sur GPU. Une seule répétition a été faite
(SSM, graine 42) : décalage de +0,23 en log.

Plan : deux runs par graine (42, 43, 44). Pour SSM-WaveNet avec changements, il
manque les répétitions des graines 43 et 44. Pour S4-TF-L-16 publié, les trois
répétitions. Les commandes sont identiques aux runs d'origine
(`scripts/product_nablafx_queue.sh`, round 5). Aucun run n'est écarté. On
analyse l'effet run, en ANOVA emboîtée (graine, répétition dans la graine).

**N — non-déterminisme dominant.** Une perturbation de l'ordre de l'arrondi suffit
à tirer un résultat dans toute la distribution, comme si la graine était redessinée.
*Prédiction* : pour SSM, l'écart-type intra-graine de l'effet run est ≥ 0,35, et
au moins une des deux nouvelles répétitions (43, 44) s'écarte de son original
d'un facteur ≥ 1,5 sur l'ESR moyen (|Δ log| ≥ 0,4). Pour S4 publié,
l'écart-type intra-graine est ≥ 0,12.

**G — graine dominante.** L'initialisation, la partition ou l'ordre des lots fixent
le facteur ; le non-déterminisme ne fait que le bruiter.
*Prédiction* : pour SSM, l'écart-type intra-graine est ≤ 0,20, les deux
nouvelles répétitions restent à moins d'un facteur 1,35 de leur original
(|Δ log| ≤ 0,3), et l'ordre des moyennes par graine (42 < 44 < 43) est conservé.

Entre les deux (écart-type intra-graine entre 0,20 et 0,35 pour SSM) : indécidable
avec trois degrés de liberté ; il faudra plus de graines et de répétitions, pas une
conclusion.

## Résultat pour SSM-WaveNet (2026-09-16 15:00, écrit après mesure)

Trois paires (`global_factor.txt`, section « Nested ») :
- effet run, écart-type intra-graine 0,18 ; composante graine 0,37 ; F(2,3) = 8,8,
  p ≈ 0,06 ;
- seconde exécution contre première, log du rapport des ESR moyens : graine 42
  +0,27, 43 −0,44, 44 −0,23 ;
- ordre des graines par effet moyen : 42 < 44 < 43, conservé.

Verdict, critère par critère :
- **N** : écart-type intra-graine ≥ 0,35, non (0,18) ; une répétition au-delà d'un
  facteur 1,5, oui (graine 43, facteur 1,55). Les deux étaient requises : N réfutée.
- **G** : écart-type intra-graine ≤ 0,20, oui ; les deux nouvelles répétitions à
  moins d'un facteur 1,35, non (graine 43) ; ordre conservé, oui. G réfutée telle
  qu'écrite.

Ni N ni G. La graine porte l'essentiel de la variance estimée (0,37² contre 0,18²,
environ 80 %), sans atteindre 5 % avec trois degrés de liberté. Le non-déterminisme
seul donne un écart-type de 0,18 par run (facteur 1,2) et a déplacé un run d'un
facteur 1,55. S4 publié : en attente.
