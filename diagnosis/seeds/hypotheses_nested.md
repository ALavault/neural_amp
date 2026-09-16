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
