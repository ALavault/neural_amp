# Versions successives de la contribution retenue (DIAGNOSIS_seeds)

Déplacé de `DIAGNOSIS_seeds.md` le 2026-09-21 pour tenir le document de tête sous 1 200 mots.
Rien n'est retiré : une explication réfutée se garde, sans quoi on la réinvente.

**Révision du 2026-09-16 (après coup, `global_factor.py`, `global_factor.txt`).** La graine
multiplie l'erreur de tous les segments par un même facteur. Rapport pire/meilleure
graine, moitié calme contre moitié forte : ×1,69 contre ×1,63 (S4, changements),
×1,35 contre ×1,44 (S4 publié), ×2,98 contre ×2,73 (ablation SSM). Seul SSM avec
changements s'écarte, avec ×4,00 contre ×2,03. Sur le log de l'ESR, l'effet run,
commun aux douze segments, domine l'interaction run × segment : F de 9 à 36.
Les 77–81 % observés sur la moitié calme relèvent de l'arithmétique. Un facteur
unique y placerait 73 à 83 % de l'écart, puisque ces segments ont déjà un ESR
2,6 à 4,8 fois plus grand. Le critère absolu n'est pas en cause non plus :
l'ESR pondéré par l'énergie, dominé par le matériel fort, varie autant (H4).
Contribution révisée : la graine fixe un facteur global d'erreur, d'écart-type
0,53 en log de l'ESR pour SSM-WaveNet (×1,7), 0,27 et 0,16 pour S4, de mécanisme
inconnu. La question suivante, graine ou non-déterminisme, a ses prédictions
écrites avant mesure (`hypotheses_nested.md`).

**Version initiale, réfutée (texte complet au commit `6025b26`).** *Prédicat* :
77 à 81 % de l'écart entre meilleure et pire graine sur les six segments calmes.
*Mécanisme* : un critère d'entraînement et de sélection absolu face à une
métrique relative laisserait le bas niveau à la graine. *Prédiction* : une
validation stratifiée par niveau ramènerait le CV de SSM sous 25 % et le rapport
calme/fort sous 3,0. Elle vise un mécanisme faux et n'est plus un critère.
