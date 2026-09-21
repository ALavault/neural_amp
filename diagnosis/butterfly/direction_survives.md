# La direction du coup de pouce survit-elle jusqu'à la fin ?

Lecture seule, CPU, sur des points de contrôle déjà écrits.
Script : `diagnosis/butterfly/direction_survives.py`. Sortie : `direction_survives.json`.

## Hypothèse testée

`k = 1` est l'enfant qui décroche dans les trois configurations où une dispersion existe : bras
« décide » de la fourche 100, et les deux bras de la fourche 5. Sa direction de perturbation est
tirée avec la même graine 1000 + k sur deux parents différents. Avec quatre enfants, trois fois de
suite a environ une chance sur seize d'être fortuit. Si la direction portait de l'information, le
déplacement final d'un enfant devrait garder un alignement avec son coup de pouce initial.

## Méthode

Le coup de pouce est reproduit exactement comme l'applique `scripts/product_fork_pilot.py` : un
générateur de graine 1000 + k tire un bit par poids, et `torch.nextafter` déplace chaque poids d'un
pas float32 dans cette direction. On mesure le cosinus entre ce vecteur et le déplacement final
θ(enfant) − θ(témoin), puis les cosinus entre coups de pouce, qui n'ont aucune raison d'être autre
chose qu'orthogonaux.

**La loi nulle n'est pas celle qu'on croit.** Un pas float32 est proportionnel à |θ|, donc le
vecteur de perturbation a des composantes d'amplitude très inégale : **336 poids portent 90 % de sa
norme**, et les degrés de liberté effectifs, (Σu²)²/Σu⁴, valent **283** et non 11 329. L'écart-type
nul du cosinus est donc 1/√283 = 0,0595, et non 1/√11 329 = 0,0094. Sans cette correction, tous les
chiffres ci-dessous paraîtraient massivement significatifs.

## Observations

Cosinus entre le coup de pouce et le déplacement final, norme du coup de pouce 1,35·10⁻⁵ :

| fourche | bras | k = 1 | k = 2 | k = 3 | k = 4 |
|---|---|---|---|---|---|
| 100 | décide | −0,034 | −0,033 | −0,012 | −0,062 |
| 100 | rejoue | −0,024 | −0,079 | −0,007 | −0,084 |
| 5 | décide | +0,010 | +0,009 | −0,017 | +0,002 |
| 5 | rejoue | +0,002 | +0,013 | −0,016 | +0,001 |

Cosinus entre coups de pouce : −0,130 à +0,074.

**Le plus grand |cosinus| observé vaut 1,4 écart-type de la loi nulle correcte.** Aucune valeur
n'est distinguable de zéro. La direction initiale ne laisse pas de trace linéaire dans le
déplacement final, ni à la fourche 100 ni à la fourche 5, dans aucun des deux bras.

## Ce que cela permet de conclure

- L'avantage apparent de `k = 1` ne s'explique pas par un alignement persistant de sa direction de
  perturbation avec le chemin que le run finit par prendre. Le déplacement final, de norme 5 à 8,
  est entièrement dominé par l'entraînement ordinaire.
- La question `Q-0010` de la mémoire partagée reste ouverte, mais ce mécanisme-là est écarté.

## Ce que cela ne permet pas de conclure

- Le cosinus ne voit que l'alignement **linéaire**. La direction pourrait compter par un chemin non
  linéaire — par exemple en poussant quelques poids d'un côté d'une décision de la validation —
  sans laisser de trace linéaire. Cette mesure ne l'exclut pas.
- Les huit cosinus de la fourche 100 sont tous négatifs, ce qui aurait une chance sur 256 d'arriver
  si les signes étaient indépendants. Ils ne le sont pas : les huit déplacements se mesurent depuis
  le même témoin et les huit coups de pouce partagent le même profil d'amplitudes. Motif non
  expliqué, non significatif pris valeur par valeur, à ne pas rapporter comme un effet.
- Une paire témoin/enfant par k, une graine, deux fourches. La seizième valeur, rejoue k = 4 à
  la fourche 5, a été ajoutée le 2026-09-21 quand ce run a existé ; elle vaut +0,001 et ne change
  rien.

## Prochaine expérience discriminante

La question de la direction se tranche en changeant la direction à perturbation égale : refaire deux
enfants à la même fourche avec les graines 1004 et 1005, dont les directions sont indépendantes de
celles déjà tirées. Si le décrochage suit toujours l'indice k = 1 plutôt que la direction, c'est
l'ordre de tirage qui compte et non la direction ; s'il suit la direction, le pré-enregistrement du
pilote A a tort de la supposer indifférente. Coût : deux runs GPU.
