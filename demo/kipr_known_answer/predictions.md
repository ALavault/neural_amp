# Banc à réponse connue : plancher de 2ᵉ harmonique de SSM-WaveNet sur une cible symétrique

Voie produit (D-PRODUCT-001), aucune revendication scientifique. **Écrit le 2026-10-06, avant tout
entraînement.** Les règles ci-dessous ne changent plus après les chiffres.

## Question

Sur la VS100R au gain 2,5, l'ampli est presque symétrique (2ᵉ harmonique 59 à 68 dB sous le
fondamental), et les modèles gardent un plancher vers −30 à −40 dB. Le bras C a soutenu une cause
structurelle : la porte `tanh·sigmoid` (`diagnosis/hammerstein/NOTE.md`). La VS100R ne permet pas de
séparer la porte des particularités de ses données : réglages mélangés, bruit, une seule prise par
couple. Les mesures KIPR (`datasets/raw/external/kipr/`, non commité) donnent des cibles dont la
symétrie est connue. Elles viennent d'un moteur DSP émulé, pas d'un ampli.

**Décision que la mesure change** : s'il y a un plancher sur une cible symétrique propre, alors
remplacer la porte (porte impaire avec asymétrie explicite, comme dans le moteur KIPR) mérite une
préinscription. Sinon, le plancher vient de la VS100R ou de son entraînement, et c'est là qu'il
faut chercher.

## Cibles (sortie gauche ; la droite en diffère de moins de 3·10⁻⁵)

À 1 kHz, H2/H1 de la cible (KIPR `metrics.json`, à recalculer avec notre mesure avant toute
lecture) :

| configuration | rôle | −24 dBFS | −12 dBFS | −6 dBFS | H3/H1 à −12 |
|---|---|---|---|---|---|
| `v1_lead_amp` | symétrique | −47,2 | −59,2 | −65,1 | −1,5 |
| `v2_lead_bias_16383` | asymétrique, témoin positif | −28,8 | −35,7 | −36,1 | −3,1 |

## Entraînement (script `known_answer.py`, commité avec ce texte)

- Données : les quatre segments `train_noise` (−36, −24, −12, −6 dBFS crête, 2 s chacun), pris
  dans la séquence complète avec le silence qui les précède (1,0 s, non compté dans la perte).
  Cible avancée de 182 échantillons : l'alignement estimé par KIPR (198) moins 16, pour laisser
  au modèle un petit retard causal à apprendre.
- Modèle : SSM-WaveNet comme les modèles VS100R (8 blocs, 16 canaux, état 4, ZOH, tête `leaky`,
  sortie `tanh`), sans bouton. Initialisation en secondes et tailles MR-STFT ramenées à 44,1 kHz
  (`at_rate` du banc).
- Optimisation : celle du banc. AdamW à lr 0,01, groupes `_optim` (`--honor-optim`), écrêtage des
  gradients à 1,0, perte L1 0,5 + MR-STFT 0,5. Les quatre segments forment chaque lot. 3 000 pas,
  lr divisé par 2 aux pas 1 800 et 2 400. FFT `fast`, sans TF32, déterministe.
- Graines 42, 43, 44, par configuration : 6 runs, inscrits dans `demo/RUNS.jsonl`. Smoke de
  20 pas : 0,52 s par pas et 3,8 Gio de GPU, soit environ 26 min par run.

## Mesure

Pour la cible et pour chaque modèle, avec la même fonction : sons de 1 kHz de la séquence, chacun
précédé de son silence. On analyse les 0,5 s de régime établi (500 périodes exactes) avec une
fenêtre de Blackman-Harris à 4 termes. On mesure l'amplitude de H1 à H5 aux raies exactes, et
H2/H1 et H3/H1 en dB. **Contrôle de l'instrument** : notre H2/H1 de la cible doit retrouver les
valeurs KIPR à 1 dB près aux trois niveaux. Sinon, aucune lecture.

E = H2/H1 du modèle − H2/H1 de la cible, en dB. Lecture primaire : médiane sur les trois graines,
puis moyenne sur −12 et −6 dBFS.

## Prédictions

- **Validité**, niveau par niveau (−12 et −6 dBFS), sur la médiane des trois graines, pour
  que des erreurs de signe opposé ne se compensent pas :
  - témoin positif (`v2_lead_bias_16383`) : |E| ≤ 5 dB et |écart de H3/H1| ≤ 3 dB ;
  - cible symétrique (`v1_lead_amp`) : |écart de H3/H1| ≤ 3 dB, pour qu'un modèle qui n'a pas
    appris la non-linéarité ne passe pas pour « sans plancher ».

  Si l'un échoue, le banc est **non concluant** : 8 s de bruit ne suffisent pas à apprendre la
  structure harmonique. Si le contrôle de l'instrument échoue, aucune lecture.
- **Plancher** (la porte impose un plancher, comme sur la VS100R) : sur `v1_lead_amp`, E ≥ +15 dB,
  c'est-à-dire un H2/H1 du modèle au-dessus d'environ −47 dB.
- **Pas de plancher** : sur `v1_lead_amp`, E ≤ +5 dB.
- Entre +5 et +15 dB : plancher partiel, rapporté tel quel.

**Lectures descriptives**, sans seuil : E à −24 dBFS ; ESR des segments `test_noise` par niveau ;
E et l'écart de H3 à 300 Hz et 3 kHz.

## Ce qui ne pourra pas être conclu

- Rien sur un ampli réel : la cible est un moteur DSP émulé.
- Le modèle n'a pas de bouton, et 8 s de bruit ne sont pas de la guitare. Un plancher trouvé ici
  dit que la porte peut le produire, pas qu'il est la seule cause sur la VS100R.
- Une absence de plancher n'innocente pas la porte sur la VS100R ; elle reporte la question sur
  les données et le conditionnement.

## Résultats (2026-10-06, six runs, exit 0)

Instrument : écart maximal 0,0007 dB avec les valeurs KIPR. Lecture par `known_answer.py measure`,
sortie `known_answer.json`.

**Verdict préinscrit : non concluant.** Le témoin positif échoue sur H2 à −6 dBFS (E médian +6,8 dB
pour un seuil de 5) ; il passe à −12 dBFS (+4,7 dB) et sur H3 aux deux niveaux (écarts sous 0,1 dB).

| | E (H2/H1 modèle − cible), médiane des graines | | | écart H3/H1 |
|---|---|---|---|---|
| | −24 dBFS | −12 dBFS | −6 dBFS | −12 / −6 dBFS |
| `v1_lead_amp` (symétrique) | +12,0 | +36,8 | +46,7 | −0,4 / −0,9 |
| `v2_lead_bias_16383` (témoin) | +0,3 | +4,7 | +6,8 | −0,1 / +0,1 |

**Défaut du protocole, trouvé après la lecture (établi).** L'ESR des segments `test_noise` vaut
environ 2 pour les six runs, à tous les niveaux : la sortie ne suit pas la forme d'onde de la cible.
Par corrélation croisée sur `test_noise` (graine 42) :
- entrée → cible brute : pic à 198 échantillons, comme l'estimation KIPR ;
- sortie du modèle → cible avancée de 182 : le modèle est en avance de 15 échantillons (symétrique,
  corrélation 0,84 à 0,86) et de 18 (témoin, 0,47 à 0,49).

Le modèle n'a pas appris les 16 échantillons de retard pur laissés par l'avance de 198 − 16. C'est
le piège déjà documenté pour F-Fuzz (`DIAGNOSIS_ffuzz.md` : ces modèles n'apprennent pas la latence
de la cible). Il fallait avancer la cible de toute la latence. Avec une cible décalée, la perte L1
pousse vers une sortie lissée ; seule la MR-STFT, peu sensible au retard, guide la forme spectrale.

**Ce que cela interdit.** L'écart entre la cible symétrique (+37 et +47 dB) et le témoin (+5 et
+7 dB) est rapporté sans interprétation : il peut venir de la porte, ou d'un apprentissage gêné
par le retard qui touche différemment les deux cibles. Le banc ne tranche pas la question de la
porte dans cet état.

**Pour une reprise** (non lancée ; sa préinscription reste à écrire) : cible avancée de 198, et
porte préalable sur l'alignement : décalage du modèle à la cible au plus 2 échantillons et ESR de
`test_noise` sous 1. Sinon, aucune lecture.
