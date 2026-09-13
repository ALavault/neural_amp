# Diagnostic de données : le plafond est-il dans la donnée ou le modèle ?

Appareil `electro_harmonix_big_muff`, **fichier de test uniquement**. Les conclusions
portent sur ce découpage, pas sur « le jeu de données ».

## Verdict

Corrélation des résidus 0.773 : partiellement partagée, ni franchement la donnée ni franchement le modèle.

Un A2 Full entraîné SUR la paire de test n'y descend qu'à 0.05953 d'ESR, contre 0.02113 sur la paire de validation traitée de la même façon. La paire de test n'est pas cohérente avec elle-même : la cible n'est pas une fonction stable de cette entrée.

**Décision : recapturer avant toute ambition SOTA.** Optimiser une architecture contre une paire que l'on ne peut pas reproduire même en la surapprenant revient à courir après du bruit de mesure.

Localisation de l'erreur : ESR médian par seconde 0.1714, maximum 0.6094 (rapport 3.6). Bande dominante du résidu : 1000-4000 Hz.

## 1. Les deux modèles font-ils la même erreur ?

| Mesure | Valeur |
| --- | ---: |
| Corrélation des résidus A2 Full / A2 Lite | 0.7733 |
| RMS du résidu, A2 Full | 0.01511 |
| RMS du résidu, A2 Lite | 0.01888 |

## 2. Où vit l'erreur

| Bande | Part du résidu | ESR dans la bande |
| --- | ---: | ---: |
| 20-200 Hz | 0.4 % | 0.0067 |
| 200-1000 Hz | 8.2 % | 0.0498 |
| 1000-4000 Hz | 47.4 % | 0.1938 |
| 4000-12000 Hz | 39.0 % | 0.6299 |
| 12000-24000 Hz | 4.8 % | 1.2487 |

| Seconde | ESR | RMS cible |
| ---: | ---: | ---: |
| 0 (silence) | 9.0272 | 0.0002 |
| 1 | 0.1104 | 0.0284 |
| 2 | 0.3020 | 0.0418 |
| 3 | 0.4672 | 0.0392 |
| 4 | 0.3663 | 0.0373 |
| 5 | 0.4017 | 0.0255 |
| 6 | 0.0877 | 0.0404 |
| 7 | 0.1487 | 0.0383 |
| 8 | 0.0627 | 0.0394 |
| 9 | 0.0751 | 0.0402 |
| 10 | 0.0303 | 0.0468 |
| 11 | 0.0594 | 0.0378 |
| 12 | 0.3064 | 0.0349 |
| 13 | 0.3572 | 0.0402 |
| 14 | 0.2499 | 0.0419 |
| 15 | 0.2574 | 0.0398 |
| 16 | 0.6094 | 0.0218 |
| 17 | 0.2599 | 0.0247 |
| 18 | 0.1714 | 0.0310 |
| 19 | 0.0276 | 0.0333 |
| 20 | 0.0892 | 0.0333 |
| 21 | 0.2000 | 0.0218 |
| 22 | 0.3276 | 0.0266 |
| 23 | 0.1989 | 0.0276 |
| 24 | 0.0983 | 0.0343 |
| 25 | 0.0945 | 0.0306 |
| 26 | 0.0315 | 0.0250 |
| 27 | 0.0479 | 0.0379 |
| 28 | 0.2623 | 0.0445 |
| 29 | 0.0987 | 0.0426 |

## 3. Cohérence interne des paires (surapprentissage délibéré)

Budget mis à l'échelle pour égaler le nombre de mises à jour du run de
référence (400 époques sur 120 s), sinon l'oracle n'a pas les moyens de
surapprendre et son échec ne prouve rien.

| Paire ajustée | Époques | Minutes | ESR A2 Full ici | ESR A2 Lite |
| --- | ---: | ---: | ---: | ---: |
| test | 0.05953 | 0.10886 |
| validation | 0.02113 | 0.06389 |

## 4. Couverture d'amplitude

99,9e centile de |x| à l'entraînement : 0.4312

| Split | p50 | p99 | p99,9 | crête | part au-dessus du p99,9 train |
| --- | ---: | ---: | ---: | ---: | ---: |
| train | 0.0244 | 0.3253 | 0.4312 | 0.5515 | 0.100 % |
| validation | 0.0140 | 0.2984 | 0.4173 | 0.5508 | 0.072 % |
| test | 0.0407 | 0.3572 | 0.4764 | 0.5494 | 0.267 % |

## 5. Courbe de transfert E[y | x] par split

Un réglage de pédale différent entre les prises se voit ici.

| Split | Écart relatif maximal à la courbe d'entraînement |
| --- | ---: |
| validation | 113.00 % |
| test | 40.47 % |
