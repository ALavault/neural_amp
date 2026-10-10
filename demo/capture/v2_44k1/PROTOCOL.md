# Variante 44,1 kHz du fichier de capture v2 : sous-protocole KIPR

Voie produit. Généré par `scripts/product_capture_v2_44k1.py` ; ne pas éditer à la main. Pour le banc KIPR seulement (KIPR Native V1 et référence Player, 44,1 kHz, rendus hors ligne). Séparée du fichier 192 kHz (`demo/capture/v2/`) et de la séance Nolly, qui ne changent pas.

- **Fichier** : `datasets/raw/internal/capture_v2/reamp_v2_44k1.wav`, 451.631 s, 44,1 kHz, 24 bits, mono, sha256 `03e56f9a1fe1ea56d728c3e0e8ee2b4309ae283131bf0226678a36b30101340d`.
- **Sections** : les mêmes que le fichier 192 kHz, dans le même ordre (`layout.json`). DI rééchantillonnée depuis le 192 kHz (147/640, Kaiser β = 8,6) ; marqueurs, silence, bruit rose, sinus glissants, multisinus, deux tons et paliers synthétisés à 44,1 kHz.
- **Identité au bit perdue** : ce fichier n'est identique au bit ni au fichier VS100R ni au fichier v2 à 192 kHz. Research-memory D-0012 ne vaut que pour le 192 kHz.
- **Emboîtement conservé** : `di_train` + `di_validation` font 150 s, soit 50 segments de 3 s (132 300 échantillons). Les extraits de 30 s (positions 0, 5, …, 45) restent contenus.
- **Indépendance A/B** : `di_train` et `di_validation` viennent des mêmes cinq enregistrements (fenêtres différentes) ; `di_test` vient de deux autres (`idmt-bass`, `idmt-gtr4-ib`). Pour deux sources disjointes : A = `di_train`, B = `di_test`.
- **Borne harmonique** : l'ordre k ne se lit que pour f < 22 050 / k Hz (ordre 3 : 7 350 Hz ; ordre 9 : 2 450 Hz). Plafonner la bande de 100 à 8 000 Hz par ordre.
- **Retard** : estimer et rapporter un retard fractionnaire résiduel entre sorties, sans recaler ni corriger les sorties ou les scores. Les 0,8 à 1,05 échantillon observés concernent des modèles VS100R à 48 kHz : ce n'est pas une plage attendue pour les moteurs KIPR.
