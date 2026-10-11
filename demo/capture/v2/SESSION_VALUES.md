# Capture v2 : valeurs relevées pendant la séance Nolly (2026-10-11)

Valeurs fixes de la séance demandées par `SESSION.md`, recopiées du relevé du worker
(`nolly_logs/session_values.md`, non commité, avec les journaux de la séance). Écarts à
`SESSION.md` : `diagnosis/probe_metrics/NOLLY_READING.md`, section « Écarts ».

| | valeur |
|---|---|
| Version de Neural DSP Archetype: Nolly, format (VST3/AU) | Archetype Nolly 2.0.0, VST3 (Archetype Nolly.vst3), 2 entrées / 2 sorties |
| Système, version de Reaper | Windows 11 Pro 10.0.26300 (x64) ; REAPER 7.80/x64 |
| Commande de l'ampli pour « gain » (G) | Rhythm Gain (GAIN) |
| Commande de l'ampli pour « grave » (B) | Rhythm Low (LOW) |
| Commande de l'ampli pour « médium » (M) | Rhythm Mid (MID) |
| Commande de l'ampli pour « aigu » (T) | Rhythm High (HIGH) |
| Commande de l'ampli pour « présence » (P) | Rhythm Presence (PRESENCE) |
| Autres commandes de l'ampli (valeur fixe) | Rhythm Master 5 (0.50), Rhythm Resonance 5 (0.50), Rhythm Level 0,0 dB ; MODE MONO ; Gate -90,0 (minimum, grisé = coupé) ; pédales, Time (delay, reverb), EQ graphique coupés ; affichage hôte 0,00..1,00 = bouton 0..10, table exacte : knob_table.json |
| Gain d'entrée, gain de sortie | Input 0,0 dB ; Output 0,0 dB |
| Mode de qualité ou de suréchantillonnage | OVERSAMPLING NORMAL (choix de l'utilisateur après le diagnostic de déterminisme ; HIGH par défaut) ; aucun suréchantillonnage REAPER |
| PDC affichée (échantillons) | 3 (REAPER, toutes les instances) ; retard mesuré aux marqueurs après compensation : médiane 175 sans baffle, [184, 188, 205] avec baffle |
| Déterminisme : résidu du nul (dBFS) | Non déterministe. Deux instances HIGH : crête -16.1 dBFS ; même instance rendue deux fois (HIGH) : crête -16.6 dBFS ; deux instances NORMAL : crête -29.2 dBFS ; résidu RMS typique ≈ -75,5 dBFS pour un signal ≈ -18 dBFS (determinism_report.md) |
| Bloc cab on : baffle, micros, placement | Cab Type Rhythm ; Cab 1 : Dynamic 57, position 0,35, distance 0,10, niveau -12,0 dB, pan 0,00, phase normale ; Cab 2 coupé ; pas d'envoi de réverbération de pièce dans Nolly |

Inventaire : 2216 paramètres (params_inventory.json).
Contrôles stéréo (NORMAL) : L = R exactement, baffle coupé et baffle actif (prechecks_normal.json).
