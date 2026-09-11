# Protocole AMP-QUALITY-ARCH-v3

Le contrat normatif se trouve dans
`configs/amp_quality_arch_v3/protocol.yaml`. La lignée commence par une gate de
représentabilité train-only, puis autorise au plus trois rounds prospectifs.
Chaque round possède son propre gel avant exécution ; aucun retry ou retuning
intra-round dépendant du résultat n'est permis.

Les systèmes primaires durent 1,5 seconde dont 0,3 seconde de pré-roll et 1,2
seconde scorée. Le composite dynamique v2 reste disponible comme stress test,
mais ne participe pas à la compétence primaire. Les cinq seeds confirmatoires,
les seuils de compatibilité v2 et les caps CPU/latence sont gelés avant tout run.
