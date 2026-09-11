# Vérification métadonnée des données de confirmation — 2026-09-05

Aucun fichier ni membre d'archive n'a été téléchargé, résolu ou lu. Cette note
porte uniquement sur les pages publiques et ne remplace pas la gate
`confirmation_data_audit` après les locks.

## Disponible publiquement

- Rodent : enregistrement Zenodo 10796378, archive
  `HarleyBenton-Rodent.zip`, environ 2,9 Go,
  `md5:7ab43d078cc195f7595c8b1e1fd09723`. La page décrit sept réglages et les
  dry inputs à deux marqueurs.
  Source : https://zenodo.org/records/10796378
- Fuzzy Logic : enregistrement Zenodo 10796322, archive
  `HarleyBenton-FuzzyLogic.zip`, environ 1,3 Go,
  `md5:a702262de22616643a47df5d090c68e1`. La page décrit trois réglages et les
  dry inputs à deux marqueurs.
  Source : https://zenodo.org/records/10796322
- Le dépôt ToneTwist référence directement les deux enregistrements dans ses
  catégories analogiques Distortion et Fuzz, et exige des longueurs identiques,
  une synchronisation par marqueurs et du WAV 48 kHz float32.
  Source : https://github.com/mcomunita/tonetwist-afx-dataset

Les tailles et checksums correspondent aux valeurs déjà gelées dans
`configs/data/amp_quality_teacher_v1.yaml`.

Le volume `/fastdata` disposait de 920 869 154 816 octets libres au contrôle,
largement plus que les 4 186 152 684 octets d'archives compressées. La capacité
de stockage n'est donc pas le facteur bloquant observé.

## Provenance de licence

Le rendu direct des pages Zenodo omet visuellement la valeur du champ Rights,
mais les notices OpenAIRE fusionnées depuis Zenodo et DataCite indiquent toutes
deux `License: CC BY NC` pour Rodent et Fuzzy Logic :

- https://explore.openaire.eu/search/result?pid=10.5281/zenodo.10796378
- https://explore.openaire.eu/search/result?pid=10.5281/zenodo.10796322

Cela supporte la famille de licence non commerciale déclarée localement. La gate
autorisée devra encore vérifier que l'identifiant machine ou le README embarqué
précise bien la version `4.0` exigée par `CC-BY-NC-4.0`. Le dépôt GitHub sous MIT
ne doit pas être utilisé comme substitut à la licence audio. Selon l'aide
officielle Zenodo, les conditions applicables viennent du champ Rights du record :
https://support.zenodo.org/help/en-gb/2-content/21-can-i-get-permission-to-use-a-specific-record

Conclusion : disponibilité, identité, checksums et famille de licence supportés.
Seule la version exacte de la licence reste une vérification fail-closed de la
gate de confirmation; une incompatibilité impose `INVALID`/arrêt.
