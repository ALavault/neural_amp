# External download ledger

Downloaded on 2026-08-27 after license approval. Archives are stored under `datasets/raw/external/`, which is ignored by Git. Published MD5 values and locally computed SHA-256 values are recorded here and in `catalog.yaml`.

| Resource | Archive | Size | Published MD5 | Local SHA-256 | Tier |
| --- | --- | ---: | --- | --- | --- |
| ToneTwist Fulltone Full Drive 2 | `tone_twist_fulltone/Fulltone-FullDrive2.zip` | 1,646,555,282 B | `0bb9809efe4545071ea117f86adce529` | `06c09fbca4ecaa4fd36b577c64afa7e385f2715ff99822d87910afdfcb8dfbac` | INTERNAL_DEV |
| ToneTwist Blackstar HT1 Overdrive | `tone_twist_blackstar/Blackstar-HT1-ChOverdrive.zip` | 60,268,284 B | `a748ac0fa6608a2dd57444bac6f132b7` | `78c35a60cb0309aa6c244f2763ff347a82228bbed090eebbdbd3d82bf9ffefdd` | INTERNAL_DEV |
| ToneTwist Electro-Harmonix Big Muff | `tone_twist_bigmuff/ElectroHarmonix-BigMuff.zip` | 49,975,728 B | `45bdd8ea776e1182b9db1d60f62b7930` | `200bee0dc2887d331f6aef61df1f08d52a0067c938e8e435bb2493ad53c639fa` | INTERNAL_DEV |
| Marshall JVM410H OD1 | `marshall_jvm410h/MarshallJVM410H.zip` | 4,369,237,770 B | `4d08dac89b44f618f2d6c2a69739e104` | `7517cb07478f44de283c513f37417b225b93cd143918226dbf05311eed196a6b` | INTERNAL_DEV |
| ToneTwist shared dry markers | `tone_twist_dry_markers/DRY-with-markers.zip` | 865,850,744 B | `bc1d1490f6c6cfe5643c0798eb5fb5a4` | `e23935658160c7693544c7daf125366c8f141cf72728e5b5780709ec96dc29d3` | INTERNAL_DEV |

## Initial structural audit

ToneTwist Blackstar and Big Muff contain `trainval` and `test` target trees plus marker-bearing `DRY` inputs. The Fulltone target archive uses matching basenames from the separate dry-marker archive. Representative headers are mono; Blackstar and Big Muff are native 44.1 kHz PCM16, Fulltone is 48 kHz float, and the Marshall representative is 44.1 kHz PCM24 with input/preamp/speakerout triplets.

No model was trained or selected from these files during this download action. Native-rate resampling, marker alignment, source/session grouping, and final split manifests remain required before M4. `EXTERNAL_REPORT_ONLY` remains inaccessible and its freeze control is unchanged.
