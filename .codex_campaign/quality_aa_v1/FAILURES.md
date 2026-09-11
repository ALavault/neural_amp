# Échecs

Le préflight unique est `INVALID`. PyYAML a décodé la clé non citée `off` comme
le booléen `false`, puis le tri JSON strict a refusé le mélange de clés
booléennes et textuelles. Les références avaient été calculées, mais aucune
sortie x2/x4 n'avait été observée. Le run n'est pas repris.
