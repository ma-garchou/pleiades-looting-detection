# labels/

`labels_pillage.gpkg` (EPSG:32636) is **not published**: it contains the exact positions of looted archaeological sites. It is only needed to **retrain** (`--retrain`). Inference works with the published weights alone.

Expected layers (polygons):

| Layer | Content |
|---|---|
| `aoi` | Exhaustively labelled area (here: the whole training scene) |
| `pit` | Looting pits (dark holes only; necropolis tombs are NOT pits) |
| `spoil` | Spoil heaps (optional, empty in our run) |
| `vegetation` | Vegetation (optional; otherwise NDVI > 0.30 on the training scene) |
| `ignore` | Uncertain areas excluded from training (optional) |
