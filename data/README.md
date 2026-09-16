# data/

Place the Pléiades scenes here (not distributed, © CNES / Airbus DS):

```
data/
├── Train_Image/Train_RGBPIR_Sudan.tif
└── Validation_image/
    ├── Validation_RGBPIR_Sudan.tif
    └── Validation_RGBPIR_Egypt.tif
```

Each file is a 4-band uint16 GeoTIFF (EPSG:32636), nodata = 65535. Band order: Sudan = B, G, R, NIR; Egypt = R, G, B, NIR.
