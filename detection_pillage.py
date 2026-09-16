#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
===============================================================================
 Détection automatique des fosses de pillage archéologique — images Pléiades
 U-Net + encodeur ResNet-50 (segmentation_models_pytorch)
-------------------------------------------------------------------------------
 Auteur : Mohamed-Ali Garchou — M2 Géomatique et Modélisation Spatiale (AMU)

 Dépôt : https://github.com/ma-garchou/pleiades-looting-detection

 UTILISATION (une seule commande) :
     python detection_pillage.py

 Le script enchaîne automatiquement :
   1. Prétraitement de l'image TRAIN (ordre des bandes, rééchantillonnage 0,5 m,
      normalisation)
   2. Rasterisation de la vérité terrain (labels/labels_pillage.gpkg)
   3. Entraînement de 2 modèles U-Net/ResNet-50 (graines 42 et 7)
      -> étape sautée si les poids sont présents dans modeles/ ou
         téléchargeables depuis les Releases GitHub (poids de nos résultats)
   4. Prédiction sur les 2 images de validation (Soudan puis Égypte)
   5. Résumé chiffré + figures dans resultats/

 Options :
     --retrain        réentraîne les modèles même si modeles/*.pt existent
     --root CHEMIN    dossier contenant Train_Image/ et Validation_image/
                      (par défaut : data/, puis le dossier du script, puis son parent)
     --test-rapide    exécution de contrôle très courte (1 époque, sans TTA)

 PROTOCOLE : le prétraitement et l'entraînement portent UNIQUEMENT sur l'image
 TRAIN. Les images de validation ne sont jamais modifiées ni apprises : elles
 sont lues telles quelles (lecture seule) au moment de la prédiction.
===============================================================================
"""
import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.transform import from_origin
from rasterio.warp import reproject

# =============================================================================
# 0. CONFIGURATION (valeurs exactes utilisées pour nos résultats)
# =============================================================================
HERE = Path(__file__).resolve().parent

# Images : `order` = bandes à lire pour obtenir Rouge, Vert, Bleu, PIR.
# ATTENTION : les fichiers du Soudan sont en B, G, R, PIR (malgré leur nom) ;
# le fichier d'Égypte est bien en R, G, B, PIR.
IMAGES = {
    "train_sudan": dict(rel="Train_Image/Train_RGBPIR_Sudan.tif", order=[3, 2, 1, 4], role="train"),
    "val_sudan":   dict(rel="Validation_image/Validation_RGBPIR_Sudan.tif", order=[3, 2, 1, 4], role="validation"),
    "val_egypt":   dict(rel="Validation_image/Validation_RGBPIR_Egypt.tif", order=[1, 2, 3, 4], role="validation"),
}
VALIDATIONS = ["val_sudan", "val_egypt"]      # ordre : 1) Soudan, 2) Égypte
NODATA_IN = 65535
TARGET_RES = 0.5                                # m (image TRAIN rééchantillonnée)
TARGET_CRS = "EPSG:32636"                       # UTM 36N
# Zone commune TRAIN / VALIDATION Soudan (exclue du calcul des scores)
OVERLAP_TRAIN_VAL = (221074.001, 2295270.175, 221551.531, 2295437.655)

# Classes
CLASSES = {0: "fond", 1: "fosse_pillage", 2: "deblais", 3: "vegetation"}
N_CLASSES = 4
IGNORE = 255
SPOIL_BUFFER_M = 0.0        # anneau de déblais automatique désactivé (fosses ~2,8 m)
NDVI_VEG_THRESHOLD = 0.30   # végétation automatique sur le TRAIN

# Réseau
ARCH, ENCODER, ENCODER_WEIGHTS, IN_CHANNELS = "Unet", "resnet50", "imagenet", 4

# Entraînement
SEEDS = [42, 7]
EPOCHS = 100
BATCH = 16
CROP = 256
SAMPLES_PER_EPOCH = 800
P_CENTER_ON_PIT = 0.6
LR = 3e-4
WEIGHT_DECAY = 1e-4
NUM_WORKERS = 2             # valeur utilisée pour nos résultats (garder 2)

# Poids entraînés publiés sur GitHub (Releases). Laisser vide pour désactiver.
MODELS_URL = "https://github.com/ma-garchou/pleiades-looting-detection/releases/download/v1.0/"

# Prédiction / post-traitement
PATCH = 512
STRIDE = 128
TTA = "d4"
PIT_THRESHOLD = 0.5
MIN_PIT_AREA_M2 = 1.0
MATCH_DIST_M = 3.0


# =============================================================================
# Chemins
# =============================================================================
class Paths:
    def __init__(self, root, test=False):
        self.root = Path(root)
        self.labels = HERE / "labels" / "labels_pillage.gpkg"
        # le mode test écrit ailleurs pour ne jamais écraser les vrais modèles/résultats
        self.models = HERE / ("modeles_test" if test else "modeles")
        self.out = HERE / ("resultats_test" if test else "resultats")
        self.data = self.out / "donnees_pretraitees"
        self.pred = self.out / "predictions"
        self.fig = self.out / "figures"
        for d in (self.models, self.data, self.pred, self.fig):
            d.mkdir(parents=True, exist_ok=True)

    def image(self, name):
        return self.root / IMAGES[name]["rel"]

    def model(self, seed):
        return self.models / f"{ARCH}_{ENCODER}_s{seed}.pt"


def find_root(user_root):
    candidates = [Path(user_root)] if user_root else [HERE / "data", HERE, HERE.parent]
    for c in candidates:
        if (c / IMAGES["train_sudan"]["rel"]).exists():
            return c.resolve()
    sys.exit("ERREUR : dossier 'Train_Image/' introuvable. Placez les images Pléiades dans data/ "
             "(data/Train_Image/, data/Validation_image/) ou utilisez --root CHEMIN.")


def log(msg):
    print(msg, flush=True)


# =============================================================================
# Lecture des images
# =============================================================================
def read_rgbn(P, name):
    """(uint16 4xHxW en R,G,B,PIR, masque valide, profil). Validation = lecture seule."""
    spec = IMAGES[name]
    if spec["role"] == "train":
        with rasterio.open(P.data / f"{name}_RGBN.tif") as src:
            img, profile = src.read(), src.profile
        valid = np.all(img > 0, axis=0)
    else:
        with rasterio.open(P.image(name)) as src:
            img, profile = src.read(spec["order"]), src.profile
        valid = np.all(img != NODATA_IN, axis=0)
    return img, valid, profile


def pixel_size(profile):
    return abs(profile["transform"].a)


def load_image(P, name):
    """Image normalisée (float32) avec les statistiques calculées sur le TRAIN."""
    raw, valid, profile = read_rgbn(P, name)
    img = raw.astype(np.float32)
    with open(P.data / "stats.json") as f:
        stats = json.load(f)["train_sudan"]
    out = np.empty_like(img)
    for i, b in enumerate(["R", "G", "B", "NIR"]):
        lo, hi = stats[b]["p1"], stats[b]["p99"]
        x = np.clip((img[i] - lo) / (hi - lo), -1.0, 2.0)
        out[i] = (x - 0.5) / 0.25
    out[:, ~valid] = 0
    return out, valid, profile


# =============================================================================
# ÉTAPE 1 — Prétraitement de l'image TRAIN
# =============================================================================
def step1_preprocess(P):
    log("\n=== ÉTAPE 1 : prétraitement de l'image TRAIN ===")
    name, spec = "train_sudan", IMAGES["train_sudan"]
    with rasterio.open(P.image(name)) as src:
        b = src.bounds
        r = TARGET_RES
        left, top = math.floor(b.left / r) * r, math.ceil(b.top / r) * r
        width = int(math.ceil((b.right - left) / r))
        height = int(math.ceil((top - b.bottom) / r))
        transform = from_origin(left, top, r, r)
        same_grid = abs(src.res[0] - r) < 1e-6
        out = np.zeros((4, height, width), dtype=np.uint16)
        for i, band_idx in enumerate(spec["order"]):
            arr = src.read(band_idx).astype(np.float32)
            arr[arr == NODATA_IN] = np.nan
            dst = np.full((height, width), np.nan, dtype=np.float32)
            reproject(arr, dst, src_transform=src.transform, src_crs=src.crs,
                      dst_transform=transform, dst_crs=TARGET_CRS,
                      src_nodata=np.nan, dst_nodata=np.nan,
                      resampling=Resampling.nearest if same_grid else Resampling.cubic)
            out[i] = np.clip(np.nan_to_num(dst, nan=0), 0, 65534).astype(np.uint16)
    valid = np.all(out > 0, axis=0)
    out[:, ~valid] = 0
    profile = dict(driver="GTiff", dtype="uint16", count=4, width=width, height=height,
                   crs=TARGET_CRS, transform=transform, nodata=0,
                   compress="lzw", tiled=True, blockxsize=256, blockysize=256)
    with rasterio.open(P.data / f"{name}_RGBN.tif", "w", **profile) as dst:
        dst.write(out)
        dst.descriptions = ("Red", "Green", "Blue", "NIR")

    stats = {}
    for i, band in enumerate(["R", "G", "B", "NIR"]):
        v = out[i][valid].astype(np.float32)
        lo, hi = np.percentile(v, [1, 99])
        stats[band] = dict(p1=float(lo), p99=float(hi), mean=float(v.mean()), std=float(v.std()))
    ndvi = (out[3].astype(np.float32) - out[0]) / (out[3].astype(np.float32) + out[0] + 1e-6)
    stats["ndvi_median"] = float(np.median(ndvi[valid]))
    with open(P.data / "stats.json", "w") as f:
        json.dump({name: stats}, f, indent=2)
    log(f"  {width} x {height} px à {TARGET_RES} m | NDVI médian du sable = {stats['ndvi_median']:.3f} "
        f"(≈ 0,12 attendu : ordre des bandes correct)")


# =============================================================================
# ÉTAPE 2 — Rasterisation de la vérité terrain
# =============================================================================
def step2_rasterize(P):
    import geopandas as gpd
    from rasterio.features import rasterize
    from shapely.geometry import box

    log("\n=== ÉTAPE 2 : rasterisation de la vérité terrain ===")
    if not P.labels.exists():
        log(f"  labels absents ({P.labels}) : étape ignorée, prédiction avec les poids fournis")
        return False

    def read_layer(layer):
        try:
            gdf = gpd.read_file(P.labels, layer=layer)
        except Exception:
            return None
        if gdf.empty:
            return None
        gdf = gdf.to_crs(TARGET_CRS)
        gdf = gdf[gdf.geometry.notna() & ~gdf.geometry.is_empty].copy()
        gdf["geometry"] = gdf.geometry.buffer(0)
        return gdf

    def burn(label, gdf, value, transform, extent):
        if gdf is None:
            return
        sub = gdf[gdf.intersects(extent)]
        if sub.empty:
            return
        mask = rasterize(((g, 1) for g in sub.geometry), out_shape=label.shape,
                         transform=transform, fill=0, dtype="uint8", all_touched=False)
        label[mask == 1] = value

    layers = {n: read_layer(n) for n in ["aoi", "pit", "spoil", "vegetation", "ignore"]}
    if layers["aoi"] is None:
        sys.exit("ERREUR : la couche 'aoi' est vide.")
    log(f"  {0 if layers['pit'] is None else len(layers['pit'])} fosses digitalisées")
    if layers["spoil"] is None and layers["pit"] is not None and SPOIL_BUFFER_M > 0:
        ring = layers["pit"].copy()
        ring["geometry"] = ring.geometry.buffer(SPOIL_BUFFER_M).difference(ring.geometry)
        layers["spoil"] = ring

    for name in IMAGES:
        is_train = IMAGES[name]["role"] == "train"
        img, valid, profile = read_rgbn(P, name)
        transform, shape = profile["transform"], (profile["height"], profile["width"])
        extent = box(*rasterio.transform.array_bounds(shape[0], shape[1], transform))
        label = np.full(shape, IGNORE, dtype=np.uint8)
        aoi = np.zeros(shape, dtype=np.uint8)
        burn(aoi, layers["aoi"], 1, transform, extent)
        if aoi.sum() == 0:
            log(f"  [{name}] pas de vérité de référence (prédiction seule)")
            continue
        label[aoi == 1] = 0
        if layers["vegetation"] is not None:
            tmp = np.zeros(shape, np.uint8)
            burn(tmp, layers["vegetation"], 1, transform, extent)
            label[(tmp == 1) & (aoi == 1)] = 3
        elif is_train:
            f = img.astype(np.float32)
            ndvi = (f[3] - f[0]) / (f[3] + f[0] + 1e-6)
            label[(ndvi > NDVI_VEG_THRESHOLD) & (aoi == 1)] = 3
        for layer, value in [("spoil", 2), ("pit", 1)]:
            tmp = np.zeros(shape, np.uint8)
            burn(tmp, layers[layer], 1, transform, extent)
            label[(tmp == 1) & (aoi == 1)] = value
        burn(label, layers["ignore"], IGNORE, transform, extent)
        if name == "val_sudan":
            burn(label, gpd.GeoDataFrame(geometry=[box(*OVERLAP_TRAIN_VAL)], crs=TARGET_CRS),
                 IGNORE, transform, extent)
        label[~valid] = IGNORE
        prof = profile.copy()
        prof.update(count=1, dtype="uint8", nodata=IGNORE, compress="lzw")
        with rasterio.open(P.data / f"{name}_label.tif", "w", **prof) as dst:
            dst.write(label, 1)
        counts = {CLASSES.get(k, "ignore"): int((label == k).sum()) for k in [0, 1, 2, 3, 255]}
        log(f"  [{name}] pixels par classe : {counts}")
    return True


def load_label(P, name):
    path = P.data / f"{name}_label.tif"
    if not path.exists():
        return None
    with rasterio.open(path) as src:
        return src.read(1)


# =============================================================================
# Modèle, augmentations, inférence
# =============================================================================
def build_model(pretrained=True):
    import segmentation_models_pytorch as smp
    weights = ENCODER_WEIGHTS if pretrained and not os.environ.get("PILLAGE_NO_PRETRAIN") else None
    return smp.create_model(arch=ARCH, encoder_name=ENCODER, encoder_weights=weights,
                            in_channels=IN_CHANNELS, classes=N_CLASSES)


def seed_everything(seed):
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def augment(x, y, rng):
    """Rotations/miroirs (D4) + radiométrie + flou + bruit."""
    k, flip = rng.integers(4), rng.random() < 0.5
    x = np.rot90(x, k, axes=(1, 2))
    y = np.rot90(y, k, axes=(0, 1))
    if flip:
        x, y = x[:, :, ::-1], y[:, ::-1]
    x, y = x.copy(), y.copy()
    nb = min(4, x.shape[0])
    if rng.random() < 0.8:
        gain = rng.uniform(0.8, 1.2) * rng.uniform(0.95, 1.05, size=(nb, 1, 1))
        bias = rng.uniform(-0.3, 0.3) + rng.uniform(-0.1, 0.1, size=(nb, 1, 1))
        x[:nb] = x[:nb] * gain + bias
    if rng.random() < 0.3:
        g = rng.uniform(0.8, 1.25)
        z = np.clip(x[:nb] * 0.25 + 0.5, 0, None)
        x[:nb] = (z ** g - 0.5) / 0.25
    if rng.random() < 0.2:
        from scipy.ndimage import gaussian_filter
        s = rng.uniform(0.3, 1.0)
        x = np.stack([gaussian_filter(b, s) for b in x])
    if rng.random() < 0.3:
        x = x + rng.normal(0, rng.uniform(0.02, 0.08), size=x.shape).astype(np.float32)
    return np.ascontiguousarray(x, dtype=np.float32), np.ascontiguousarray(y)


def random_rescale(x, y, scale):
    import torch
    import torch.nn.functional as F
    xt = torch.from_numpy(x)[None]
    yt = torch.from_numpy(y.astype(np.int64))[None, None].float()
    size = (int(round(x.shape[1] * scale)), int(round(x.shape[2] * scale)))
    xt = F.interpolate(xt, size=size, mode="bilinear", align_corners=False)
    yt = F.interpolate(yt, size=size, mode="nearest")
    return xt[0].numpy(), yt[0, 0].numpy().astype(np.uint8)


def gaussian_window(size, sigma_scale=1 / 4):
    ax = np.arange(size) - (size - 1) / 2
    g = np.exp(-(ax ** 2) / (2 * (size * sigma_scale) ** 2))
    w = np.outer(g, g)
    return (w / w.max()).astype(np.float32) + 1e-3


TTA_SETS = {"none": [(0, False)], "flip": [(0, False), (0, True)],
            "d4": [(k, f) for k in range(4) for f in (False, True)]}


def predict_tta(model, batch, tta):
    import torch
    out = 0
    for k, flip in TTA_SETS[tta]:
        b = torch.rot90(batch, k, dims=(2, 3))
        if flip:
            b = torch.flip(b, dims=(3,))
        p = torch.softmax(model(b), dim=1)
        if flip:
            p = torch.flip(p, dims=(3,))
        out = out + torch.rot90(p, -k, dims=(2, 3))
    return out / len(TTA_SETS[tta])


def sliding_window(model, img, stride, tta, device, batch_size=8):
    """Fenêtre glissante PATCH px avec fondu gaussien -> probabilités (N_CLASSES x H x W)."""
    import torch
    _, H, W = img.shape
    ph = max(0, PATCH - H) + (-(max(H, PATCH) - PATCH)) % stride
    pw = max(0, PATCH - W) + (-(max(W, PATCH) - PATCH)) % stride
    padded = np.pad(img, ((0, 0), (0, ph), (0, pw)), mode="reflect")
    _, Hp, Wp = padded.shape
    probs = np.zeros((N_CLASSES, Hp, Wp), np.float32)
    weights = np.zeros((Hp, Wp), np.float32)
    win = gaussian_window(PATCH)
    coords = [(r, c) for r in range(0, Hp - PATCH + 1, stride) for c in range(0, Wp - PATCH + 1, stride)]
    model.eval()
    with torch.no_grad():
        for i in range(0, len(coords), batch_size):
            chunk = coords[i:i + batch_size]
            batch = torch.from_numpy(np.stack([padded[:, r:r + PATCH, c:c + PATCH] for r, c in chunk])).to(device)
            with torch.autocast(device_type="cuda", enabled=(device == "cuda")):
                p = predict_tta(model, batch, tta).float().cpu().numpy()
            for (r, c), pi in zip(chunk, p):
                probs[:, r:r + PATCH, c:c + PATCH] += pi * win
                weights[r:r + PATCH, c:c + PATCH] += win
    probs /= weights[None]
    return probs[:, :H, :W]


def confusion(pred, gt):
    m = gt != IGNORE
    return np.bincount(N_CLASSES * gt[m].astype(np.int64) + pred[m],
                       minlength=N_CLASSES ** 2).reshape(N_CLASSES, N_CLASSES)


def scores_from_confusion(cm):
    tp = np.diag(cm).astype(float)
    fp, fn = cm.sum(0) - tp, cm.sum(1) - tp
    return tp / np.maximum(tp + fp + fn, 1), 2 * tp / np.maximum(2 * tp + fp + fn, 1)


# =============================================================================
# ÉTAPE 3 — Entraînement
# =============================================================================
class CropDataset:
    """Extraits aléatoires CROP x CROP, dont P_CENTER_ON_PIT centrés sur une fosse."""

    def __init__(self, img, lab, n_samples, crop, seed):
        self.img, self.lab, self.n, self.crop = img, lab, n_samples, crop
        self.rng = np.random.default_rng(seed)
        self.pit_px = np.argwhere(lab == 1)
        self.H, self.W = lab.shape

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        import torch
        rng = self.rng
        scale = rng.uniform(0.85, 1.15)
        size = min(int(round(self.crop / scale)), self.W, self.H)
        for _ in range(50):
            if len(self.pit_px) and rng.random() < P_CENTER_ON_PIT:
                pr, pc = self.pit_px[rng.integers(len(self.pit_px))]
                r = int(np.clip(pr - rng.integers(size // 4, 3 * size // 4), 0, self.H - size))
                c = int(np.clip(pc - rng.integers(size // 4, 3 * size // 4), 0, self.W - size))
            else:
                r = int(rng.integers(0, max(1, self.H - 1 - size + 2)))
                c = int(rng.integers(0, self.W - size + 1))
            r = min(r, self.H - size)
            y = self.lab[r:r + size, c:c + size]
            if (y != IGNORE).mean() > 0.3:
                break
        else:
            r = int(rng.integers(0, self.H - size + 1))
            c = int(rng.integers(0, self.W - size + 1))
            y = self.lab[r:r + size, c:c + size]
        x = self.img[:, r:r + size, c:c + size]
        x, y = random_rescale(x, y, self.crop / size)
        x, y = augment(x, y, rng)
        return torch.from_numpy(x), torch.from_numpy(y.astype(np.int64))


def class_weights(lab_train):
    """Fond = 1 ; classe rare = sqrt(n_fond / n_classe), plafonné à 50 ; classe absente = 0."""
    import torch
    counts = np.array([(lab_train == k).sum() for k in range(N_CLASSES)], float)
    w = np.zeros(N_CLASSES)
    present = counts > 0
    w[present] = np.sqrt(counts[0] / counts[present])
    return torch.tensor(np.clip(w, 0, 50), dtype=torch.float32)


def train_one(P, seed, epochs, device):
    import torch
    import torch.nn as nn
    import segmentation_models_pytorch as smp
    from torch.utils.data import DataLoader

    seed_everything(seed)
    img, _, _ = load_image(P, "train_sudan")
    lab = load_label(P, "train_sudan")
    w = class_weights(lab[lab != IGNORE])
    log(f"  [graine {seed}] poids de classes : {w.numpy().round(2)}")

    ds = CropDataset(img, lab, SAMPLES_PER_EPOCH, CROP, seed)
    dl = DataLoader(ds, batch_size=BATCH, shuffle=True, num_workers=NUM_WORKERS,
                    drop_last=True, pin_memory=(device == "cuda"))
    model = build_model().to(device)
    ce = nn.CrossEntropyLoss(weight=w.to(device), ignore_index=IGNORE, label_smoothing=0.0)
    dice = smp.losses.DiceLoss("multiclass", ignore_index=IGNORE)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    steps, warm = epochs * len(dl), 5 * len(dl)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else
        0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, steps - warm))))
    scaler = torch.amp.GradScaler("cuda", enabled=(device == "cuda"))
    history = []
    for ep in range(1, epochs + 1):
        model.train()
        t0, tot = time.time(), 0.0
        for x, y in dl:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            if (y != IGNORE).sum() == 0:
                continue
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", enabled=(device == "cuda")):
                logits = model(x)
                loss = ce(logits, y) + dice(logits, y)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            tot += loss.item()
        history.append(round(tot / len(dl), 4))
        if ep == 1 or ep % 10 == 0 or ep == epochs:
            log(f"  [graine {seed}] époque {ep:3d}/{epochs}  perte {history[-1]:.3f}  ({time.time() - t0:.0f} s)")
    torch.save(model.state_dict(), P.model(seed))
    with open(P.out / f"historique_perte_s{seed}.json", "w") as f:
        json.dump(history, f)
    log(f"  modèle enregistré : {P.model(seed)}")


def try_download_model(P, seed):
    """Télécharge les poids publiés (GitHub Releases) s'ils sont absents."""
    if not MODELS_URL:
        return False
    import torch
    url = MODELS_URL + P.model(seed).name
    try:
        log(f"  [graine {seed}] téléchargement des poids : {url}")
        torch.hub.download_url_to_file(url, str(P.model(seed)), progress=True)
        return True
    except Exception as e:
        log(f"  [graine {seed}] téléchargement impossible ({e})")
        if P.model(seed).exists():
            P.model(seed).unlink()
        return False


def step3_train(P, retrain, epochs, device, have_labels):
    log("\n=== ÉTAPE 3 : entraînement U-Net / ResNet-50 (image TRAIN uniquement) ===")
    for seed in SEEDS:
        if not retrain and (P.model(seed).exists() or try_download_model(P, seed)):
            log(f"  [graine {seed}] poids pré-entraînés utilisés : {P.model(seed).name}")
            continue
        if not have_labels:
            sys.exit("ERREUR : ni poids entraînés (modeles/) ni vérité terrain (labels/) : "
                     "impossible d'entraîner. Voir README.")
        train_one(P, seed, epochs, device)


def load_models(P, device):
    import torch
    models = []
    for seed in SEEDS:
        m = build_model(pretrained=False).to(device)
        m.load_state_dict(torch.load(P.model(seed), map_location=device))
        m.eval()
        models.append(m)
    return models


# =============================================================================
# ÉTAPE 4 — Prédiction sur les validations
# =============================================================================
def postprocess(prob_pit, valid, threshold, res):
    from scipy import ndimage as ndi
    binary = (prob_pit >= threshold) & valid
    binary = ndi.binary_opening(binary, structure=ndi.generate_binary_structure(2, 1))
    binary = ndi.binary_fill_holes(binary)
    min_px = int(round(MIN_PIT_AREA_M2 / (res ** 2)))
    lab, n = ndi.label(binary)
    if n == 0:
        return binary
    sizes = np.bincount(lab.ravel())
    keep = sizes >= min_px
    keep[0] = False
    return keep[lab]


def vectorize(binary, prob_pit, transform, crs, path):
    import geopandas as gpd
    from rasterio.features import shapes
    from scipy import ndimage as ndi
    from shapely.geometry import shape
    labels, n = ndi.label(binary)
    if n == 0:
        return 0
    mean_p = ndi.mean(prob_pit, labels, index=np.arange(1, n + 1))
    geoms, ids = [], []
    for geom, val in shapes(labels.astype(np.int32), mask=labels > 0, transform=transform):
        geoms.append(shape(geom))
        ids.append(int(val))
    gdf = gpd.GeoDataFrame({"id": ids}, geometry=geoms, crs=crs).dissolve(by="id").reset_index()
    gdf["surface_m2"] = gdf.area.round(2)
    gdf["diam_eq_m"] = (2 * np.sqrt(gdf.area / np.pi)).round(2)
    gdf["circularite"] = (4 * np.pi * gdf.area / gdf.length ** 2).round(3)
    gdf["proba_moy"] = [round(float(mean_p[i - 1]), 3) for i in gdf["id"]]
    if path.exists():
        path.unlink()
    gdf.to_file(path, layer="fosses_detectees", driver="GPKG")
    return len(gdf)


def write_tif(path, arr, profile, dtype, nodata=None):
    p = profile.copy()
    p.update(count=arr.shape[0] if arr.ndim == 3 else 1, dtype=dtype, nodata=nodata,
             compress="lzw", tiled=True, blockxsize=256, blockysize=256)
    p.pop("photometric", None)
    with rasterio.open(path, "w", **p) as dst:
        dst.write(arr if arr.ndim == 3 else arr[None])


def save_figure(P, name, valid, binary):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy import ndimage as ndi
    raw, _, _ = read_rgbn(P, name)
    rgb = raw[:3].astype(float)
    lo, hi = np.percentile(rgb[:, valid], [2, 98])
    rgb = np.clip((rgb - lo) / (hi - lo), 0, 1).transpose(1, 2, 0)
    rgb[~valid] = 1
    cy, cx = np.array(ndi.center_of_mass(binary, *ndi.label(binary))).reshape(-1, 2).T \
        if binary.any() else (np.array([]), np.array([]))
    fig, ax = plt.subplots(figsize=(12, 12 * rgb.shape[0] / rgb.shape[1]))
    ax.imshow(rgb)
    ax.scatter(cx, cy, s=60, facecolors="none", edgecolors="red", linewidths=1.2)
    ax.set_title(f"{name} : {len(cx)} fosses de pillage détectées")
    ax.axis("off")
    fig.savefig(P.fig / f"{name}_detections.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def step4_predict(P, device, stride, tta):
    log("\n=== ÉTAPE 4 : prédiction sur les images de VALIDATION (lecture seule) ===")
    models = load_models(P, device)
    results = {}
    for name in VALIDATIONS:
        t0 = time.time()
        img, valid, profile = load_image(P, name)
        probs = sum(sliding_window(m, img, stride, tta, device) for m in models) / len(models)
        probs[:, ~valid] = 0
        np.save(P.pred / f"{name}_probs.npy", probs.astype(np.float16))
        cls = probs.argmax(0).astype(np.uint8)
        cls[~valid] = 255
        write_tif(P.pred / f"{name}_carte_4classes.tif", cls, profile, "uint8", 255)
        for k, lab in [(1, "fosses"), (2, "deblais"), (3, "vegetation")]:
            write_tif(P.pred / f"{name}_carte_{lab}.tif", (cls == k).astype(np.uint8), profile, "uint8")
        write_tif(P.pred / f"{name}_probabilite_pillage.tif",
                  np.clip(probs[1] + probs[2], 0, 1).astype(np.float32), profile, "float32")
        res = pixel_size(profile)
        binary = postprocess(probs[1], valid, PIT_THRESHOLD, res)
        write_tif(P.pred / f"{name}_fosses_binaire.tif", binary.astype(np.uint8), profile, "uint8")
        n = vectorize(binary, probs[1], profile["transform"], profile["crs"],
                      P.pred / f"{name}_fosses.gpkg")
        save_figure(P, name, valid, binary)
        results[name] = dict(fosses_detectees=n, resolution_m=round(res, 3),
                             duree_s=round(time.time() - t0, 1))
        log(f"  [{name}] {n} fosses détectées  ({results[name]['duree_s']} s)")
    return results


# =============================================================================
# ÉTAPE 5 — Diagnostic + résumé
# =============================================================================
def object_scores(pred_bin, gt_pit, eval_mask, res):
    from scipy import ndimage as ndi
    from scipy.spatial import cKDTree

    def cent(b):
        lab, n = ndi.label(b)
        return np.zeros((0, 2)) if n == 0 else np.array(ndi.center_of_mass(b, lab, range(1, n + 1)))
    gt_c, pr_c = cent(gt_pit & eval_mask), cent(pred_bin)
    if len(pr_c):
        pr_c = pr_c[eval_mask[pr_c[:, 0].astype(int), pr_c[:, 1].astype(int)]]
    tp = 0
    if len(gt_c) and len(pr_c):
        d, j = cKDTree(gt_c).query(pr_c, distance_upper_bound=MATCH_DIST_M / res)
        used = set()
        for i in np.argsort(d):
            if np.isfinite(d[i]) and j[i] not in used:
                used.add(j[i])
                tp += 1
    fp, fn = len(pr_c) - tp, len(gt_c) - tp
    p, r = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
    return dict(precision=round(p, 3), rappel=round(r, 3), f1=round(2 * p * r / max(p + r, 1e-9), 3),
                vrais_pos=tp, faux_pos=fp, faux_neg=fn)


def step5_summary(P, device, results):
    log("\n=== ÉTAPE 5 : diagnostic et résumé ===")
    summary = {"validations": results}
    # Diagnostic sur l'image TRAIN (ajustement du modèle à la vérité terrain)
    lab = load_label(P, "train_sudan")
    diag = "non calculé (labels absents)"
    if lab is not None:
        models = load_models(P, device)
        img, valid, profile = load_image(P, "train_sudan")
        probs = sum(sliding_window(m, img, 256, "none", device) for m in models) / len(models)
        iou, f1 = scores_from_confusion(confusion(probs.argmax(0), lab))
        summary["diagnostic_train"] = {CLASSES[k]: dict(iou=round(float(iou[k]), 3), f1=round(float(f1[k]), 3))
                                       for k in (0, 1, 3)}
        diag = f"IoU fosse {iou[1]:.3f} · F1 fosse {f1[1]:.3f}"
    log(f"  TRAIN : {diag}")
    # Scores sur les validations si une vérité de référence y a été digitalisée
    for name in VALIDATIONS:
        gt = load_label(P, name)
        if gt is None or (gt != IGNORE).sum() == 0:
            continue
        pr = np.load(P.pred / f"{name}_probs.npy").astype(np.float32)
        _, v, prof = read_rgbn(P, name)
        res = pixel_size(prof)
        b = postprocess(pr[1], v, PIT_THRESHOLD, res)
        summary["validations"][name]["scores_objet"] = object_scores(b, gt == 1, gt != IGNORE, res)
    with open(P.out / "resume_resultats.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    lines = ["RÉSUMÉ — détection des fosses de pillage (U-Net / ResNet-50)", "=" * 62,
             f"Diagnostic TRAIN : {diag}"]
    for name in VALIDATIONS:
        r = summary["validations"][name]
        lines.append(f"{name:10s} : {r['fosses_detectees']} fosses détectées"
                     + (f" · scores {r['scores_objet']}" if "scores_objet" in r else ""))
    lines += ["", f"Cartes et GeoPackages : {P.pred}", f"Figures : {P.fig}"]
    txt = "\n".join(lines)
    (P.out / "resume_resultats.txt").write_text(txt, encoding="utf-8")
    log("\n" + txt)


# =============================================================================
# Programme principal
# =============================================================================
def main():
    global EPOCHS, SAMPLES_PER_EPOCH
    try:  # accents lisibles dans tous les terminaux (Windows compris)
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    ap = argparse.ArgumentParser(description="Détection des fosses de pillage (Pléiades, U-Net/ResNet-50)")
    ap.add_argument("--root", help="dossier contenant Train_Image/ et Validation_image/")
    ap.add_argument("--retrain", action="store_true", help="réentraîner même si modeles/*.pt existent")
    ap.add_argument("--test-rapide", action="store_true", help="contrôle express (1 époque, sans TTA)")
    args = ap.parse_args()

    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    stride, tta, epochs = STRIDE, TTA, EPOCHS
    if args.test_rapide:
        epochs, SAMPLES_PER_EPOCH, stride, tta = 1, 16, 512, "none"
    P = Paths(find_root(args.root), test=args.test_rapide)
    log(f"Données : {P.root}\nSorties : {P.out}\nCalcul  : {device}"
        + (f" ({torch.cuda.get_device_name(0)})" if device == "cuda" else
           "  -> pas de GPU : les calculs seront très longs (voir README)"))

    t0 = time.time()
    step1_preprocess(P)
    have_labels = step2_rasterize(P)
    step3_train(P, args.retrain or args.test_rapide, epochs, device, have_labels)
    results = step4_predict(P, device, stride, tta)
    step5_summary(P, device, results)
    log(f"\nTerminé en {(time.time() - t0) / 60:.1f} min.")


if __name__ == "__main__":
    main()
