"""Shared helpers for the Mapping the Unseen Seafloor project."""
import urllib.request, urllib.parse
from pathlib import Path
import numpy as np

DATA_DIR = Path("data"); DATA_DIR.mkdir(exist_ok=True)
FIG_DIR = Path("figures"); FIG_DIR.mkdir(exist_ok=True)

REGION = dict(west=140.0, east=148.0, south=9.0, north=15.0)

def fetch_gmrt(region=REGION, resolution="med", layer="topo"):
    """Download (once) a GMRT grid as ESRI ASCII. layer='topo' = full grid,
    layer='topo-mask' = only cells measured by ship multibeam sonar."""
    r = region
    out = DATA_DIR / f"gmrt_{r['west']}_{r['east']}_{r['south']}_{r['north']}_{resolution}" \
                     f"{'' if layer == 'topo' else '_' + layer}.asc"
    if out.exists():
        return out
    params = dict(minlongitude=r["west"], maxlongitude=r["east"],
                  minlatitude=r["south"], maxlatitude=r["north"],
                  format="esriascii", resolution=resolution, layer=layer)
    url = "https://www.gmrt.org/services/GridServer?" + urllib.parse.urlencode(params)
    print("Downloading:", url)
    req = urllib.request.Request(url, headers={"User-Agent": "student-seafloor-project"})
    try:
        with urllib.request.urlopen(req, timeout=300) as resp:
            out.write_bytes(resp.read())
    except Exception as e:
        raise RuntimeError(f"Download failed ({e}). Manual fallback: https://www.gmrt.org/GMRTMapTool/ "
                           f"-> export ESRI ASCII -> save as {out.resolve()}") from e
    return out

def read_esri_ascii(path):
    """Returns lon (1D), lat (1D, south->north), z (2D, m, negative = below sea level)."""
    header = {}
    with open(path) as f:
        while True:
            pos = f.tell()
            line = f.readline()
            key = line.split()[0].lower() if line.strip() else ""
            if key and not key.lstrip("-").replace(".", "").isdigit() and key != "nan":
                header[key] = float(line.split()[1])
            else:
                f.seek(pos)
                break
        z = np.loadtxt(f, dtype=np.float32)
    ncols, nrows = int(header["ncols"]), int(header["nrows"])
    z = z.reshape(nrows, ncols)
    dx = header.get("cellsize", header.get("dx"))
    dy = header.get("cellsize", header.get("dy"))
    x0 = header["xllcenter"] if "xllcenter" in header else header["xllcorner"] + dx / 2
    y0 = header["yllcenter"] if "yllcenter" in header else header["yllcorner"] + dy / 2
    nodata = header.get("nodata_value")
    if nodata is not None:
        z[z == nodata] = np.nan
    lon = x0 + dx * np.arange(ncols)
    lat = y0 + dy * np.arange(nrows)
    return lon, lat, z[::-1]

def load(region=REGION, resolution="med", layer="topo"):
    return read_esri_ascii(fetch_gmrt(region, resolution, layer))

def cell_size_m(lon, lat):
    """East-west spacing per row (varies with latitude) and north-south spacing, in meters."""
    dlon, dlat = np.diff(lon).mean(), np.diff(lat).mean()
    dx = dlon * 111_320 * np.cos(np.deg2rad(lat))
    dy = dlat * 110_574
    return dx, dy

def hillshade(z, dx, dy, azimuth=315, altitude=45, exaggeration=3.0):
    z = np.nan_to_num(z, nan=np.nanmean(z)) * exaggeration
    dx = float(np.mean(dx))
    dzdy, dzdx = np.gradient(z, dy, dx)
    n = np.stack([-dzdx, -dzdy, np.ones_like(z)])
    n /= np.linalg.norm(n, axis=0)
    az, alt = np.deg2rad(azimuth), np.deg2rad(altitude)
    L = np.array([np.sin(az) * np.cos(alt), np.cos(az) * np.cos(alt), np.sin(alt)])
    return np.clip(np.tensordot(L, n, axes=1), 0, 1)

def distance_km(lat1, lon1, lat2, lon2):
    """Great-circle distance (haversine)."""
    p1, p2 = np.deg2rad(lat1), np.deg2rad(lat2)
    dp, dl = p2 - p1, np.deg2rad(lon2 - lon1)
    a = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * 6371 * np.arcsin(np.sqrt(a))

FEATURES = ["depth", "slope", "roughness", "bpi_fine", "bpi_broad", "curv_min", "curv_max"]
FEATURE_INFO = {
    "depth": "elevation (m, negative = deep)",
    "slope": "slope angle (°) after 1 km smoothing",
    "roughness": "RMS of small-scale bumps (< ~5 km) in a 5 km window (m)",
    "bpi_fine": "height above the average of the surrounding 5 km (m)",
    "bpi_broad": "height above the average of the surrounding 20 km (m)",
    "curv_min": "Hessian eigenvalue λ1 (most-bent direction), ×1e5 per m",
    "curv_max": "Hessian eigenvalue λ2 (least-bent direction), ×1e5 per m",
}

def load_clean():
    """Cleaned grid (saved by notebook 03), ship mask and cell sizes."""
    from scipy import ndimage
    lon, lat, _ = load()
    z = np.load(DATA_DIR / "z_clean.npy")
    _, _, zm = load(layer="topo-mask")
    ship = ~np.isnan(zm)
    dx, dy = cell_size_m(lon, lat)
    return lon, lat, z, ship, dx, dy

def compute_features(z, dx, dy):
    """One row of numbers per grid cell describing the local seafloor."""
    from scipy import ndimage
    DX = float(np.mean(dx))
    ok = ~np.isnan(z)
    zf = np.where(ok, z, np.nanmedian(z))
    zs = ndimage.gaussian_filter(zf, 1000 / DX)
    gy, gx = np.gradient(zs, dy, DX)
    slope = np.degrees(np.arctan(np.hypot(gx, gy)))
    resid = zf - ndimage.gaussian_filter(zf, 2500 / DX)
    rough = np.sqrt(ndimage.uniform_filter(resid ** 2, int(5000 / DX)))
    bpi_f = zs - ndimage.uniform_filter(zs, int(5000 / DX))
    bpi_b = zs - ndimage.uniform_filter(zs, int(20000 / DX))
    z2 = ndimage.gaussian_filter(zf, 2000 / DX)
    zy, zx = np.gradient(z2, dy, DX)
    zyy, zyx = np.gradient(zy, dy, DX)
    zxy, zxx = np.gradient(zx, dy, DX)
    b = 0.5 * (zxy + zyx); m = 0.5 * (zxx + zyy); r = np.sqrt((0.5 * (zxx - zyy)) ** 2 + b ** 2)
    F = np.stack([zf, slope, rough, bpi_f, bpi_b, (m - r) * 1e5, (m + r) * 1e5], axis=-1)
    F[~ok] = np.nan
    return F

def detect_peaks(z, dx, dy, lat=None, lon=None, sigma_m=1500, r_local_m=3000, r_base_m=15000, h_floor=100):
    """Seamount candidates: local maxima of the smoothed grid, with height above the 10th-percentile
    base and a roundness score from the Hessian (1 = perfectly round, 0 = ridge-like)."""
    from scipy import ndimage
    DX = float(np.mean(dx))
    zf = np.where(np.isnan(z), np.nanmedian(z), z)
    zs = ndimage.gaussian_filter(zf, sigma_m / DX)
    r = int(r_local_m / DX)
    disk = np.hypot(*np.mgrid[-r:r + 1, -r:r + 1]) <= r
    is_max = ndimage.maximum_filter(zs, footprint=disk) == zs
    k = 4
    base = ndimage.percentile_filter(zs[::k, ::k], 10, size=int(2 * r_base_m / (DX * k)) + 1)
    base = ndimage.zoom(base, (zs.shape[0] / base.shape[0], zs.shape[1] / base.shape[1]), order=1)
    height = zs - base[:zs.shape[0], :zs.shape[1]]
    z2 = ndimage.gaussian_filter(zf, 2 * sigma_m / DX)
    zy, zx = np.gradient(z2, dy, DX)
    zyy, zyx = np.gradient(zy, dy, DX)
    zxy, zxx = np.gradient(zx, dy, DX)
    b = 0.5 * (zxy + zyx); m = 0.5 * (zxx + zyy); rr = np.sqrt((0.5 * (zxx - zyy)) ** 2 + b ** 2)
    l1, l2 = m - rr, m + rr
    with np.errstate(divide="ignore", invalid="ignore"):
        roundness = np.clip(np.where(l1 < 0, l2 / l1, 0), 0, 1)
    edge = int(r_base_m / DX / 3)
    valid = ~np.isnan(z)
    valid[:edge] = valid[-edge:] = False; valid[:, :edge] = valid[:, -edge:] = False
    ii, jj = np.nonzero(is_max & valid & (height > h_floor))
    out = dict(i=ii, j=jj, h=height[ii, jj], round=roundness[ii, jj])
    if lat is not None: out["lat"] = lat[ii]
    if lon is not None: out["lon"] = lon[jj]
    return out

def load_catalog(lon, lat):
    """Gevorgian et al. (2023) 'good' seamount catalog, restricted to the grid's area."""
    p = next((DATA_DIR / "SIO_Seamounts").rglob("good.xyhrdnc"))
    rows = []
    for line in p.read_text(errors="ignore").splitlines():
        s = line.split()
        if len(s) < 7: continue
        try: rows.append([float(s[0]) % 360, float(s[1]), float(s[2]), float(s[6])])
        except ValueError: pass
    c = np.array(rows)
    k = (c[:, 0] > lon[0]) & (c[:, 0] < lon[-1]) & (c[:, 1] > lat[0]) & (c[:, 1] < lat[-1])
    c = c[k]
    return dict(lon=c[:, 0], lat=c[:, 1], h=c[:, 2], charted=c[:, 3] == 1)

def match(d_lat, d_lon, c_lat, c_lon, max_km=10):
    """One-to-one matching within max_km (Hungarian algorithm). Returns matched masks."""
    from scipy.optimize import linear_sum_assignment
    dm = np.zeros(len(d_lat), bool); cm = np.zeros(len(c_lat), bool)
    if len(d_lat) == 0 or len(c_lat) == 0:
        return dm, cm
    D = distance_km(d_lat[:, None], d_lon[:, None], c_lat[None, :], c_lon[None, :])
    cost = np.where(D <= max_km, D, 1e6)
    r, c = linear_sum_assignment(cost)
    ok = cost[r, c] <= max_km
    dm[r[ok]] = True; cm[c[ok]] = True
    return dm, cm

def block_spectra(z, ship, dx, dy, N=64, detrend=True, window=True, extra_mask=None, n_bins=16):
    """Radially averaged power spectra of N x N blocks that are fully ship (>=99%) or fully satellite (<=1%).
    Returns wavelengths (km), log-power arrays for ship and satellite blocks, and block centres."""
    DX = float(np.mean(dx))
    win = np.outer(np.hanning(N), np.hanning(N)) if window else np.ones((N, N))
    kx, ky = np.meshgrid(np.fft.fftfreq(N, DX), np.fft.fftfreq(N, dy))
    K = np.hypot(kx, ky)
    edges = np.logspace(np.log10(1.5 / (N * DX)), np.log10(0.5 / min(DX, dy)), n_bins + 1)
    keep = np.array([((K >= edges[i]) & (K < edges[i + 1])).sum() > 0 for i in range(n_bins)])
    lo_e, hi_e = edges[:-1][keep], edges[1:][keep]
    wl = 1 / np.sqrt(lo_e * hi_e) / 1000
    yy, xx = np.mgrid[:N, :N]; A = np.c_[xx.ravel(), yy.ravel(), np.ones(N * N)]
    out = {True: [], False: []}; centres = {True: [], False: []}
    for i in range(0, z.shape[0] - N + 1, N):
        for j in range(0, z.shape[1] - N + 1, N):
            b = z[i:i + N, j:j + N]
            if np.isnan(b).any(): continue
            if extra_mask is not None and extra_mask[i:i + N, j:j + N].mean() < 0.7: continue
            s = ship[i:i + N, j:j + N].mean()
            if 0.01 < s < 0.99: continue
            if detrend:
                coef, *_ = np.linalg.lstsq(A, b.ravel(), rcond=None)
                b = b - (A @ coef).reshape(N, N)
            else:
                b = b - b.mean()
            P = np.abs(np.fft.fft2(b * win)) ** 2
            out[s >= 0.99].append(np.log([P[(K >= a) & (K < c)].mean() for a, c in zip(lo_e, hi_e)]))
            centres[s >= 0.99].append((i + N // 2, j + N // 2))
    return wl, np.array(out[True]), np.array(out[False]), centres

def half_power_wavelength(wl, S_ship, S_sat):
    """Wavelength (km) where satellite/ship power ratio first drops below 0.5 (interpolated, log scale)."""
    R = np.exp(S_sat.mean(0) - S_ship.mean(0))
    order = np.argsort(-wl); wl, R = wl[order], R[order]
    below = np.nonzero(R < 0.5)[0]
    if len(below) == 0: return np.nan
    k = below[0]
    if k == 0: return wl[0]
    x0, x1 = np.log(wl[k - 1]), np.log(wl[k]); y0, y1 = R[k - 1], R[k]
    return float(np.exp(x0 + (0.5 - y0) * (x1 - x0) / (y1 - y0)))
