import os
import sys
import math
import ROOT

if __name__ == "__main__" and "ANALYSIS_PATH" in os.environ:
    sys.path.append(os.environ["ANALYSIS_PATH"])

from FLAF.Common.HistHelper import *    
from FLAF.Common.Utilities import *    


# Backgrounds
_DEFAULT_BKG = ["DY", "ST", "TT", "VV", "WtoLNu", "W", "ttH", "VH", "H"]

_REGIONS = {
    "A": "OS_Iso",
    "B": "OS_AntiIso",
    "C": "SS_Iso",
    "D": "SS_AntiIso",
}


def _key(channel, region, category, uncName, scale):
    return ((channel, region, category), (uncName, scale))


def _safe_clone(h, name_suffix=""):
    if h is None:
        return None
    name = h.GetName()
    if name_suffix:
        return h.Clone(f"{name}_{name_suffix}")
    return h.Clone()


def _pick_sample(histograms, base_name, *, ml=False):
    if ml and (base_name + "_ML_shape") in histograms:
        return histograms[base_name + "_ML_shape"]
    return histograms.get(base_name, {})


def _auto_background_list(all_samples_list):
    print(f"[QCD] Auto background list from samples: {all_samples_list}")
    out = []
    for s in all_samples_list:
        if s == "data" or s == "QCD":
            continue
        if s.endswith("_ML_shape"):
            continue
        if s.startswith("GluGlu"):   
            continue
        if s.startswith("VBFHH"):   
            continue
        out.append(s)
    if not out:
        out = [s for s in _DEFAULT_BKG if s in all_samples_list]
    return out


def _integral(h):
    if h is None:
        return 0.0
    return float(h.Integral()) 


def _ab_subtract_backgrounds(histograms, backgrounds, key):
    if "data" not in histograms or key not in histograms["data"]:
        raise RuntimeError(f"[QCD] Missing data histogram for key={key}")
    h = _safe_clone(histograms["data"][key], "dataOnly")
    for b in backgrounds:
        hmap = histograms.get(b, {})
        if key in hmap:
            h.Add(hmap[key], -1.0)
    return h


def _ml_shape_for_A(histograms, backgrounds, key_A, verbose=True):
    data_ml_map = _pick_sample(histograms, "data", ml=True)
    if key_A not in data_ml_map:
        raise RuntimeError(f"[QCD] Missing data(ML) shape for key {key_A}")
    h_shape = _safe_clone(data_ml_map[key_A], "qcd_shape")

    if verbose:
        print(f"[QCD][ML] start ML shape: data_ML_shape int={_integral(h_shape):.3f}")

    for b in backgrounds:
        bmap_ml = _pick_sample(histograms, b, ml=True)
        if key_A in bmap_ml:
            val = _integral(bmap_ml[key_A])
            if verbose:
                print(f"  subtract {b}_ML_shape int={val:.3f}")
            h_shape.Add(bmap_ml[key_A], -1.0)
        else:
            bmap_nom = histograms.get(b, {})
            if key_A in bmap_nom:
                val = _integral(bmap_nom[key_A])
                if verbose:
                    print(f"  subtract {b} (nominal) int={val:.3f}")
                h_shape.Add(bmap_nom[key_A], -1.0)

    if verbose:
        print(f"  => ML shape after subtraction int={_integral(h_shape):.3f}")
    return h_shape

def QCD_Estimation_ABCD(histograms, all_samples_list, channel, category, uncName, scale, wantNegativeContributions):
    kB = _key(channel, _REGIONS["B"], category, uncName, scale)
    kC = _key(channel, _REGIONS["C"], category, uncName, scale)
    kD = _key(channel, _REGIONS["D"], category, uncName, scale)

    bkg = _auto_background_list(all_samples_list)

    hB = _ab_subtract_backgrounds(histograms, bkg, kB)
    hC = _ab_subtract_backgrounds(histograms, bkg, kC)
    hD = _ab_subtract_backgrounds(histograms, bkg, kD)

    nB, nC, nD = _integral(hB), _integral(hC), _integral(hD)
    if nD <= 0 or nB < 0 or nC < 0:
        print(f"[QCD][ABCD] WARN yields: B'={nB:.3f}, C'={nC:.3f}, D'={nD:.3f}")

    # Central
    h_central = _safe_clone(hB, "QCD_ABCD_central")
    sf = (nC / nD) if nD > 0 else 0.0
    h_central.Scale(sf)

    # Up/Down placeholders 
    h_up   = _safe_clone(h_central, "Up")
    h_down = _safe_clone(h_central, "Down")

    # Simple Poisson like
    err_norm = math.sqrt(abs(_integral(h_central))) if _integral(h_central) > 0 else 0.0
    err_var  = err_norm

    # Optional negativity fix
    if wantNegativeContributions:
        ok, dbg, neg = FixNegativeContributions(h_central)
        if not ok:
            print("[QCD][ABCD] Negative bins after estimate. Zeroing histogram.")
            h_central.Scale(0.0)
            h_up.Scale(0.0)
            h_down.Scale(0.0)

    return h_central, h_up, h_down, err_norm, err_var


def QCD_Estimation_ML(histograms, all_samples_list, channel, category, uncName, scale, wantNegativeContributions):
    """
    Hybrid ML + ABCD:
      - Use ABCD to get target yield: (B'/D')*C'
      - Take ML-weighted 'A' shape from data, subtract prompt MC ML-shapes
      - Scale that shape to match the ABCD target yield
    """
    kA = _key(channel, _REGIONS["A"], category, uncName, scale)
    kB = _key(channel, _REGIONS["B"], category, uncName, scale)
    kC = _key(channel, _REGIONS["C"], category, uncName, scale)
    kD = _key(channel, _REGIONS["D"], category, uncName, scale)

    bkg = _auto_background_list(all_samples_list)
    print(f"[QCD][ML] backgrounds used: {bkg}")

    hB = _ab_subtract_backgrounds(histograms, bkg, kB)
    hC = _ab_subtract_backgrounds(histograms, bkg, kC)
    hD = _ab_subtract_backgrounds(histograms, bkg, kD)
    nB, nC, nD = _integral(hB), _integral(hC), _integral(hD)
    print(f"[QCD][ML] B'={nB:.3f}, C'={nC:.3f}, D'={nD:.3f}")

    target = (nB / nD) * nC if (nD > 0 and nC > 0 and nB >= 0) else 0.0
    print(f"[QCD][ML] target yield (B'/D')*C' = {target:.3f}")

    # ML template in A
    try:
        h_shape = _ml_shape_for_A(histograms, bkg, kA)
    except RuntimeError as e:
        print(f"[QCD][ML] {e}")
        h_fallback = _safe_clone(hB, "fallback_shape")
        sf = (nC / nD) if nD > 0 else 0.0
        h_fallback.Scale(sf)
        h_up = _safe_clone(h_fallback, "Up")
        h_down = _safe_clone(h_fallback, "Down")
        err = math.sqrt(abs(_integral(h_fallback))) if _integral(h_fallback) > 0 else 0.0
        return h_fallback, h_up, h_down, err, err

    shape_int = _integral(h_shape)
    if shape_int > 0 and target > 0:
        h_shape.Scale(target / shape_int)
    else:
        print(f"[QCD][ML] WARN non-positive (target={target:.3f}, shape_int={shape_int:.3f}). Zeroing.")
        h_shape.Scale(0.0)

    h_up   = _safe_clone(h_shape, "Up")
    h_down = _safe_clone(h_shape, "Down")
    err = math.sqrt(abs(target)) if target > 0 else 0.0

    if wantNegativeContributions:
        ok, dbg, neg = FixNegativeContributions(h_shape)
        if not ok:
            print("[QCD][ML] Negative bins after estimate. Zeroing histogram.")
            h_shape.Scale(0.0)
            h_up.Scale(0.0)
            h_down.Scale(0.0)

    return h_shape, h_up, h_down, err, err


def QCD_Estimation_ML_Full(histograms, all_samples_list, channel, category, uncName, scale, wantNegativeContributions):
    """
    Full ML:
      - data_ML_shape(A) - sum(mc_ML_shape(A)) = final QCD
    """
    kA = _key(channel, _REGIONS["A"], category, uncName, scale)

    bkg = _auto_background_list(all_samples_list)
    print(f"[QCD][ML] backgrounds used: {bkg}")

    try:
        h_final = _ml_shape_for_A(histograms, bkg, kA)
    except RuntimeError as e:
        print(f"[QCD][ML_FULL] {e} -> returning zeros.")
        z = ROOT.TH1D()
        return z, z, z, 0.0, 0.0

    final_yield = _integral(h_final)
    if final_yield < 0:
        print(f"[QCD][ML_FULL] Negative yield after subtraction ({final_yield:.3f}). Zeroing.")
        h_final.Scale(0.0)

    h_up   = _safe_clone(h_final, "Up")
    h_down = _safe_clone(h_final, "Down")

    err = math.sqrt(abs(_integral(h_final))) if _integral(h_final) > 0 else 0.0

    if wantNegativeContributions:
        ok, dbg, neg = FixNegativeContributions(h_final)
        if not ok:
            print("[QCD][ML_FULL] Negative bins after estimate. Zeroing histogram.")
            h_final.Scale(0.0)
            h_up.Scale(0.0)
            h_down.Scale(0.0)

    return h_final, h_up, h_down, err, err


def AddQCDInHistDict(
    var,
    all_histograms,        
    channels,              
    categories,           
    uncName,            
    all_samples_list,     
    scales,                
    wantNegativeContributions=False,
    mode="abcd",            # "abcd", "ml", "ml_full"
):

    if "QCD" not in all_histograms:
        all_histograms["QCD"] = {}

    mode = mode.lower()
    if mode == "abcd":
        estimator = QCD_Estimation_ABCD
    elif mode == "ml":
        estimator = QCD_Estimation_ML
    elif mode == "ml_full":
        estimator = QCD_Estimation_ML_Full
    else:
        raise RuntimeError(f"[QCD] Unknown mode '{mode}'")

    last_err_norm, last_err_var = 0.0, 0.0

    for ch in channels:
        for cat in categories:
            for sc in (scales + ["Central"]):
                if uncName == "Central" and sc != "Central":
                    continue
                if uncName != "Central" and sc == "Central":
                    continue

                kA = _key(ch, _REGIONS["A"], cat, uncName, sc)
                try:
                    hcen, hup, hdown, e1, e2 = estimator(
                        all_histograms, all_samples_list, ch, cat, uncName, sc, wantNegativeContributions
                    )
                except Exception as ex:
                    print(f"[QCD] ERROR computing QCD for {(ch,cat,uncName,sc)}: {ex}")
                    # leave empty histogram instead of crashing
                    hcen = ROOT.TH1D()
                    hup  = ROOT.TH1D()
                    hdown= ROOT.TH1D()
                    e1 = e2 = 0.0

                all_histograms["QCD"][kA] = hcen
                last_err_norm, last_err_var = e1, e2

            if uncName == "QCDScale":
                kUp   = _key(ch, _REGIONS["A"], cat, "QCDScale", "Up")
                kDown = _key(ch, _REGIONS["A"], cat, "QCDScale", "Down")
                # For now they equal Central. placeholder
                all_histograms["QCD"][kUp]   = _safe_clone(all_histograms["QCD"].get(_key(ch, _REGIONS["A"], cat, "Central", "Central"), ROOT.TH1D()), "Up")
                all_histograms["QCD"][kDown] = _safe_clone(all_histograms["QCD"].get(_key(ch, _REGIONS["A"], cat, "Central", "Central"), ROOT.TH1D()), "Down")

            if uncName == "Central":
                print(f"[QCD][{mode}] {ch}/{cat} var={var} -> err(norm)={last_err_norm:.3f}, err(var)={last_err_var:.3f}")

    return last_err_norm, last_err_var