# -*- coding: utf-8 -*-
"""RUNG 2A: GEFS 31-member ensemble MEAN(geavg) + SPREAD(gespr) fetcher.

Source: NOAA public S3 (noaa-gefs-pds, public domain), byte-range GRIB2 + eccodes
in-memory decode. For target day D (KST 01..24h) we use cycle (D-1) 00Z — the SAME
init time as the competition's provided NWP (09:00 KST = 00Z, deemed available 13:00+
KST; NOAA dissemination of 0.5deg pgrb2ap5 completes ~04:30 UTC = 13:30 KST, before the
~14:00 KST reference time in the rules' example -> leak-safe, documented for round 2).

Leads f015..f039 (3-hourly) cover valid KST 00..24 of D (hourly interp later).
Variables: UGRD/VGRD at 925mb/850mb/10m  x  {ens mean, ens spread}.
Farm value = IDW of the 4 surrounding 0.5deg points at (37.2833, 128.96).
Output: gefs_raw.csv (long: cycle, lead, kind, var, value). Resume-safe.
"""
import warnings; warnings.filterwarnings("ignore")
import sys, os, io, time, threading
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
import requests
import numpy as np, pandas as pd
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor
import eccodes as ec

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "gefs_raw.csv")
LAT, LON = 37.2833, 128.96
LEADS = list(range(15, 40, 3))
KINDS = ["geavg", "gespr"]
WANT = [("UGRD", "925 mb"), ("VGRD", "925 mb"), ("UGRD", "850 mb"),
        ("VGRD", "850 mb"), ("UGRD", "10 m above ground"), ("VGRD", "10 m above ground")]
VKEY = {("UGRD", "925 mb"): "u925", ("VGRD", "925 mb"): "v925", ("UGRD", "850 mb"): "u850",
        ("VGRD", "850 mb"): "v850", ("UGRD", "10 m above ground"): "u10",
        ("VGRD", "10 m above ground"): "v10"}
_tl = threading.local()
DEC = threading.Lock()          # eccodes is NOT thread-safe: serialize decode (network stays parallel)


def sess():
    if not hasattr(_tl, "s"):
        _tl.s = requests.Session()
    return _tl.s


FAILLOG = [0]


def _why(msg):
    if FAILLOG[0] < 30:
        FAILLOG[0] += 1
        print("[why] %s" % msg, flush=True)


def fetch_file(cyc, lead, kind, tries=4):
    """one grib file -> {vkey: farm_value} or None. Retries with backoff."""
    for a in range(tries):
        res = _fetch_once(cyc, lead, kind)
        if res is not None:
            return res
        time.sleep(0.5 * (a + 1))
    return None


def _fetch_once(cyc, lead, kind):
    url = ("https://noaa-gefs-pds.s3.amazonaws.com/gefs.%s/00/atmos/pgrb2ap5/%s.t00z.pgrb2a.0p50.f%03d"
           % (cyc, kind, lead))
    try:
        r = sess().get(url + ".idx", timeout=45, headers={"Connection": "close"})
        if r.status_code != 200:
            _why("idx HTTP %d %s %s f%d" % (r.status_code, cyc, kind, lead))
            return None
        ent = []
        for l in r.text.split("\n"):
            f = l.split(":")
            if len(f) >= 5:
                ent.append((int(f[1]), f[3], f[4]))
        sel = []
        for i, (off, var, lev) in enumerate(ent):
            if (var, lev) in WANT:
                end = ent[i + 1][0] - 1 if i + 1 < len(ent) else None
                sel.append((VKEY[(var, lev)], off, end))
        if len(sel) != 6:
            _why("sel %d %s %s f%d" % (len(sel), cyc, kind, lead))
            return None
        # group contiguous ranges
        sel.sort(key=lambda x: x[1])
        groups = []
        for vk, off, end in sel:
            if groups and off == groups[-1][2] + 1:
                groups[-1][1].append(vk); groups[-1][2] = end
            else:
                groups.append([off, [vk], end])
        out = {}
        for off, vks, end in groups:
            hdr = {"Range": "bytes=%d-%s" % (off, "" if end is None else str(end))}
            hdr["Connection"] = "close"
            rb = sess().get(url, headers=hdr, timeout=90)
            if rb.status_code not in (200, 206):
                _why("range HTTP %d %s %s f%d" % (rb.status_code, cyc, kind, lead))
                return None
            blob = rb.content
            pos = 0
            for vk in vks:
                if blob[pos:pos + 4] != b"GRIB":
                    _why("notGRIB %s %s f%d len%d" % (cyc, kind, lead, len(blob)))
                    return None
                mlen = int.from_bytes(blob[pos + 8:pos + 16], "big")
                with DEC:
                    h = ec.codes_new_from_message(blob[pos:pos + mlen])
                    near = ec.codes_grib_find_nearest(h, LAT, LON, npoints=4)
                    w = np.array([1.0 / max(p.distance, 1e-6) for p in near])
                    out[vk] = float(np.sum(w * [p.value for p in near]) / w.sum())
                    ec.codes_release(h)
                pos += mlen
        return out
    except Exception as e:
        _why("EXC %s: %s %s %s f%d" % (type(e).__name__, str(e)[:60], cyc, kind, lead))
        try:                                   # S3 closed keep-alive: rebuild this thread's session
            if hasattr(_tl, "s"):
                _tl.s.close(); del _tl.s
        except Exception:
            pass
        return None


def main():
    done = set()
    if os.path.exists(OUT):
        d0 = pd.read_csv(OUT, encoding="utf-8-sig", dtype={"cycle": str})
        vcols = list(VKEY.values())
        ok = d0[~d0[vcols].isna().all(axis=1)]          # drop all-NaN (failed) rows -> retry them
        ok.to_csv(OUT, index=False, encoding="utf-8-sig")
        done = set(zip(ok["cycle"], ok["lead"], ok["kind"]))
        print("resume: %d ok, %d failed rows dropped for retry" % (len(done), len(d0) - len(ok)), flush=True)
    # cycles: (target day D-1) for D in 2022-01-01 .. 2025-12-31 (+D=2026-01-01 boundary hour)
    cycles = [d.strftime("%Y%m%d") for d in pd.date_range("2021-12-31", "2025-12-31", freq="D")]
    tasks = [(c, l, k) for c in cycles for l in LEADS for k in KINDS if (c, l, k) not in done]
    print("tasks todo: %d (of %d)" % (len(tasks), len(cycles) * len(LEADS) * len(KINDS)), flush=True)
    lock = threading.Lock(); buf = []; nd = [0]; nf = [0]; t0 = time.time()

    def work(t):
        c, l, k = t
        res = fetch_file(c, l, k)
        time.sleep(0.01)
        with lock:
            if res is None:
                nf[0] += 1
                buf.append(dict(cycle=c, lead=l, kind=k, **{v: np.nan for v in VKEY.values()}))
            else:
                buf.append(dict(cycle=c, lead=l, kind=k, **res))
            nd[0] += 1
            if nd[0] % 400 == 0:
                df = pd.DataFrame(buf); buf.clear()
                head = not os.path.exists(OUT)
                df.to_csv(OUT, index=False, encoding="utf-8-sig", mode="a" if not head else "w", header=head)
                el = time.time() - t0
                print("progress %d/%d (fail %d) [%.0fs, %.1f/s]" %
                      (nd[0], len(tasks), nf[0], el, nd[0] / max(el, 1)), flush=True)

    with ThreadPoolExecutor(max_workers=16) as ex:
        list(ex.map(work, tasks))
    with lock:
        if buf:
            df = pd.DataFrame(buf)
            head = not os.path.exists(OUT)
            df.to_csv(OUT, index=False, encoding="utf-8-sig", mode="a" if not head else "w", header=head)
    print("[GEFS] DONE %d tasks, %d failed [%.0fs]" % (nd[0], nf[0], time.time() - t0), flush=True)


if __name__ == "__main__":
    main()
