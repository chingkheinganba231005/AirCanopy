"""Live prices of single graphics cards on RunPod and Vast.ai, for choosing where to run GPU jobs.

Both publish their prices through public interfaces that need no account; this records one
snapshot of each, so that the numbers in the journal come from a dated computation.

- RunPod sets one price per card model for each of its two clouds: Secure Cloud (data centres
  RunPod vets) and Community Cloud (other hosts, cheaper). Its API also says how many cards
  are in stock (stockStatus).
- Vast.ai is a marketplace: each host sets its own price, so there is no list price, only the
  offers open at one moment. This asks its offer search for verified, rentable, on-demand
  machines with one card of each model, and records how many there were, their hourly prices
  (dph_total: the card plus a small default disk), the lowest price for an interruptible
  rental (min_bid), and the memory bandwidth Vast.ai measured on each machine (gpu_mem_bw,
  GB/s). The A100 comes with 40 or 80 GB; only the 80 GB cards are kept.

Writes results/gpu_offers.json, replacing the previous snapshot.

    uv run python benchmarks/gpu_offers.py
"""

from __future__ import annotations

import datetime as dt
import json
import statistics
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SEARCH = "https://console.vast.ai/api/v0/bundles/"
RUNPOD = "https://api.runpod.io/graphql"
#: RunPod's ids for the same models.
RUNPOD_IDS = {"RTX 3090": "NVIDIA GeForce RTX 3090", "RTX 4090": "NVIDIA GeForce RTX 4090",
              "RTX 5090": "NVIDIA GeForce RTX 5090", "RTX A6000": "NVIDIA RTX A6000",
              "L40S": "NVIDIA L40S", "A100 PCIE": "NVIDIA A100 80GB PCIe",
              "A100 SXM4": "NVIDIA A100-SXM4-80GB", "H100 PCIE": "NVIDIA H100 PCIe",
              "H100 SXM": "NVIDIA H100 80GB HBM3"}  # fmt: skip
#: Vast.ai's model names, and the least memory (GB) an offer must have to count.
MODELS = {"RTX 3090": 20, "RTX 4090": 20, "RTX 5090": 28, "RTX A6000": 40, "L40S": 40,
          "A100 PCIE": 70, "A100 SXM4": 70, "H100 PCIE": 70, "H100 SXM": 70}  # fmt: skip


def search(model: str) -> list[dict]:
    query = {"gpu_name": {"eq": model}, "num_gpus": {"eq": 1}, "rentable": {"eq": True},
             "verified": {"eq": True}, "type": "on-demand", "order": [["dph_total", "asc"]],
             "limit": 500}  # fmt: skip
    url = SEARCH + "?" + urllib.parse.urlencode({"q": json.dumps(query)})
    with urllib.request.urlopen(url, timeout=60) as response:
        return json.loads(response.read()).get("offers", [])


def quantile(values: list[float], q: float) -> float:
    return statistics.quantiles(values, n=4, method="inclusive")[int(q * 4) - 1]


def summarise(model: str, offers: list[dict]) -> dict:
    keep = [o for o in offers if o.get("gpu_ram", 0) / 1024 >= MODELS[model]
            and not o.get("rented") and o.get("verification") == "verified"]  # fmt: skip
    out = {"model": model, "min_gpu_ram_gb": MODELS[model], "offers": len(keep)}
    if not keep:
        return out
    price = sorted(o["dph_total"] for o in keep)
    out.update(
        usd_per_hour={"min": round(price[0], 3), "median": round(statistics.median(price), 3),
                      "max": round(price[-1], 3)},
        interruptible_min_bid_usd_per_hour=round(min(o.get("min_bid", price[0]) for o in keep), 3),
        measured_bandwidth_gb_s_median=round(statistics.median(o["gpu_mem_bw"] for o in keep)),
        reliability_median=round(statistics.median(o.get("reliability2") or 0 for o in keep), 3),
        countries=sorted({(o.get("geolocation") or "?").split(",")[-1].strip() for o in keep}),
    )  # fmt: skip
    if len(price) >= 4:
        out["usd_per_hour"]["quartiles"] = [round(quantile(price, q), 3) for q in (0.25, 0.75)]
    return out


def runpod_prices() -> list[dict]:
    fields = ("id displayName memoryInGb secureCloud communityCloud securePrice communityPrice "
              "lowestPrice(input: {gpuCount: 1}) { stockStatus }")  # fmt: skip
    body = json.dumps({"query": f"query {{ gpuTypes {{ {fields} }} }}"}).encode()
    request = urllib.request.Request(RUNPOD, body, {"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=60) as response:
        types = {t["id"]: t for t in json.loads(response.read())["data"]["gpuTypes"]}
    rows = []
    for model, gpu_id in RUNPOD_IDS.items():
        t = types.get(gpu_id)
        if t is None:
            rows.append({"model": model, "id": gpu_id, "listed": False})
            continue
        rows.append({
            "model": model, "id": gpu_id, "listed": True, "memory_gb": t["memoryInGb"],
            "secure_usd_per_hour": t["securePrice"] if t["secureCloud"] else None,
            "community_usd_per_hour": t["communityPrice"] if t["communityCloud"] else None,
            "stock": (t.get("lowestPrice") or {}).get("stockStatus"),
        })  # fmt: skip
    return rows


def main() -> None:
    at = dt.datetime.now(dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    vast = [summarise(m, search(m)) for m in MODELS]
    runpod = runpod_prices()
    for r in runpod + vast:
        print(r)
    out = {
        "what": "hourly prices of one graphics card: RunPod's list prices, and the verified, "
        "rentable, on-demand single-GPU offers on Vast.ai, at one moment",
        "run_at": at,
        "runpod": {"source": RUNPOD, "models": runpod},
        "vast": {"source": SEARCH, "models": vast},
    }
    path = ROOT / "results" / "gpu_offers.json"
    path.write_text(json.dumps(out, indent=2) + "\n")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
