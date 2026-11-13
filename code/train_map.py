# -*- coding: utf-8 -*-
"""Map-only training for the StreamMapNet reproduction and temporal-fusion ablations."""
from __future__ import annotations

import argparse
import csv
import os
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from edgebev.config import CFG
from edgebev.metrics import map_ap, temporal_consistency
from edgebev.models.head_model import HeadModel
from edgebev.models.heads import decode_map, map_loss
from train_heads import BEVHeadDataset, collate

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
RUNS = os.path.join(ROOT, "runs")

VARIANTS = {
    "gru_q": dict(map_temporal="gru", query_prop=10),
    "gru_noq": dict(map_temporal="gru", query_prop=0),
    "none": dict(map_temporal="none", query_prop=0),
    "res2_q": dict(map_temporal="res2", query_prop=10),
    "res2_noq": dict(map_temporal="res2", query_prop=0),
}


def forward_seq(model, batch, cfg, umc=None):
    T = batch["bev"].shape[1]
    bevs = [batch["bev"][:, j] for j in range(T)]
    uncs = [batch["unc"][:, j] for j in range(T)]
    h_map = None
    q_mem = p_mem = None
    outs = []
    for ti in range(T):
        if cfg["map_temporal"] == "res2":
            feat = model.fuse_det(bevs[:ti + 1], uncs[:ti + 1], umc)
        else:
            h_map = model.fuse_map(bevs[ti], h_map, umc, uncs[ti])
            feat = h_map
        dx = 0.0 if ti == 0 else float((batch["ego"][:, ti, 0] - batch["ego"][:, ti - 1, 0]).mean())
        logits, pts, q_mem, p_mem = model.map_step(model.up(feat), q_mem, p_mem, dx, umc=umc)
        outs.append((logits, pts))
    return outs


def evaluate(model, dl, cfg):
    model.eval()
    preds, gts, tc = [], [], []
    with torch.no_grad():
        for batch in dl:
            outs = forward_seq(model, batch, cfg)
            T = batch["bev"].shape[1]
            seq = [[] for _ in range(batch["bev"].shape[0])]
            for ti, (logits, pts) in enumerate(outs):
                maps = decode_map(logits, pts, CFG)
                for bi in range(batch["bev"].shape[0]):
                    mm = batch["map_mask"][bi, ti].numpy() > 0
                    gts.append([(int(c), p) for c, p, ok in
                                zip(batch["map_cls"][bi, ti].numpy(), batch["map_line"][bi, ti].numpy(), mm) if ok])
                    preds.append(maps[bi])
                    seq[bi].append(maps[bi])
            for bi in range(batch["bev"].shape[0]):
                tc.append(temporal_consistency(seq[bi], batch["ego"][bi].numpy(), CFG))
    mm = map_ap(preds, gts)
    return {"map_mAP": mm["mAP"], "map_AP@0.5": mm["AP@0.5"], "map_AP@1": mm["AP@1"],
            "map_AP@1.5": mm["AP@1.5"], "temporal_iou": float(np.mean(tc))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="gru_q", choices=list(VARIANTS))
    ap.add_argument("--tag", default="crf")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--eval_every", type=int, default=5)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    torch.manual_seed(0)
    cfg = VARIANTS[args.variant]
    out = os.path.join(RUNS, f"map_{args.tag}_{args.variant}")
    os.makedirs(out, exist_ok=True)
    model = HeadModel(CFG, "res2" if cfg["map_temporal"] == "res2" else "none",
                      cfg["map_temporal"], cfg["query_prop"])
    tr = BEVHeadDataset(args.tag, "train"); va = BEVHeadDataset(args.tag, "val")
    dl_tr = DataLoader(tr, batch_size=args.batch, shuffle=True, collate_fn=collate)
    dl_va = DataLoader(va, batch_size=args.batch, shuffle=False, collate_fn=collate)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr,
                            weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    hist, best = [], -1
    for ep in range(1, args.epochs + 1):
        model.train(); t0 = time.time(); tot = 0.0
        for batch in dl_tr:
            opt.zero_grad()
            outs = forward_seq(model, batch, cfg)
            T = batch["bev"].shape[1]
            loss = 0.0
            for ti, (logits, pts) in enumerate(outs):
                lm, _ = map_loss(logits, pts, batch["map_cls"][:, ti], batch["map_line"][:, ti],
                                 batch["map_mask"][:, ti], CFG)
                loss = loss + lm
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            opt.step()
            tot += float(loss)
        sched.step()
        tot /= len(dl_tr)
        metrics = evaluate(model, dl_va, cfg) if (ep % args.eval_every == 0 or ep == args.epochs) else {}
        print(f"[map-{args.tag}-{args.variant}] ep {ep:02d} loss {tot:.4f} | {metrics} | "
              f"{time.time()-t0:.1f}s", flush=True)
        row = {"epoch": ep, "train_loss": tot, **metrics, "seconds": time.time() - t0}
        hist.append(row)
        fields = ["epoch", "train_loss", "map_mAP", "map_AP@0.5", "map_AP@1", "map_AP@1.5",
                  "temporal_iou", "seconds"]
        with open(os.path.join(out, "history.csv"), "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore"); w.writeheader()
            for r in hist:
                w.writerow({k: r.get(k, "") for k in fields})
        if metrics and metrics["map_mAP"] > best:
            best = metrics["map_mAP"]
            torch.save({"model": model.state_dict(), "variant": args.variant, "tag": args.tag,
                        "cfg": cfg, "metrics": metrics}, os.path.join(out, "best.pt"))
    print("best map mAP", best)


if __name__ == "__main__":
    main()
