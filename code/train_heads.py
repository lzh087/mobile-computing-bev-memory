# -*- coding: utf-8 -*-
"""Stage-2: train temporal detection/map heads on cached BEV features."""
from __future__ import annotations

import argparse
import csv
import os
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from edgebev.config import CFG
from edgebev.metrics import detection_map, map_ap, temporal_consistency
from edgebev.models.head_model import HeadModel
from edgebev.models.heads import (decode_det, decode_map, det_loss, gaussian_heatmap,
                                   map_loss, occ_loss)

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
DATA = os.path.join(ROOT, "data", "synthetic")
BEV = os.path.join(ROOT, "data", "bev")
RUNS = os.path.join(ROOT, "runs")

VARIANTS = {
    "full": dict(temporal="res2", map_temporal="gru", query_prop=10),
    "no_temporal": dict(temporal="none", map_temporal="none", query_prop=0),
    "det_no_temporal": dict(temporal="none", map_temporal="gru", query_prop=10),
    "map_no_temporal": dict(temporal="res2", map_temporal="none", query_prop=10),
    "no_qprop": dict(temporal="res2", map_temporal="gru", query_prop=0),
    "gru_det": dict(temporal="gru", map_temporal="gru", query_prop=10),
}


class BEVHeadDataset(Dataset):
    def __init__(self, tag: str, split: str):
        self.tag, self.split = tag, split
        self.dir = os.path.join(DATA, split)
        self.files = sorted(os.listdir(self.dir))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        name = self.files[i]
        s = np.load(os.path.join(self.dir, name))
        b = np.load(os.path.join(BEV, self.tag, self.split, name))
        out = {
            "bev": torch.from_numpy(b["bev"].astype(np.float32)),
            "unc": torch.from_numpy(b["unc"].astype(np.float32)),
            "ego": torch.from_numpy(s["ego"]),
            "det_cls": torch.from_numpy(s["det_cls"]), "det_center": torch.from_numpy(s["det_center"]),
            "det_size": torch.from_numpy(s["det_size"]), "det_yaw": torch.from_numpy(s["det_yaw"]),
            "det_mask": torch.from_numpy(s["det_mask"]),
            "map_cls": torch.from_numpy(s["map_cls"]), "map_line": torch.from_numpy(s["map_line"]),
            "map_mask": torch.from_numpy(s["map_mask"]),
            "occ": torch.from_numpy(s["occ"]).float(),
        }
        return out


def collate(batch):
    return {k: torch.stack([b[k] for b in batch], 0) for k in batch[0]}


def evaluate(model, dl):
    model.eval()
    preds_det, gts_det, preds_map, gts_map, tc = [], [], [], [], []
    ious = []
    with torch.no_grad():
        for batch in dl:
            T = batch["bev"].shape[1]
            h_map = None
            q_mem = p_mem = None
            seq_preds = [[] for _ in range(batch["bev"].shape[0])]
            for ti in range(T):
                bevs = [batch["bev"][:, j] for j in range(ti + 1)]
                uncs = [batch["unc"][:, j] for j in range(ti + 1)]
                fused = model.fuse_det(bevs, uncs)
                det = model.det(model.up(fused))
                h_map = model.fuse_map(batch["bev"][:, ti], h_map)
                dx = 0.0 if ti == 0 else float((batch["ego"][:, ti, 0] - batch["ego"][:, ti - 1, 0]).mean())
                logits_m, pts, q_mem, p_mem = model.map_step(model.up(h_map), q_mem, p_mem, dx)
                dets = decode_det(det, CFG)
                maps = decode_map(logits_m, pts, CFG)
                op = (torch.sigmoid(model.occ(model.up(fused))) > 0.5).squeeze(1).numpy()
                for bi in range(batch["bev"].shape[0]):
                    gt_occ = batch["occ"][bi, ti].numpy() > 0.5
                    inter = np.logical_and(op[bi], gt_occ).sum()
                    uni = np.logical_or(op[bi], gt_occ).sum()
                    ious.append(inter / max(uni, 1))
                    m = batch["det_mask"][bi, ti].numpy() > 0
                    cs = batch["det_center"][bi, ti][m].numpy()
                    ss = batch["det_size"][bi, ti][m].numpy()
                    ys = batch["det_yaw"][bi, ti][m].numpy()
                    cc = batch["det_cls"][bi, ti][m].numpy()
                    gts_det.append([[float(cs[j, 0]), float(cs[j, 1]), float(ss[j, 0]), float(ss[j, 1]),
                                     float(ys[j]), int(cc[j])] for j in range(len(cs))])
                    preds_det.append(dets[bi])
                    mm = batch["map_mask"][bi, ti].numpy() > 0
                    gts_map.append([(int(c), p) for c, p, ok in
                                    zip(batch["map_cls"][bi, ti].numpy(), batch["map_line"][bi, ti].numpy(), mm) if ok])
                    preds_map.append(maps[bi])
                    seq_preds[bi].append(maps[bi])
            for bi in range(batch["bev"].shape[0]):
                tc.append(temporal_consistency(seq_preds[bi], batch["ego"][bi].numpy(), CFG))
    dm = detection_map(preds_det, gts_det)
    mm = map_ap(preds_map, gts_map)
    return {"det_mAP": dm["mAP"], "det_mATE": dm["mATE"], "det_AP@1": dm["AP@1"],
            "map_mAP": mm["mAP"], "map_AP@1": mm["AP@1"], "temporal_iou": float(np.mean(tc)),
            "occ_iou": float(np.mean(ious)) if ious else 0.0}


def train(args):
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    out = os.path.join(RUNS, f"heads_{args.tag}_{args.variant}")
    os.makedirs(out, exist_ok=True)
    cfg = VARIANTS[args.variant]
    model = HeadModel(CFG, cfg["temporal"], cfg["map_temporal"], cfg["query_prop"])
    tr = BEVHeadDataset(args.tag, "train")
    va = BEVHeadDataset(args.tag, "val")
    dl_tr = DataLoader(tr, batch_size=args.batch, shuffle=True, collate_fn=collate)
    dl_va = DataLoader(va, batch_size=args.batch, shuffle=False, collate_fn=collate)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    hist, best = [], -1
    for ep in range(1, args.epochs + 1):
        model.train(); t0 = time.time()
        sums = {"loss": 0.0, "det": 0.0, "map": 0.0, "occ": 0.0}
        for batch in dl_tr:
            opt.zero_grad()
            T = batch["bev"].shape[1]
            h_map = None; q_mem = p_mem = None; loss = 0.0
            for ti in range(T):
                bevs = [batch["bev"][:, j] for j in range(ti + 1)]
                uncs = [batch["unc"][:, j] for j in range(ti + 1)]
                fused = model.fuse_det(bevs, uncs)
                det = model.det(model.up(fused))
                occ_logits = model.occ(model.up(fused))
                lo, _ = occ_loss(occ_logits, batch["occ"][:, ti])
                hm = []
                for bi in range(batch["bev"].shape[0]):
                    m = batch["det_mask"][bi, ti].numpy() > 0
                    hm.append(gaussian_heatmap(batch["det_center"][bi, ti][m].numpy(),
                                               batch["det_cls"][bi, ti][m].numpy(), CFG))
                ld, _ = det_loss(det, np.stack(hm), batch["det_center"][:, ti],
                                 batch["det_size"][:, ti], batch["det_yaw"][:, ti], CFG)
                h_map = model.fuse_map(batch["bev"][:, ti], h_map)
                dx = 0.0 if ti == 0 else float((batch["ego"][:, ti, 0] - batch["ego"][:, ti - 1, 0]).mean())
                logits_m, pts, q_mem, p_mem = model.map_step(model.up(h_map), q_mem, p_mem, dx)
                lm, _ = map_loss(logits_m, pts, batch["map_cls"][:, ti], batch["map_line"][:, ti],
                                 batch["map_mask"][:, ti], CFG)
                loss = loss + ld + lm + 0.5 * lo
                sums["det"] += float(ld); sums["map"] += float(lm); sums["occ"] += float(lo)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            opt.step()
            sums["loss"] += float(loss)
        sched.step()
        for k in sums:
            sums[k] /= len(dl_tr)
        metrics = evaluate(model, dl_va) if (ep % args.eval_every == 0 or ep == args.epochs) else {}
        print(f"[heads-{args.tag}-{args.variant}] ep {ep:02d} loss {sums['loss']:.3f} "
              f"det {sums['det']:.2f} map {sums['map']:.2f} occ {sums['occ']:.2f} | "
              f"{metrics} | {time.time()-t0:.1f}s", flush=True)
        row = {"epoch": ep, "train_loss": sums["loss"], "train_det": sums["det"], "train_map": sums["map"],
               "train_occ": sums["occ"], **metrics, "seconds": time.time() - t0}
        hist.append(row)
        fields = ["epoch", "train_loss", "train_det", "train_map", "train_occ", "det_mAP", "det_mATE",
                  "det_AP@1", "map_mAP", "map_AP@1", "occ_iou", "temporal_iou", "seconds"]
        with open(os.path.join(out, "history.csv"), "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            w.writeheader()
            for r in hist:
                w.writerow({k: r.get(k, "") for k in fields})
        if metrics:
            score = metrics["det_mAP"] + metrics["map_mAP"] + metrics["occ_iou"]
            if score > best:
                best = score
                torch.save({"model": model.state_dict(), "variant": args.variant, "tag": args.tag,
                            "metrics": metrics, "cfg": cfg}, os.path.join(out, "best.pt"))
    print("best", best)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="full", choices=list(VARIANTS))
    ap.add_argument("--tag", default="crf")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eval_every", type=int, default=3)
    train(ap.parse_args())
