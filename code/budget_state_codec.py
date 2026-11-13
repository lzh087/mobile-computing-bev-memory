"""An actual FP16 binary codec for ONE streaming BEV/query state on CPU.

Payload length includes a 24-byte header, cell indices, BEV values, pooled
background, query embeddings AND polyline reference points. It excludes model
parameters, activations and decoder workspace. It is not a process-RAM metric.
"""
from __future__ import annotations

import struct
import numpy as np
import torch
import torch.nn.functional as F

HEADER = struct.Struct('<4s10H')


def encode_state(h, unc, q, pts, budget=1.0, query_count=10, mode='uncertainty'):
    if h.shape[0] != 1:
        raise ValueError('codec expects batch size 1')
    _, c, height, width = h.shape
    if height * width > 65535:
        raise ValueError('uint16 codec supports at most 65535 cells')
    nq = min(query_count, q.shape[1])
    q, pts = q[:, :nq], pts[:, :nq]
    pool = 0 if budget >= 1.0 else 4
    if pool == 0:
        k = bh = bw = 0
        parts = [h.detach().numpy().astype('<f2').tobytes()]
    else:
        k = max(1, min(height * width, round(budget * height * width)))
        mag = h.detach().abs().mean(1).flatten()
        if mode == 'uncertainty':
            score = mag * unc.detach().flatten()
        elif mode == 'magnitude':
            score = mag
        elif mode == 'random':
            score = torch.rand_like(mag)
        else:
            raise ValueError(mode)
        idx = score.topk(k).indices
        background = F.avg_pool2d(h, pool, pool, ceil_mode=True)
        bh, bw = background.shape[-2:]
        values = h.flatten(2)[0, :, idx]
        parts = [background.detach().numpy().astype('<f2').tobytes(),
                 idx.numpy().astype('<u2').tobytes(),
                 values.detach().numpy().astype('<f2').tobytes()]
    dim = q.shape[-1]
    npoints = pts.shape[-2]
    head = HEADER.pack(b'BBV1', c, height, width, pool, k, bh, bw, nq, dim, npoints)
    return b''.join([head, *parts, q.detach().numpy().astype('<f2').tobytes(),
                     pts.detach().numpy().astype('<f2').tobytes()])


def decode_state(payload):
    magic, c, height, width, pool, k, bh, bw, nq, dim, npoints = HEADER.unpack_from(payload)
    if magic != b'BBV1':
        raise ValueError('invalid state header')
    offset = HEADER.size
    def read(count, dtype):
        nonlocal offset
        result = np.frombuffer(payload, dtype=dtype, count=count, offset=offset).copy()
        offset += count * np.dtype(dtype).itemsize
        return torch.from_numpy(result.astype(np.float32))
    if pool == 0:
        h = read(c * height * width, '<f2').reshape(1,c,height,width)
    else:
        bg = read(c * bh * bw, '<f2').reshape(1,c,bh,bw)
        idx = read(k, '<u2').long()
        vals = read(c * k, '<f2').reshape(1,c,k)
        h = F.interpolate(bg, size=(height,width), mode='bilinear', align_corners=False)
        h.flatten(2).scatter_(2,idx[None,None,:].expand(1,c,k),vals)
    q = read(nq * dim, '<f2').reshape(1,nq,dim)
    pts = read(nq * npoints * 2, '<f2').reshape(1,nq,npoints,2)
    if offset != len(payload):
        raise ValueError('payload length mismatch')
    return h, q, pts


def check_codec():
    torch.manual_seed(7)
    h = torch.randn(1,64,50,50)
    u = torch.rand(1,1,50,50)
    q = torch.randn(1,10,128)
    p = torch.randn(1,10,12,2)
    full = encode_state(h,u,q,p,1.0,10)
    hh,qq,pp = decode_state(full)
    assert len(full) == 24 + 320000 + 10 * (128+24) * 2
    assert torch.equal(hh,h.half().float())
    assert torch.equal(qq,q.half().float())
    assert torch.equal(pp,p.half().float())
    sparse = encode_state(h,u,q,p,0.1,3)
    assert len(sparse) == 24 + 64*13*13*2 + 250*(64+1)*2 + 3*(128+24)*2
    h2,q2,p2=decode_state(sparse)
    assert h2.shape == h.shape and q2.shape[1] == p2.shape[1] == 3
    assert torch.isfinite(h2).all()
    return {'checks':'passed','header_bytes':HEADER.size,'dense_bytes':len(full),
            'sparse_bytes':len(sparse),'background_shape':[13,13],
            'dense_fp16_max_error':float((h-hh).abs().max())}


if __name__ == '__main__':
    import json
    print(json.dumps(check_codec(),indent=2))
