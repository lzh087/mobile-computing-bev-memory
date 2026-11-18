"""Train current-frame students on CPU and compare future-prior weighting rules."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
from statistics import mean, stdev
import time

import numpy as np
import scipy
import torch
from torch import nn
from torch.nn import functional as F

from bev_data import CLASSES, FAR_START, LENGTH, OFFSETS, WIDTH, align_future, aligned_inputs, make_split, mask_far_observation


METHODS = ('supervised', 'uniform_kd', 'far_kd', 'visible_far_kd')
TEMPERATURE = 2.0
FOCUS_WEIGHT = 0.4
THRESHOLDS = np.arange(0.30, 0.801, 0.05).round(2).tolist()
OCCLUSION = (slice(28, 44), slice(5, 19))


class MapCNN(nn.Module):
    def __init__(self, input_channels: int, hidden_channels: int):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(input_channels, hidden_channels, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_channels, hidden_channels, 3, dilation=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(hidden_channels, hidden_channels, 3, dilation=3, padding=3),
            nn.ReLU(),
            nn.Conv2d(hidden_channels, len(CLASSES), 1),
        )

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        return self.features(observation)


def weights_for(method: str, current: torch.Tensor, future_visible: torch.Tensor) -> torch.Tensor:
    """Return a normalized spatial weighting map, shared across classes."""
    uniform = torch.full_like(current[:, 3:4], 1.0 / (LENGTH * WIDTH))
    if method == 'uniform_kd':
        return uniform
    far = torch.zeros_like(uniform)
    far[..., FAR_START:, :] = 1
    focus = far
    if method == 'visible_far_kd':
        focus = far * (1 - current[:, 3:4]) * future_visible
    mass = focus.sum(dim=(-2, -1), keepdim=True)
    focused = (1 - FOCUS_WEIGHT) * uniform + FOCUS_WEIGHT * focus / mass.clamp_min(1)
    return torch.where(mass > 0, focused, uniform)


def fit(model, inputs, labels, pos_weight, args, seed, soft=None, weights=None):
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sampler = torch.Generator().manual_seed(seed)
    history = []
    for epoch in range(args.epochs):
        total = 0.0
        kd = 0.0
        for indices in torch.randperm(len(inputs), generator=sampler).split(args.batch_size):
            logits = model(inputs[indices])
            truth = F.binary_cross_entropy_with_logits(logits, labels[indices], pos_weight=pos_weight)
            loss = truth
            if soft is not None:
                per_pixel = F.binary_cross_entropy_with_logits(
                    logits / TEMPERATURE, soft[indices], reduction='none'
                )
                distillation = (per_pixel * weights[indices]).sum(dim=(-2, -1)).mean()
                loss = truth + args.kd_weight * TEMPERATURE ** 2 * distillation
                kd += float(distillation.detach()) * len(indices)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total += float(loss.detach()) * len(indices)
        history.append({'epoch': epoch + 1, 'loss': total / len(inputs), 'kd_bce': kd / len(inputs)})
    return history


@torch.inference_mode()
def probabilities(model: MapCNN, inputs: torch.Tensor, batch_size: int) -> torch.Tensor:
    model.eval()
    return torch.cat([model(batch).sigmoid() for batch in inputs.split(batch_size)])


def score(predictions: torch.Tensor, labels: torch.Tensor, threshold: float) -> dict:
    pred = predictions >= threshold
    gt = labels.bool()
    regions = {
        'all': (slice(None), slice(None)),
        'near': (slice(0, FAR_START), slice(None)),
        'far': (slice(FAR_START, LENGTH), slice(None)),
        'corridor': OCCLUSION,
    }
    output = {}
    for name, (xs, ys) in regions.items():
        pp, gg = pred[..., xs, ys], gt[..., xs, ys]
        tp = (pp & gg).sum(dim=(0, 2, 3)).double()
        count = pp.sum(dim=(0, 2, 3)) + gg.sum(dim=(0, 2, 3))
        f1 = (2 * tp / count.clamp_min(1)).tolist()
        output[f'{name}_f1'] = mean(f1)
        output[f'{name}_class_f1'] = dict(zip(CLASSES, f1, strict=True))
    return output


def validate_threshold(predictions: torch.Tensor, labels: torch.Tensor) -> float:
    return max(THRESHOLDS, key=lambda threshold: (score(predictions, labels, threshold)['all_f1'], -abs(threshold - 0.5)))


@torch.inference_mode()
def measure_latency(model: MapCNN, current: torch.Tensor) -> dict:
    model.eval()
    for _ in range(24):
        model(current[:1])
    times = []
    for sample in current[:128].split(1):
        start = time.perf_counter_ns()
        model(sample)
        times.append((time.perf_counter_ns() - start) / 1e6)
    return {'p50_ms': float(np.median(times)), 'p95_ms': float(np.percentile(times, 95)), 'samples': len(times)}


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def sanity_checks(splits) -> dict:
    impulse = torch.zeros(1, 4, LENGTH, WIDTH)
    impulse[0, 0, 1, 2] = 1
    impulse[0, 3, -1, -1] = 1
    aligned = align_future(impulse, 4)
    assert aligned[0, 0, 5, 2] == 1 and aligned.sum() == 1
    assert not aligned[..., :4, :].any()
    used_ids = [scene_id for split in splits for scene_id in split.scene_ids]
    assert len(used_ids) == len(set(used_ids))
    current = splits[-1].views[:3, 0]
    assert not mask_far_observation(current)[..., 28:44, 5:19].any()
    zero = torch.zeros_like(current[:, 3:4])
    result = weights_for('visible_far_kd', current, zero)
    assert torch.allclose(result.sum(dim=(-2, -1)), torch.ones(3, 1))
    assert torch.equal(result, weights_for('uniform_kd', current, zero))
    return {'alignment_impulse': 'passed', 'no_wrap': 'passed', 'disjoint_scenes': 'passed',
            'empty_focus_fallback': 'passed', 'far_occlusion_mask': 'passed'}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=Path('runs_week11'))
    parser.add_argument('--train', type=int, default=384)
    parser.add_argument('--validation', type=int, default=96)
    parser.add_argument('--test', type=int, default=192)
    parser.add_argument('--epochs', type=int, default=16)
    parser.add_argument('--seeds', nargs='+', type=int, default=[11, 23, 47])
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--batch-size', type=int, default=24)
    parser.add_argument('--lr', type=float, default=0.002)
    parser.add_argument('--kd-weight', type=float, default=0.8)
    args = parser.parse_args()
    if min(args.epochs, args.threads, args.batch_size) < 1 or args.lr <= 0:
        parser.error('epochs, threads, batch size and learning rate must be positive')
    if not np.isfinite(args.kd_weight) or args.kd_weight < 0:
        parser.error('KD weight must be finite and non-negative')
    if len(set(args.seeds)) != len(args.seeds):
        parser.error('training seeds must be unique')
    args.out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(args.threads)
    train = make_split('train', args.train)
    val = make_split('validation', args.validation)
    test = make_split('test', args.test)
    checks = sanity_checks((train, val, test))
    current_train = train.views[:, 0].contiguous()
    current_val = val.views[:, 0].contiguous()
    current_test = test.views[:, 0].contiguous()
    occluded_test = mask_far_observation(current_test)
    train_teacher, future_visible = aligned_inputs(train.views)
    val_teacher, _ = aligned_inputs(val.views)
    test_teacher, _ = aligned_inputs(test.views)
    wrong_teacher, _ = aligned_inputs(test.views, extra_shift=2)
    future_changed = test.views[:2].clone()
    future_changed[:, 1:] = 0
    pos_rate = train.labels.mean(dim=(0, 2, 3)).clamp_min(1e-5)
    pos_weight = ((1 - pos_rate) / pos_rate).sqrt().clamp(1, 7).reshape(-1, 1, 1)
    focus_coverage = ((1 - current_train[:, 3:4, FAR_START:]) * future_visible[:, :, FAR_START:]).mean()
    experiment = {'utc_started': datetime.now(timezone.utc).isoformat(), 'platform': platform.platform(),
                  'processor': platform.processor(), 'python': platform.python_version(),
                  'torch': torch.__version__, 'numpy': np.__version__, 'scipy': scipy.__version__,
                  'device': 'cpu', 'threads': args.threads, 'resolution_m': 1.0,
                  'grid_shape': [LENGTH, WIDTH], 'near_m': [0, FAR_START], 'far_m': [FAR_START, LENGTH],
                  'offsets_m': list(OFFSETS), 'epochs': args.epochs, 'seeds': args.seeds,
                  'batch_size': args.batch_size, 'learning_rate': args.lr,
                  'temperature': TEMPERATURE, 'kd_weight': args.kd_weight, 'focus_weight': FOCUS_WEIGHT,
                  'current_missing_future_visible_far_ratio': float(focus_coverage),
                  'classes': list(CLASSES), 'positive_weights': pos_weight.flatten().tolist(),
                  'test_occlusion': {'x_m': [28, 44], 'y_m': [-7, 7]},
                  'split_ids': {split.name: split.scene_ids for split in (train, val, test)}, 'checks': checks}
    print('config', json.dumps({key: val for key, val in experiment.items() if key != 'split_ids'}), flush=True)
    start = time.perf_counter()
    results, histories = [], []
    for seed in args.seeds:
        torch.manual_seed(seed)
        teacher = MapCNN(12, 24)
        teacher_history = fit(teacher, train_teacher, train.labels, pos_weight, args, seed)
        teacher_threshold = validate_threshold(probabilities(teacher, val_teacher, args.batch_size), val.labels)
        with torch.inference_mode():
            teacher.eval()
            soft = torch.cat([(teacher(batch) / TEMPERATURE).sigmoid() for batch in train_teacher.split(args.batch_size)])
        # Store regular tensors rather than inference tensors for student autograd.
        soft = soft.clone()
        histories.extend([{'seed': seed, 'method': 'teacher', **row} for row in teacher_history])
        for condition, teacher_inputs in [('aligned', test_teacher), ('pose_plus_2m', wrong_teacher)]:
            metrics = score(probabilities(teacher, teacher_inputs, args.batch_size), test.labels, teacher_threshold)
            teacher_row = {'seed': seed, 'method': 'teacher', 'condition': condition, 'threshold': teacher_threshold, **metrics}
            results.append(teacher_row)
            print('result', json.dumps(teacher_row), flush=True)
        for method in METHODS:
            # Matched initialization and minibatch order for all student variants.
            torch.manual_seed(seed + 1_000)
            student = MapCNN(4, 16)
            use_kd = method != 'supervised'
            weights = weights_for(method, current_train, future_visible) if use_kd else None
            losses = fit(student, current_train, train.labels, pos_weight, args, seed,
                         soft if use_kd else None, weights)
            histories.extend([{'seed': seed, 'method': method, **row} for row in losses])
            threshold = validate_threshold(probabilities(student, current_val, args.batch_size), val.labels)
            assert torch.equal(probabilities(student, current_test[:2], args.batch_size),
                               probabilities(student, future_changed[:, 0], args.batch_size))
            latency = measure_latency(student, current_test) if seed == args.seeds[-1] else None
            for condition, current in [('standard', current_test), ('far_occluded', occluded_test)]:
                probs = probabilities(student, current, args.batch_size)
                metrics = score(probs, test.labels, threshold)
                row = {'seed': seed, 'method': method, 'condition': condition, 'threshold': threshold, **metrics}
                results.append(row)
                print('result', json.dumps(row), flush=True)
                if condition == 'standard' and seed == args.seeds[-1]:
                    torch.save(student.state_dict(), args.out / f'{method}_student.pt')
                    (args.out / f'{method}_deployment.json').write_text(json.dumps({
                        'parameters': sum(p.numel() for p in student.parameters()),
                        'parameter_bytes_fp32': sum(p.numel() * p.element_size() for p in student.parameters()),
                        'persistent_temporal_state_bytes': 0,
                        'latency_cpu_forward_only': latency,
                        'seed': seed, 'input_shape': [1, 4, LENGTH, WIDTH],
                    }, indent=2), encoding='utf-8')
                    np.savez_compressed(args.out / f'{method}_examples.npz',
                                        probs=probs[:6].numpy(), current=current_test[:6].numpy(),
                                        gt=test.labels[:6].numpy(), future=test_teacher[:6, 4:].numpy(),
                                        threshold=threshold)
    groups = sorted({(row['method'], row['condition']) for row in results})
    summary = []
    for method, condition in groups:
        members = [r for r in results if r['method'] == method and r['condition'] == condition]
        values = {'method': method, 'condition': condition, 'seeds': len(members)}
        for metric in ('all_f1', 'near_f1', 'far_f1', 'corridor_f1'):
            scores = [r[metric] for r in members]
            values[metric + '_mean'] = mean(scores)
            values[metric + '_sd'] = stdev(scores) if len(scores) > 1 else None
        summary.append(values)
    checks['student_future_invariance'] = 'passed for every seed and method'
    experiment['seconds_after_data_generation'] = time.perf_counter() - start
    (args.out / 'protocol.json').write_text(json.dumps(experiment, indent=2), encoding='utf-8')
    (args.out / 'results.json').write_text(json.dumps(results, indent=2), encoding='utf-8')
    (args.out / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    write_csv(args.out / 'summary.csv', summary)
    write_csv(args.out / 'per_seed.csv', [{k: v for k, v in r.items() if not isinstance(v, dict)} for r in results])
    write_csv(args.out / 'train_history.csv', histories)
    print('completed', json.dumps({'results': len(results), 'seconds': experiment['seconds_after_data_generation']}), flush=True)


if __name__ == '__main__':
    main()
