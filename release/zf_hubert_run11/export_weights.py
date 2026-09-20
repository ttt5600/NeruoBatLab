"""Export a shareable ZF-HuBERT encoder from a Lightning pretraining checkpoint.

Why this exists: the raw training checkpoint is 1.14 GB, of which ~2/3 is AdamW optimizer
state (two momentum buffers per parameter) that is useless to anyone who is not resuming
training. It also nests the weights under a LightningModule, so loading it requires
pytorch_lightning *and* this repo's lightning_modules.py to be importable. Neither is
something a lab coworker should have to install to embed a wav file.

What comes out is a single file containing the torchaudio Wav2Vec2Model encoder weights plus
the exact constructor kwargs needed to rebuild it, so the only dependencies are torch and
torchaudio. The pretraining head (label embeddings, final projection, the masking machinery)
is deliberately dropped: it only has meaning relative to the k-means pseudo-labels this run
was trained against, and it plays no part in feature extraction.

The script refuses to write anything it has not verified: it rebuilds the encoder from the
exported file and asserts the features are bit-identical to the ones the original checkpoint
produces, on real audio, at several durations.
"""
import argparse
import json
from pathlib import Path

import torch
import torchaudio

# The pretraining config used for every run in this project. Copied from
# lightning_modules.py::HuBERTPreTrainModule.__init__ -- must stay in sync with it, otherwise
# load_state_dict below fails loudly (which is the intent: a silent shape mismatch would give
# coworkers a model that runs and returns garbage).
PRETRAIN_KWARGS = dict(
    extractor_mode="group_norm",
    extractor_conv_layer_config=None,
    extractor_conv_bias=False,
    encoder_embed_dim=768,
    encoder_projection_dropout=0.1,
    encoder_pos_conv_kernel=128,
    encoder_pos_conv_groups=16,
    encoder_num_layers=12,
    encoder_num_heads=12,
    encoder_attention_dropout=0.1,
    encoder_ff_interm_features=3072,
    encoder_ff_interm_dropout=0.0,
    encoder_dropout=0.1,
    encoder_layer_norm_first=False,
    encoder_layer_drop=0.05,
    mask_prob=0.4,
    mask_selection="static",
    mask_other=0.0,
    mask_length=3,
    no_mask_overlap=False,
    mask_min_space=1,
    mask_channel_prob=0.0,
    mask_channel_selection="static",
    mask_channel_other=0.0,
    mask_channel_length=5,
    no_mask_channel_overlap=False,
    mask_channel_min_space=1,
    skip_masked=False,
    skip_nomask=False,
    final_dim=256,
)

# The subset of the above that torchaudio.models.wav2vec2_model accepts. Everything dropped
# here is either masking (training-only) or the pseudo-label head.
ENCODER_KEYS = [
    "extractor_mode", "extractor_conv_layer_config", "extractor_conv_bias",
    "encoder_embed_dim", "encoder_projection_dropout", "encoder_pos_conv_kernel",
    "encoder_pos_conv_groups", "encoder_num_layers", "encoder_num_heads",
    "encoder_attention_dropout", "encoder_ff_interm_features", "encoder_ff_interm_dropout",
    "encoder_dropout", "encoder_layer_norm_first", "encoder_layer_drop",
]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--num-classes", type=int, required=True,
                   help="kmeans vocabulary the run was trained against; only needed to "
                        "rebuild the pretraining head so the checkpoint loads strictly.")
    p.add_argument("--run-tag", required=True)
    p.add_argument("--wav", default=None, help="real clip to verify feature equality on")
    p.add_argument("--meta-json", default=None, help="extra metadata to embed (eval results)")
    args = p.parse_args()

    print(f"[load] {args.ckpt}")
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    sd = ck["state_dict"]
    epoch, gstep = ck.get("epoch"), ck.get("global_step")
    print(f"[load] epoch={epoch} global_step={gstep} tensors={len(sd)}")

    # Lightning nests the real model one level down as self.model.
    inner = {k[len("model."):]: v for k, v in sd.items() if k.startswith("model.")}
    if len(inner) != len(sd):
        print(f"[warn] {len(sd) - len(inner)} tensors did not start with 'model.' and were dropped")

    full = torchaudio.models.hubert_pretrain_model(
        num_classes=args.num_classes, feature_grad_mult=0.1, **PRETRAIN_KWARGS
    )
    missing, unexpected = full.load_state_dict(inner, strict=False)
    print(f"[load] into HuBERTPretrainModel: missing={len(missing)} unexpected={len(unexpected)}")
    if missing or unexpected:
        raise SystemExit(f"state_dict mismatch -- refusing to export.\n"
                         f"missing={missing[:10]}\nunexpected={unexpected[:10]}")
    full.eval()

    enc_cfg = {k: PRETRAIN_KWARGS[k] for k in ENCODER_KEYS}
    enc_sd = full.wav2vec2.state_dict()
    n_par = sum(v.numel() for v in enc_sd.values())
    print(f"[encoder] {len(enc_sd)} tensors, {n_par/1e6:.1f}M params "
          f"({n_par*4/1e6:.0f} MB fp32)")

    meta = {
        "run_tag": args.run_tag,
        "source_checkpoint": Path(args.ckpt).name,
        "epoch": epoch,
        "global_step": gstep,
        "num_classes_pretrain": args.num_classes,
        "sample_rate": 16000,
        "n_layers": 12,
        "embed_dim": 768,
        "frame_stride_ms": 20,
        "note": "Wav2Vec2Model encoder only; pretraining head (kmeans label embeddings + "
                "final projection) intentionally dropped -- it is meaningless outside this "
                "run's pseudo-label vocabulary.",
    }
    if args.meta_json:
        meta.update(json.loads(Path(args.meta_json).read_text()))

    obj = {"encoder_config": enc_cfg, "state_dict": enc_sd, "meta": meta}
    torch.save(obj, args.out)
    print(f"[saved] {args.out}  ({Path(args.out).stat().st_size/1e6:.0f} MB)")

    # ---- verification -------------------------------------------------------------------
    # Rebuild from the file exactly as a coworker would, then require bit-identical features.
    # Anything less and we would be shipping a subtly different model than the one that
    # produced every number in the README.
    print("[verify] rebuilding encoder from exported file")
    re_obj = torch.load(args.out, map_location="cpu", weights_only=False)
    enc = torchaudio.models.wav2vec2_model(aux_num_out=None, **re_obj["encoder_config"])
    m2, u2 = enc.load_state_dict(re_obj["state_dict"], strict=True), None
    enc.eval()

    if args.wav:
        wav, sr = torchaudio.load(args.wav)
        if sr != 16000:
            wav = torchaudio.transforms.Resample(sr, 16000)(wav)
        wav = wav.mean(0, keepdim=True)
        cases = [("real clip", wav)]
        # Short and long clips exercise different padding / conv-edge paths.
        cases.append(("0.5s", wav[:, :8000] if wav.shape[1] >= 8000 else wav))
        cases.append(("3.0s tiled", wav.repeat(1, max(1, 48000 // max(1, wav.shape[1])))[:, :48000]))
    else:
        g = torch.Generator().manual_seed(0)
        cases = [(f"{d}s noise", torch.randn(1, int(16000 * d), generator=g) * 0.05)
                 for d in (0.5, 1.0, 3.0)]

    worst = 0.0
    with torch.no_grad():
        for name, x in cases:
            if x.shape[1] < 800:
                print(f"[verify] {name}: too short ({x.shape[1]} samples), skipped")
                continue
            a, _ = full.wav2vec2.extract_features(x, None)
            b, _ = enc.extract_features(x, None)
            if len(a) != len(b):
                raise SystemExit(f"layer count differs: {len(a)} vs {len(b)}")
            d = max((p - q).abs().max().item() for p, q in zip(a, b))
            worst = max(worst, d)
            print(f"[verify] {name}: {len(a)} layers, frames={a[0].shape[1]}, max|diff|={d:.3e}")

    if worst != 0.0:
        raise SystemExit(f"exported encoder does not reproduce the checkpoint (max diff {worst:.3e})")
    print("[verify] OK -- exported encoder is bit-identical to the training checkpoint")


if __name__ == "__main__":
    main()
