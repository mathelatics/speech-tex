#!/usr/bin/env python3
"""make_test_model.py — synthesize a tiny phonon2 GGUF (F32) for tests/CLI.

Deterministic weights: numpy default_rng(seed). The same math is mirrored in
tests/test_smoke.c's fp32 reference, so golden-token validation compares the C
engine against this file bit-for-bit-ish (float32 associativity aside).

GGUF v3 layout written here:
  magic "GGUF" | u32 ver | u64 n_tensors | u64 n_kv | kv pairs |
  tensor infos | pad to alignment | contiguous F32 payloads (aligned)

Arch keys match src/model/hparams.c ("phonon2.*"). Fixed dims must match
PH_EMBD=256 / PH_N_HEADS=4 in model.h (embd forced by hparams reader).
"""
import argparse
import struct
import sys

import numpy as np

MAGIC   = b"GGUF"
VERSION = 3
ALIGN   = 32

# ---- fixed arch (must equal model.h constants) ----
EMBD   = 256
N_HEAD = 4
HD     = EMBD // N_HEAD      # 64
FFN    = 1024                # phonon2.feed_forward_length

U8, I8, U16, I16, U32, I32, F32, BOOL, STR, ARR, U64, I64, F64 = range(13)


def w_str(f, s):
    b = s.encode("utf-8")
    f.write(struct.pack("<Q", len(b)))
    f.write(b)


def w_scalar(f, kind, value):
    fmt = {U8: "<B", I8: "<b", U16: "<H", I16: "<h", U32: "<I", I32: "<i",
           F32: "<f", BOOL: "<B", U64: "<Q", I64: "<q", F64: "<d"}[kind]
    f.write(struct.pack(fmt, value))


def w_kv(f, key, kind, value):
    w_str(f, key)
    if kind == ARR:
        elem_kind, items = value
        f.write(struct.pack("<I", ARR))
        f.write(struct.pack("<I", elem_kind))
        f.write(struct.pack("<Q", len(items)))
        for it in items:
            if elem_kind == STR:
                w_str(f, it)
            else:
                w_scalar(f, elem_kind, it)
    else:
        f.write(struct.pack("<I", kind))
        w_scalar(f, kind, value)


def build(path, seed, layers, vocab_size, ctx_train):
    rng = np.random.default_rng(seed)

    def rand(*shape, scale=0.02):
        n = int(np.prod(shape))
        return (rng.standard_normal(n) * scale).astype(np.float32)

    f = open(path, "wb")
    infos = []          # (name, shape)
    pays = []

    def T(name, shape, arr):
        assert arr.dtype == np.float32 and arr.size == int(np.prod(shape)), name
        infos.append((name, tuple(shape)))
        pays.append(arr.tobytes())

    # ---- vocabulary pieces: printable ASCII single bytes + a few words ----
    base = [chr(c) for c in range(32, 127)]                 # 95 specials-free pieces
    words = ["hello", "world", "the ", "quick", "brown ", "fox", "jumps",
             " over ", "lazy", "dog", "speech", "phonos", "test"]
    pieces = base + words
    while len(pieces) < vocab_size:
        pieces.append("x%04d" % len(pieces))
    pieces = pieces[:vocab_size]
    BOS, EOS, PAD, AUDIO = 0, 1, 2, 3
    pieces[0], pieces[1], pieces[2], pieces[3] = "<bos>", "<eos>", "<pad>", "<audio>"

    # ---- metadata ----
    metas = [
        ("general.architecture",               STR, "phonon2"),
        ("general.name",                       STR, "phonos-synth"),
        ("general.file_type",                  U32, 0),
        ("general.alignment",                  U32, ALIGN),
        ("phonon2.vocab_size",                 U32, vocab_size),
        ("phonon2.block_count",                U32, layers),
        ("phonon2.embedding_length",           U32, EMBD),
        ("phonon2.attention.head_count",       U32, N_HEAD),
        ("phonon2.feed_forward_length",        U32, FFN),
        ("phonon2.context_length",             U32, ctx_train),
        ("phonon2.rope.theta",                 F32, 10000.0),
        ("phonon2.attention.layer_norm_rms_epsilon", F32, 1e-5),
        ("phonon2.has_speech",                 U32, 0),
        ("tokenizer.ggml.model",               STR, "byte-level-greedy"),
        ("tokenizer.ggml.tokens",              ARR, (STR, pieces)),
        ("tokenizer.ggml.tokens.special",      ARR, (STR, ["<bos>", "<eos>", "<pad>", "<audio>"])),
        ("tokenizer.ggml.bos_token_id",        U32, BOS),
        ("tokenizer.ggml.eos_token_id",        U32, EOS),
        ("tokenizer.ggml.padding_token_id",    U32, PAD),
        ("tokenizer.ggml.audio_token_id",      U32, AUDIO),
    ]

    # ---- weights ----
    V = vocab_size
    T("token_embd.weight",  (V, EMBD), rand(V, EMBD, scale=0.02))
    T("output_norm.weight", (EMBD,),   rand(EMBD, scale=0.0) + 1.0)
    T("output_norm.bias",   (EMBD,),   rand(EMBD, scale=0.001))
    T("output.weight",      (V, EMBD), rand(V, EMBD, scale=0.02))
    for L in range(layers):
        p = "blk.%d." % L
        T(p+"attn_norm.weight", (EMBD,),  rand(EMBD, scale=0.0)+1.0)
        T(p+"attn_norm.bias",   (EMBD,),  rand(EMBD, scale=0.001))
        T(p+"attn_q.weight",    (EMBD, EMBD), rand(EMBD, EMBD, scale=0.02))
        T(p+"attn_k.weight",    (HD,   EMBD), rand(HD,   EMBD, scale=0.02))
        T(p+"attn_v.weight",    (HD,   EMBD), rand(HD,   EMBD, scale=0.02))
        T(p+"attn_o.weight",    (EMBD, EMBD), rand(EMBD, EMBD, scale=0.02))
        T(p+"ffn_norm.weight",  (EMBD,),  rand(EMBD, scale=0.0)+1.0)
        T(p+"ffn_norm.bias",    (EMBD,),  rand(EMBD, scale=0.001))
        T(p+"ffn_up.weight",    (FFN, EMBD), rand(FFN, EMBD, scale=0.02))
        T(p+"ffn_down.weight",  (EMBD, FFN), rand(EMBD, FFN, scale=0.02))

    # ---- serialize ----
    f.write(MAGIC)
    f.write(struct.pack("<I", VERSION))
    f.write(struct.pack("<Q", len(infos)))
    f.write(struct.pack("<Q", len(metas)))
    for k, t, v in metas:
        w_kv(f, k, t, v)

    offsets, total = [], 0
    for name, shape in infos:
        offsets.append(total)
        nbytes = 4 * int(np.prod(shape))
        total += (nbytes + ALIGN - 1) // ALIGN * ALIGN

    for (name, shape), off in zip(infos, offsets):
        w_str(f, name)
        f.write(struct.pack("<I", len(shape)))
        for d in shape:
            f.write(struct.pack("<Q", d))
        f.write(struct.pack("<I", 0))            # F32
        f.write(struct.pack("<Q", off))

    pad = (-f.tell()) % ALIGN
    f.write(b"\x00" * pad)
    data_start = f.tell()
    for (name, shape), pay, off in zip(infos, pays, offsets):
        gap = (data_start + off) - f.tell()
        assert gap >= 0, (name, gap)
        f.write(b"\x00" * gap)
        f.write(pay)
    f.close()

    import os
    size_mb = os.path.getsize(path) / 1e6
    print("wrote %s: %d tensors, %.1f MB, V=%d layers=%d ctx=%d" %
          (path, len(infos), size_mb, V, layers, ctx_train))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="models/phonos-tiny.gguf")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--vocab", type=int, default=128)
    ap.add_argument("--ctx", type=int, default=256)
    a = ap.parse_args()
    import os
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    build(a.out, a.seed, a.layers, a.vocab, a.ctx)
