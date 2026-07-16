"""Decoder-only 4-part compositional a2m (audio -> face+upper+lower+hand).

Extends the face-only harness to all 4 parts, reusing the LoM MixedDatasetCB for
data (BEAT2 = all parts; TFHP/YT = face-only, body auto-masked). Pretrained backbone
(gpt2 / Qwen3): KEEP text vocab, ENLARGE with audio + 4 motion codebooks + delimiters.

Sequence (single causal stream):
  [BOS] audio [SEP] [FACE_S] face(4L) [UP_S] upper(L) [LOW_S] lower(L) [HAND_S] hand(L) [EOS]
Loss on the motion side; for face-only samples the body CODE positions are masked to -100
(delimiters + face stay supervised), mirroring MLM.forward's face_only masking.
"""
import os, sys, json, time, argparse, math
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, '.')

N_AUDIO = 16384
N_FACE = 512
N_BODY = 256                                   # hand / upper / lower each
MAX_POS = 2560                                 # BEAT2 full window ~2311 tokens


# ---------------------------------------------------------------------------
# vocab (base = pretrained text vocab size; new tokens appended after)
# ---------------------------------------------------------------------------
def make_vocab(base):
    ao = base
    fo = base + N_AUDIO
    uo = fo + N_FACE
    lo = uo + N_BODY
    ho = lo + N_BODY
    sp = ho + N_BODY
    return dict(base=base, audio_offset=ao, face_offset=fo, upper_offset=uo,
                lower_offset=lo, hand_offset=ho,
                bos=sp, sep=sp + 1, face_s=sp + 2, up_s=sp + 3, low_s=sp + 4,
                hand_s=sp + 5, eos=sp + 6, pad=sp + 7, vocab_size=sp + 8)


# ---------------------------------------------------------------------------
# config + dataset
# ---------------------------------------------------------------------------
def load_cfg(cfg_path='./configs/config_mixed_stage3_a2m_4part.yaml'):
    from omegaconf import OmegaConf
    from lom.config import get_module_config
    OmegaConf.register_new_resolver("eval", eval, replace=True)
    cfg_assets = OmegaConf.load('./configs/assets.yaml')
    cfg_base = OmegaConf.load('./configs/default.yaml')
    cfg_exp = OmegaConf.merge(cfg_base, OmegaConf.load(cfg_path))
    cfg_exp = get_module_config(cfg_exp, cfg_assets.CONFIG_FOLDER)
    return OmegaConf.merge(cfg_exp, cfg_assets)


def build_mixed_dataset(cfg, split='train', debug=False):
    from lom.data.mixed_dataset.dataset_mixed_cb import MixedDatasetCB
    return MixedDatasetCB(split=split, args=cfg.DATASET, dataset_configs=cfg.DATASET.datasets,
                          stage=cfg.TRAIN.STAGE, audio_down=int(cfg.DATASET.audio_down), debug=debug,
                          use_cache=True, save_cache=True)   # cache BEAT2 so parallel runs don't re-process


def _ids(a, lo, hi):
    return np.clip(np.asarray(a).reshape(-1).astype(np.int64), lo, hi)


class FourPartDataset(Dataset):
    """Wraps a MixedDatasetCB (or an index subset of it) -> (input_ids, labels)."""
    def __init__(self, mixed, vocab, indices=None):
        self.m = mixed
        self.v = vocab
        self.idx = list(range(len(mixed))) if indices is None else list(indices)

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, i):
        v = self.v
        s = self.m[self.idx[i]]
        face_only = bool(isinstance(s['tasks'], dict) and s['tasks'].get('face_only'))
        au = _ids(s['audio_token'], 0, N_AUDIO - 1) + v['audio_offset']
        fa = _ids(s['face_token'], 0, N_FACE - 1) + v['face_offset']
        up = _ids(s['upper_token'], 0, N_BODY - 1) + v['upper_offset']
        lo = _ids(s['lower_token'], 0, N_BODY - 1) + v['lower_offset']
        ha = _ids(s['hand_token'], 0, N_BODY - 1) + v['hand_offset']

        ids = [v['bos']] + au.tolist() + [v['sep']]
        prompt_len = len(ids)
        ids.append(v['face_s']); f0 = len(ids); ids += fa.tolist()
        ids.append(v['up_s']);   u0 = len(ids); ids += up.tolist(); u1 = len(ids)
        ids.append(v['low_s']);  l0 = len(ids); ids += lo.tolist(); l1 = len(ids)
        ids.append(v['hand_s']); h0 = len(ids); ids += ha.tolist(); h1 = len(ids)
        ids.append(v['eos'])
        ids = np.asarray(ids, dtype=np.int64)
        labels = ids.copy()
        labels[:prompt_len] = -100                    # don't supervise the prompt (audio)
        if face_only:                                 # supervise face + delimiters + EOS only
            labels[u0:u1] = -100; labels[l0:l1] = -100; labels[h0:h1] = -100
        return torch.from_numpy(ids), torch.from_numpy(labels)


def make_collate(pad):
    def collate(batch):
        maxlen = max(x[0].shape[0] for x in batch); B = len(batch)
        input_ids = torch.full((B, maxlen), pad, dtype=torch.long)
        labels = torch.full((B, maxlen), -100, dtype=torch.long)
        attn = torch.zeros((B, maxlen), dtype=torch.long)
        for i, (ids, lab) in enumerate(batch):
            L = ids.shape[0]
            input_ids[i, :L] = ids; labels[i, :L] = lab; attn[i, :L] = 1
        return {'input_ids': input_ids, 'labels': labels, 'attention_mask': attn}
    return collate


# ---------------------------------------------------------------------------
# model: pretrained backbone, KEEP text vocab, ENLARGE, extend positions to MAX_POS
# ---------------------------------------------------------------------------
def build_pretrained_4part(repo):
    import torch.nn as nn
    if 'gpt2' in repo.lower():
        from transformers import GPT2LMHeadModel
        model = GPT2LMHeadModel.from_pretrained(repo)
    else:
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(repo)
    base = model.get_input_embeddings().weight.shape[0]
    vocab = make_vocab(base)
    model.resize_token_embeddings(vocab['vocab_size'])
    tr = getattr(model, 'transformer', None)
    if tr is not None and hasattr(tr, 'wpe'):        # gpt2: extend learned positions
        cur, d = tr.wpe.weight.shape
        if cur < MAX_POS:
            new = nn.Embedding(MAX_POS, d)
            with torch.no_grad():
                new.weight[:cur].copy_(tr.wpe.weight)
                new.weight[cur:].copy_(tr.wpe.weight[-1:].expand(MAX_POS - cur, d))
            tr.wpe = new
            model.config.n_positions = MAX_POS; model.config.n_ctx = MAX_POS
            model.config.max_position_embeddings = MAX_POS
    model.config.pad_token_id = vocab['pad']
    return model, vocab


def n_params(m):
    return sum(p.numel() for p in m.parameters())


# ---------------------------------------------------------------------------
# train
# ---------------------------------------------------------------------------
def parse():
    p = argparse.ArgumentParser()
    p.add_argument('--name', required=True)
    p.add_argument('--pretrained', required=True)
    p.add_argument('--batch_size', type=int, default=4)
    p.add_argument('--lr', type=float, default=2e-4)
    p.add_argument('--min_lr_ratio', type=float, default=0.1)
    p.add_argument('--warmup', type=int, default=500)
    p.add_argument('--weight_decay', type=float, default=0.05)
    p.add_argument('--grad_clip', type=float, default=1.0)
    p.add_argument('--max_steps', type=int, default=40000)
    p.add_argument('--eval_every', type=int, default=1500)
    p.add_argument('--log_every', type=int, default=50)
    p.add_argument('--num_workers', type=int, default=6)
    p.add_argument('--grad_ckpt', action='store_true')
    p.add_argument('--grad_accum', type=int, default=1)    # micro-batches per optimizer step (single-GPU eff batch)
    p.add_argument('--out_root', default='experiments/a2f_decoder')
    p.add_argument('--seed', type=int, default=1234)
    p.add_argument('--debug', action='store_true')
    p.add_argument('--resume', default='')                 # warm-continue from a checkpoint (weights only)
    p.add_argument('--warm', default='')                    # warm-start weights but reset best (fresh cosine restart)
    p.add_argument('--cfg', default='./configs/config_mixed_stage3_a2m_4part.yaml')
    return p.parse_args()


@torch.no_grad()
def evaluate(model, loader, device, max_batches=50):
    model.eval(); tot_loss, tot_tok, tot_c = 0.0, 0, 0
    for bi, b in enumerate(loader):
        if bi >= max_batches:
            break
        ii = b['input_ids'].to(device); lab = b['labels'].to(device); at = b['attention_mask'].to(device)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            o = model(input_ids=ii, attention_mask=at, labels=lab)
        logits = o.logits[:, :-1, :]; tgt = lab[:, 1:]; mask = tgt != -100
        tot_c += (logits.argmax(-1)[mask] == tgt[mask]).sum().item()
        tot_tok += mask.sum().item(); tot_loss += o.loss.item() * mask.sum().item()
    model.train()
    return (tot_loss / max(1, tot_tok), tot_c / max(1, tot_tok))


def main():
    a = parse()
    import torch.distributed as dist
    ws = int(os.environ.get('WORLD_SIZE', '1'))
    rank = int(os.environ.get('RANK', '0'))
    local_rank = int(os.environ.get('LOCAL_RANK', '0'))
    ddp = ws > 1
    if ddp:
        dist.init_process_group('nccl')
        torch.cuda.set_device(local_rank)
        device = f'cuda:{local_rank}'
    else:
        device = 'cuda'
    is_main = rank == 0
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    out = os.path.join(a.out_root, a.name)
    if is_main:
        os.makedirs(out, exist_ok=True)
    if ddp:
        dist.barrier()
    logf = open(os.path.join(out, 'log.jsonl'), 'a') if is_main else None

    def log(d):
        if not is_main:
            return
        d = {'t': round(time.time(), 1), **d}
        logf.write(json.dumps(d) + '\n'); logf.flush()
        print(f"[{a.name}] " + '  '.join(f"{k}={v}" for k, v in d.items() if k != 't'), flush=True)

    model, vocab = build_pretrained_4part(a.pretrained)
    model = model.to(device)
    if a.grad_ckpt:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
        model.config.use_cache = False
    log({'event': 'model', 'repo': a.pretrained, 'params_M': round(n_params(model) / 1e6, 1),
         'base_vocab': vocab['base'], 'vocab_size': vocab['vocab_size'],
         'ddp_world': ws, 'eff_batch': a.batch_size * ws * max(1, a.grad_accum)})
    resume_best = float('inf')
    if a.resume:
        rck = torch.load(a.resume, map_location='cpu', weights_only=False)
        model.load_state_dict(rck['model'])
        resume_best = rck.get('val_loss', float('inf'))
        log({'event': 'resume', 'from': a.resume, 'prev_step': rck.get('step'), 'prev_val_loss': resume_best})
    if a.warm:                                              # warm-start weights but reset best (task mix changed)
        wck = torch.load(a.warm, map_location='cpu', weights_only=False)
        model.load_state_dict(wck['model'])
        log({'event': 'warm', 'from': a.warm, 'src_step': wck.get('step'), 'src_val_loss': wck.get('val_loss')})

    decay, no_decay = [], []
    for nm, p in model.named_parameters():
        (no_decay if (p.ndim < 2 or 'norm' in nm.lower() or 'ln' in nm.lower()
                      or 'bias' in nm.lower() or 'wte' in nm or 'wpe' in nm or 'embed' in nm) else decay).append(p)
    opt = torch.optim.AdamW([{'params': decay, 'weight_decay': a.weight_decay},
                             {'params': no_decay, 'weight_decay': 0.0}], lr=a.lr, betas=(0.9, 0.95), eps=1e-8)

    raw = model  # unwrapped module for eval + checkpointing (keeps ckpt single-GPU loadable)
    if ddp:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank)

    cfg = load_cfg(a.cfg)
    mixed = build_mixed_dataset(cfg, 'train', debug=a.debug)
    n = len(mixed)
    rng = np.random.RandomState(a.seed); perm = rng.permutation(n)
    n_val = max(2, int(n * 0.03)); val_idx = perm[:n_val]; train_idx = perm[n_val:]
    log({'event': 'data', 'total': n, 'train': len(train_idx), 'val': len(val_idx)})
    collate = make_collate(vocab['pad'])
    train_ds = FourPartDataset(mixed, vocab, train_idx)
    if ddp:
        sampler = torch.utils.data.distributed.DistributedSampler(train_ds, num_replicas=ws, rank=rank,
                                                                  shuffle=True, drop_last=True)
        tl = DataLoader(train_ds, batch_size=a.batch_size, sampler=sampler, drop_last=True,
                        num_workers=a.num_workers, collate_fn=collate, pin_memory=True,
                        persistent_workers=a.num_workers > 0)
    else:
        sampler = None
        tl = DataLoader(train_ds, batch_size=a.batch_size, shuffle=True, drop_last=True,
                        num_workers=a.num_workers, collate_fn=collate, pin_memory=True,
                        persistent_workers=a.num_workers > 0)
    vl = DataLoader(FourPartDataset(mixed, vocab, val_idx), batch_size=a.batch_size, shuffle=False,
                    num_workers=2, collate_fn=collate, pin_memory=True)

    def lr_at(step):
        if step < a.warmup:
            return a.lr * step / max(1, a.warmup)
        prog = min(1.0, (step - a.warmup) / max(1, a.max_steps - a.warmup))
        return a.lr * (a.min_lr_ratio + (1 - a.min_lr_ratio) * 0.5 * (1 + math.cos(math.pi * prog)))

    accum = max(1, a.grad_accum)
    step = 0; best_val = resume_best; t0 = time.time(); rl = 0.0; rt = 0; epoch = 0; micro = 0
    model.train(); opt.zero_grad(set_to_none=True)
    while step < a.max_steps:
        if sampler is not None:
            sampler.set_epoch(epoch)
        for b in tl:
            if step >= a.max_steps:
                break
            ii = b['input_ids'].to(device, non_blocking=True)
            lab = b['labels'].to(device, non_blocking=True)
            at = b['attention_mask'].to(device, non_blocking=True)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                o = model(input_ids=ii, attention_mask=at, labels=lab); loss = o.loss / accum
            loss.backward()
            nt = (lab != -100).sum().item(); rl += o.loss.item() * nt; rt += nt
            micro += 1
            if micro % accum != 0:                         # accumulate; step only every `accum` micro-batches
                continue
            lr = lr_at(step)
            for g in opt.param_groups:
                g['lr'] = lr
            torch.nn.utils.clip_grad_norm_(model.parameters(), a.grad_clip); opt.step()
            opt.zero_grad(set_to_none=True); step += 1
            if step % a.log_every == 0:
                sps = a.log_every / (time.time() - t0); t0 = time.time()
                log({'event': 'train', 'step': step, 'loss': round(rl / max(1, rt), 4),
                     'lr': round(lr, 6), 'steps_per_s': round(sps, 2)}); rl = 0.0; rt = 0
            if step % a.eval_every == 0 or step == a.max_steps:
                if is_main:
                    vloss, vacc = evaluate(raw, vl, device)
                    log({'event': 'eval', 'step': step, 'val_loss': round(vloss, 4), 'val_acc': round(vacc, 4)})
                    ck = {'model': raw.state_dict(), 'cfg': {'arch': 'pretrained_4part', 'pretrained': a.pretrained},
                          'vocab': vocab, 'step': step, 'val_loss': vloss, 'val_acc': vacc}
                    torch.save(ck, os.path.join(out, 'last.pt'))
                    if vloss < best_val:
                        best_val = vloss; torch.save(ck, os.path.join(out, 'best.pt'))
                        log({'event': 'best', 'step': step, 'val_loss': round(vloss, 4), 'val_acc': round(vacc, 4)})
                if ddp:
                    dist.barrier()
        epoch += 1
    log({'event': 'done', 'step': step, 'best_val': round(best_val, 4)})
    if is_main and logf:
        logf.close()
    if ddp:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
