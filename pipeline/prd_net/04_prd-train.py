"""
PRD-Net — Step 4: Training Loop
=================================
Trains PRDNet using the pre-built peer cache as supervision signal.

For each training patient the model receives:
  - the patient's own embedding (x)
  - the mean embedding of its positive peers (pos_proto)
  - the mean embedding of its negative peers (neg_proto)

The logit output is trained with BCEWithLogitsLoss:
  label = 1 if the patient is a long-stay case, 0 otherwise.

Steps in this file:
  1. load_caches()         — load embedding and peer caches from disk
  2. PRDDataset            — Dataset that builds (x, pos_proto, neg_proto, label)
                             per patient using the two caches  [TODO]
  3. train()               — training loop with validation                [TODO]
  4. __main__              — wire everything together                      [TODO]

Run AFTER: 01_extract-embeddings.py  (prd_net_embeddings.pkl)
           02_peer-groups.py         (prd_net_peers.pkl)
Output:    output/best_prd_model.pt
"""

import pickle
import sys
import importlib.util as _ilu
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from sklearn.metrics import f1_score

sys.path.append(str(Path(__file__).parent.parent))
from config import OUTPUT_DIR, ICD_CATEGORIES
from prd_net.config_prd import EMBEDDING_CACHE_PATH, PEER_CACHE_PATH, BATCH_SIZE, HIDDEN_DIM, LR, EPOCHS, PATIENCE, K_PEERS, AGE_TOLERANCE

# Clinical filter column names — must match 02_peer-groups.py exactly
ICU_COLS = ["icu_micu", "icu_sicu", "icu_ccu", "icu_cvicu",
            "icu_micu_sicu", "icu_tsicu", "icu_neuro_sicu"]
ICD_COLS = [f"icd_{cat}" for cat in ICD_CATEGORIES]
ADM_COLS = ["adm_emergency", "adm_urgent", "adm_elective", "adm_observation"]
_spec = _ilu.spec_from_file_location("prd_model", Path(__file__).parent / "03_prd-model.py")
_mod  = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
PRDNet = _mod.PRDNet


# ══════════════════════════════════════════════════════════════════════════════
# STEP 1 — Load caches
# ══════════════════════════════════════════════════════════════════════════════

def load_caches() -> tuple[dict, dict]:
    """
    Load the pre-built embedding and peer caches from disk.

    Returns:
        embeddings : {stay_id (int) -> np.array of shape (128,)}
                     One vector per patient covering all splits.
        peers      : {stay_id (int) -> (pos_list, neg_list)}
                     Peer indices into the training set for each training patient.
    """
    with open(EMBEDDING_CACHE_PATH, "rb") as f:
        embeddings = pickle.load(f)

    with open(PEER_CACHE_PATH, "rb") as f:
        peers = pickle.load(f)

    return embeddings, peers


# ══════════════════════════════════════════════════════════════════════════════
# STEP 2 — Dataset and DataLoader
# ══════════════════════════════════════════════════════════════════════════════

class PRDDataset(Dataset):
    """
    Lightweight dataset for PRD-Net training.
    Works directly from the embedding cache — no raw time-series needed.

    Returns per patient: (embedding tensor, label, stay_id)
    stay_id is needed at training time to look up peers in compute_prototypes.
    """

    def __init__(self, stay_ids: np.ndarray, embedding_cache: dict,
                 labels: np.ndarray):
        self.stay_ids = stay_ids
        self.embeddings = np.stack([embedding_cache[int(sid)] for sid in stay_ids])
        self.labels = labels.astype(np.float32)

    def __len__(self):
        return len(self.stay_ids)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.embeddings[idx], dtype=torch.float32),
            torch.tensor(self.labels[idx],     dtype=torch.float32),
            int(self.stay_ids[idx]),   # stay_id passed through for peer lookup
        )


def make_dataloader(stay_ids: np.ndarray, embedding_cache: dict,
                    labels: np.ndarray, shuffle: bool = True) -> DataLoader:
    """Build a DataLoader from the embedding cache for a given set of patients."""
    dataset = PRDDataset(stay_ids, embedding_cache, labels)
    return DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=shuffle)


# ══════════════════════════════════════════════════════════════════════════════
# STEP 3 — Compute peer prototypes
# ══════════════════════════════════════════════════════════════════════════════

def compute_prototypes(
    batch_patient_ids: list[int],
    peer_cache: dict,
    embedding_cache: dict,
    train_stay_ids: np.ndarray,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    For each patient in the batch, average the embeddings of their positive
    and negative peers to produce two prototype vectors.

    The peer cache stores row indices (into the training DataFrame). We convert
    those to stay_ids via train_stay_ids before looking up embeddings.

    Args:
        batch_patient_ids : list of stay_ids for the current batch
        peer_cache        : {stay_id -> (pos_row_indices, neg_row_indices)}
        embedding_cache   : {stay_id -> np.array of shape (embedding_dim,)}
        train_stay_ids    : array mapping row index → stay_id for training set

    Returns:
        pos_proto : (batch, embedding_dim) tensor — mean of positive peer embeddings
        neg_proto : (batch, embedding_dim) tensor — mean of negative peer embeddings
        Both are detached from the computation graph (fixed supervision signal).
    """
    def _weighted_mean(peer_embs: np.ndarray, target_emb: np.ndarray) -> np.ndarray:
        dists   = np.linalg.norm(peer_embs - target_emb, axis=1)
        weights = 1.0 / (dists + 1e-6)
        weights /= weights.sum()
        return (peer_embs * weights[:, None]).sum(axis=0)

    pos_protos, neg_protos = [], []

    for sid in batch_patient_ids:
        pos_idxs, neg_idxs = peer_cache[sid]
        target_emb = embedding_cache[int(sid)]

        # Convert row indices → stay_ids → embeddings, then distance-weighted mean
        pos_embs = np.stack([embedding_cache[int(train_stay_ids[i])] for i in pos_idxs])
        neg_embs = np.stack([embedding_cache[int(train_stay_ids[i])] for i in neg_idxs])

        pos_protos.append(_weighted_mean(pos_embs, target_emb))
        neg_protos.append(_weighted_mean(neg_embs, target_emb))

    pos_proto = torch.tensor(np.stack(pos_protos), dtype=torch.float32).detach()
    neg_proto = torch.tensor(np.stack(neg_protos), dtype=torch.float32).detach()

    return pos_proto, neg_proto


# ══════════════════════════════════════════════════════════════════════════════
# STEP 4 — Single training step
# ══════════════════════════════════════════════════════════════════════════════

def train_step(model, batch, peer_cache, embedding_cache,
               train_stay_ids, optimizer, loss_fn) -> float:
    """
    Run one gradient update on a single batch.

    Prototypes (128-dim raw embeddings) are encoded through model.encode()
    before the delta is computed — this projects them to hidden_dim (64)
    so they match the encoded patient h inside forward().
    Prototype encoding is wrapped in torch.no_grad() since prototypes are
    fixed supervision and should not accumulate gradients.

    Args:
        model          : PRDNet instance in train mode
        batch          : (x, y, patient_ids) from PRDDataset
        peer_cache     : {stay_id -> (pos_idxs, neg_idxs)}
        embedding_cache: {stay_id -> np.array}
        train_stay_ids : maps row index → stay_id for the training set
        optimizer      : torch optimiser
        loss_fn        : BCEWithLogitsLoss instance

    Returns:
        loss value as a plain float
    """
    x, y, patient_ids = batch

    # Build prototype tensors (128-dim raw embeddings, detached)
    # DataLoader returns patient_ids as a tensor — convert to plain Python ints
    pos_proto_raw, neg_proto_raw = compute_prototypes(
        patient_ids.tolist(), peer_cache, embedding_cache, train_stay_ids
    )

    # Encode prototypes to hidden_dim so shapes match h inside forward()
    with torch.no_grad():
        pos_proto = model.encode(pos_proto_raw)  # (batch, hidden_dim)
        neg_proto = model.encode(neg_proto_raw)  # (batch, hidden_dim)

    optimizer.zero_grad()

    logit, _, _ = model(x, pos_proto, neg_proto)  # (batch,)
    loss = loss_fn(logit, y)

    loss.backward()
    optimizer.step()

    return loss.item()


# ══════════════════════════════════════════════════════════════════════════════
# STEP 5 — Train one epoch
# ══════════════════════════════════════════════════════════════════════════════

def train_epoch(model, dataloader, peer_cache, embedding_cache,
                train_stay_ids, optimizer, loss_fn) -> float:
    """
    Run train_step for every batch in the dataloader.
    Shows a tqdm progress bar and returns the mean loss over the epoch.

    Args:
        model, dataloader, peer_cache, embedding_cache,
        train_stay_ids, optimizer, loss_fn  — passed straight to train_step

    Returns:
        mean loss over the epoch (float)
    """
    model.train()
    total_loss = 0.0

    for batch in tqdm(dataloader, desc="Training", unit="batch", leave=False):
        total_loss += train_step(
            model, batch, peer_cache, embedding_cache,
            train_stay_ids, optimizer, loss_fn
        )

    return total_loss / len(dataloader)


# ══════════════════════════════════════════════════════════════════════════════
# STEP 6 — Clinically filtered prototypes for val / test
# ══════════════════════════════════════════════════════════════════════════════

def build_filtered_prototypes(
    query_df: pd.DataFrame,
    query_stay_ids: np.ndarray,
    train_df: pd.DataFrame,
    all_train_stay_ids: np.ndarray,
    all_train_labels: np.ndarray,
    embedding_cache: dict,
    k: int,
    age_tol: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    For each query patient (val or test), apply the same ICD+ICU+age filter
    used during training peer-group construction (02_peer-groups.py), then
    select the K nearest training patients per class and return their mean
    embedding as the prototype.

    Previously, val/test used raw K-nearest without any clinical filter, while
    training used clinically filtered peers. This mismatch meant the model saw
    a structurally different prototype at eval time than it was trained on.
    Applying the same filter end-to-end removes that inconsistency.

    Falls back to unfiltered K-nearest if the filtered pool for a class is empty
    (rare patients with no ICD/ICU match in the training set).

    Args:
        query_df          : DataFrame for query patients (X_val or X_test, unscaled)
        query_stay_ids    : ordered stay_id array for query patients
        train_df          : X_train (unscaled) — contains ICD, ICU, age columns
        all_train_stay_ids: stay_id array aligned with train_df rows
        all_train_labels  : binary label array aligned with train_df rows
        embedding_cache   : {stay_id -> np.array (128,)}
        k                 : number of peers per class (K_PEERS)
        age_tol           : maximum age difference in years (AGE_TOLERANCE)

    Returns:
        pos_proto_raw : (N_query, 128) mean embedding of K nearest positive peers
        neg_proto_raw : (N_query, 128) mean embedding of K nearest negative peers
    """
    all_train_emb = np.stack([embedding_cache[int(sid)] for sid in all_train_stay_ids])

    train_icd = train_df[ICD_COLS].values   # (N_train, n_icd)
    train_icu = train_df[ICU_COLS].values   # (N_train, n_icu)
    train_adm = train_df[ADM_COLS].values   # (N_train, n_adm)
    train_age = train_df["age"].values       # (N_train,)

    pos_global_idx = np.where(all_train_labels == 1)[0]
    neg_global_idx = np.where(all_train_labels == 0)[0]

    # Build stay_id → row-index lookup for the query DataFrame
    sid_to_row = {int(sid): i for i, sid in enumerate(query_df["stay_id"].values)}
    query_icd  = query_df[ICD_COLS].values
    query_icu  = query_df[ICU_COLS].values
    query_adm  = query_df[ADM_COLS].values
    query_age  = query_df["age"].values

    N          = len(query_stay_ids)
    emb_dim    = all_train_emb.shape[1]
    pos_protos = np.zeros((N, emb_dim), dtype=np.float32)
    neg_protos = np.zeros((N, emb_dim), dtype=np.float32)

    def _knn_mean(candidates: np.ndarray, target_emb: np.ndarray) -> np.ndarray:
        dists   = np.linalg.norm(all_train_emb[candidates] - target_emb, axis=1)
        k_use   = min(k, len(candidates))
        top_idx = np.argsort(dists)[:k_use]
        top_embs = all_train_emb[candidates[top_idx]]
        top_dists = dists[top_idx]
        weights = 1.0 / (top_dists + 1e-6)
        weights /= weights.sum()
        return (top_embs * weights[:, None]).sum(axis=0)

    for i, sid in enumerate(tqdm(query_stay_ids, desc="Filtered prototypes", leave=False)):
        q_row  = sid_to_row[int(sid)]
        target = embedding_cache[int(sid)]

        # Hard filter: same primary ICD chapter + ICU type + admission type (binary match)
        q_icd = int(np.argmax(query_icd[q_row])) if query_icd[q_row].max() == 1 else None
        q_icu = int(np.argmax(query_icu[q_row])) if query_icu[q_row].max() == 1 else None
        q_adm = int(np.argmax(query_adm[q_row])) if query_adm[q_row].max() == 1 else None

        mask = np.ones(len(train_df), dtype=bool)
        if q_icd is not None:
            mask &= train_icd[:, q_icd] == 1
        if q_icu is not None:
            mask &= train_icu[:, q_icu] == 1
        if q_adm is not None:
            mask &= train_adm[:, q_adm] == 1

        # Age filter: within ±age_tol years
        mask &= np.abs(train_age - query_age[q_row]) <= age_tol

        candidates = np.where(mask)[0]
        pos_cands  = candidates[all_train_labels[candidates] == 1]
        neg_cands  = candidates[all_train_labels[candidates] == 0]

        # Fall back to unfiltered pool if no match survives the filter
        if len(pos_cands) == 0:
            pos_cands = pos_global_idx
        if len(neg_cands) == 0:
            neg_cands = neg_global_idx

        pos_protos[i] = _knn_mean(pos_cands, target)
        neg_protos[i] = _knn_mean(neg_cands, target)

    return torch.tensor(pos_protos), torch.tensor(neg_protos)


# ══════════════════════════════════════════════════════════════════════════════
# STEP 7 — Validation epoch
# ══════════════════════════════════════════════════════════════════════════════

def val_epoch(model, val_emb: torch.Tensor, val_labels: torch.Tensor,
              val_pos_proto_raw: torch.Tensor, val_neg_proto_raw: torch.Tensor,
              loss_fn) -> tuple[float, float]:
    """
    Evaluate on the full validation set using pre-computed clinically filtered
    prototypes (built once before training with ICD+ICU+age filter).

    Prototypes are re-encoded with current model weights each call so they
    reflect the evolving representation space.

    Args:
        model             : PRDNet instance
        val_emb           : (N_val, input_dim) val patient embeddings
        val_labels        : (N_val,) binary labels
        val_pos_proto_raw : (N_val, input_dim) pre-computed positive prototypes
        val_neg_proto_raw : (N_val, input_dim) pre-computed negative prototypes
        loss_fn           : BCEWithLogitsLoss instance

    Returns:
        (val loss, val F1) — tuple of floats
    """
    model.eval()
    with torch.no_grad():
        pos_proto    = model.encode(val_pos_proto_raw)
        neg_proto    = model.encode(val_neg_proto_raw)
        logits, _, _ = model(val_emb, pos_proto, neg_proto)
        loss         = loss_fn(logits, val_labels).item()

    logits_np = logits.numpy()
    labels_np = val_labels.numpy()
    probs     = 1 / (1 + np.exp(-logits_np))
    preds     = (probs >= 0.5).astype(int)
    val_f1    = f1_score(labels_np, preds, zero_division=0)

    return loss, val_f1


# ══════════════════════════════════════════════════════════════════════════════
# STEP 8 — Threshold tuning
# ══════════════════════════════════════════════════════════════════════════════

def find_best_threshold(logits: np.ndarray, labels: np.ndarray) -> tuple[float, float]:
    """
    Scan decision thresholds in [0.10, 0.90] and return the one maximising F1.

    Call this once on val logits from the best checkpoint — never on test data,
    to avoid leaking label information into the threshold choice.

    Returns:
        (best_threshold, best_f1)
    """
    probs = 1 / (1 + np.exp(-logits))
    best_thresh, best_f1 = 0.5, 0.0
    for t in np.arange(0.10, 0.91, 0.01):
        preds = (probs >= t).astype(int)
        f1 = f1_score(labels, preds, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = round(float(t), 2)
    return best_thresh, best_f1


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":

    # ── Load caches ───────────────────────────────────────────────────────────
    print("Loading caches...")
    embedding_cache, peer_cache = load_caches()
    print(f"  Embeddings : {len(embedding_cache):,} patients")
    print(f"  Peer cache : {len(peer_cache):,} patients")

    # ── Load training labels and stay_id order ────────────────────────────────
    print("Loading training data...")
    X_train = pd.read_parquet(OUTPUT_DIR / "X_train.parquet")
    y_train = pd.read_parquet(OUTPUT_DIR / "y_train.parquet")

    # all_train_stay_ids: full row-index → stay_id lookup used by compute_prototypes
    # Must stay unfiltered — peer row indices reference positions in this array
    all_train_stay_ids = X_train["stay_id"].values
    labels_all = y_train.set_index("stay_id").loc[all_train_stay_ids, "los_gt7"].values

    # Filter out the 6 patients with 0 peers on either side — they crash compute_prototypes
    valid_mask = np.array([
        sid in peer_cache and len(peer_cache[sid][0]) > 0 and len(peer_cache[sid][1]) > 0
        for sid in all_train_stay_ids
    ])
    train_stay_ids = all_train_stay_ids[valid_mask]
    labels         = labels_all[valid_mask]
    print(f"  Training patients (after peer filter) : {len(train_stay_ids):,}")

    # ── Load validation data ──────────────────────────────────────────────────
    print("Loading validation data...")
    X_val = pd.read_parquet(OUTPUT_DIR / "X_val.parquet")
    y_val = pd.read_parquet(OUTPUT_DIR / "y_val.parquet")

    val_stay_ids = X_val["stay_id"].values
    val_labels   = y_val.set_index("stay_id").loc[val_stay_ids, "los_gt7"].values
    print(f"  Validation patients : {len(val_stay_ids):,}")

    # ── Build train dataloader ────────────────────────────────────────────────
    train_loader = make_dataloader(train_stay_ids, embedding_cache, labels, shuffle=True)

    # ── Val embeddings and labels as tensors (no DataLoader needed) ───────────
    val_emb      = torch.tensor(
        np.stack([embedding_cache[int(sid)] for sid in val_stay_ids]), dtype=torch.float32
    )
    val_labels_t = torch.tensor(val_labels, dtype=torch.float32)

    # ── Clinically filtered val prototypes — computed once, re-encoded each epoch
    # Applies the same ICD+ICU+age filter as training peer groups (02_peer-groups.py)
    # so the model sees a consistent prototype structure at val time.
    print("Building clinically filtered val prototypes...")
    val_pos_proto_raw, val_neg_proto_raw = build_filtered_prototypes(
        X_val, val_stay_ids, X_train, all_train_stay_ids, labels_all,
        embedding_cache, K_PEERS, AGE_TOLERANCE,
    )
    print(f"  Done — {len(val_stay_ids):,} val prototypes built")

    # ── Model, optimizer, loss ────────────────────────────────────────────────
    INPUT_DIM = next(iter(embedding_cache.values())).shape[0]  # 128
    model     = PRDNet(input_dim=INPUT_DIM, hidden_dim=HIDDEN_DIM)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    # pos_weight computed dynamically from training labels to counteract class imbalance
    n_pos = int(labels.sum())
    n_neg = len(labels) - n_pos
    pos_weight = torch.tensor([n_neg / n_pos], dtype=torch.float32)
    loss_fn   = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    print(f"\nPRDNet  |  input={INPUT_DIM}  hidden={HIDDEN_DIM}  "
          f"params={sum(p.numel() for p in model.parameters()):,}")
    print(f"Training  |  epochs={EPOCHS}  batch={BATCH_SIZE}  lr={LR}\n")

    # ── Training loop ─────────────────────────────────────────────────────────
    best_val_f1    = 0.0
    epochs_no_improve = 0
    ckpt_dir  = Path(__file__).parent / "checkpoints"
    ckpt_dir.mkdir(exist_ok=True)
    ckpt_path = ckpt_dir / "prd_net_v1.pt"

    print(f"{'Epoch':<7} {'Train Loss':<12} {'Val Loss':<12} {'Val F1':<10} {'Best'}")
    print("─" * 50)
    for epoch in range(1, EPOCHS + 1):
        train_loss       = train_epoch(model, train_loader, peer_cache, embedding_cache,
                                       all_train_stay_ids, optimizer, loss_fn)
        val_loss, val_f1 = val_epoch(model, val_emb, val_labels_t,
                                         val_pos_proto_raw, val_neg_proto_raw, loss_fn)

        is_best = val_f1 > best_val_f1
        if is_best:
            best_val_f1 = val_f1
            epochs_no_improve = 0
            torch.save(model.state_dict(), ckpt_path)
        else:
            epochs_no_improve += 1

        marker = " ✓" if is_best else f" (no improve {epochs_no_improve}/{PATIENCE})"
        print(f"{epoch:>3}/{EPOCHS}  {train_loss:<12.4f} {val_loss:<12.4f} {val_f1:<10.4f}{marker}")

        if epochs_no_improve >= PATIENCE:
            print(f"\nEarly stopping — val F1 did not improve for {PATIENCE} epochs.")
            break

    print(f"\nBest val F1 : {best_val_f1:.4f}")
    print(f"Saved       : {ckpt_path}")

    # ── Threshold tuning on best checkpoint ───────────────────────────────────
    print("\nTuning decision threshold on val set (best checkpoint)...")
    model.load_state_dict(torch.load(ckpt_path, weights_only=True))
    model.eval()
    with torch.no_grad():
        pos_proto    = model.encode(val_pos_proto_raw)
        neg_proto    = model.encode(val_neg_proto_raw)
        logits, _, _ = model(val_emb, pos_proto, neg_proto)
    best_thresh, tuned_f1 = find_best_threshold(logits.numpy(), val_labels)
    print(f"  Threshold 0.50 → F1 {best_val_f1:.4f}")
    print(f"  Threshold {best_thresh:.2f}  → F1 {tuned_f1:.4f}  (optimal)")

    thresh_path = ckpt_dir / "prd_net_v1_threshold.pt"
    torch.save({"threshold": best_thresh}, thresh_path)
    print(f"  Saved threshold : {thresh_path}")
