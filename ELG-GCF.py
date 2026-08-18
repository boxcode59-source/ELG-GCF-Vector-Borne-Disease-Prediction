"""
================================================================================
ELG-GCF: Explainable, Evidence-Grounded LLM-Guided GRU-CDeepNN Framework
for Vector-Borne Disease Prediction
================================================================================
Single-cell implementation covering every module described in the paper:

  3.2  Data Preprocessing
  3.3  Hierarchical Patient Symptom Profiling & Clinical Ontology Construction
  3.4  LLM-Guided Semantic Symptom Relation Mapping        (LLM-SSRM)
  3.5  LLM-Enhanced Disease Prototype Construction          (LLM-DPC)
  3.6  Prototype-Guided Disease Similarity Encoding         (PGDSE)
  3.7  GRU-CDeepNN Disease Prediction
  3.8  SHAP-Counterfactual Disease Explanation               (SHAP-CDE)
  3.9  Evidence-Grounded LLM Clinical Recommendation
  4.x  Training curves, 5-fold CV, ablation study, statistical analysis

Dataset : Kaggle "Vector Borne Disease Prediction" (trainn.csv)
          https://www.kaggle.com/datasets/richardbernat/vector-borne-disease-prediction
          -> 64 binary symptom columns + 1 "prognosis" (disease) label, 11 classes.

Run:
    pip install pandas numpy scikit-learn tensorflow shap matplotlib scipy anthropic
    python ELG_GCF_full_implementation.py

Notes on the LLM components
----------------------------
The paper uses an LLM (3.4.2-3.4.3, 3.9.2) to (a) classify the semantic
relation between symptom pairs and (b) generate evidence-grounded clinical
recommendations. Both hooks are implemented with a real Anthropic API call
path (`use_llm_api=True` + `ANTHROPIC_API_KEY` env var) *and* a fast,
deterministic co-occurrence-based proxy that is used automatically when no
API key/client is available, so the full pipeline (including SHAP-CDE and
GRU-CDeepNN training) can be executed and evaluated end-to-end without any
external API calls.
"""

import os
import json
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

import tensorflow as tf
from tensorflow.keras import layers, models

import shap

# --------------------------------------------------------------------------
# CONFIG
# --------------------------------------------------------------------------
CSV_PATH             = "trainn.csv"     # path to the downloaded Kaggle dataset
RANDOM_STATE          = 42
ACTIVATION_THRESHOLD  = 0.02            # tau  (eq. 19) minimum symptom prevalence to be considered
RELATION_THRESHOLD    = 0.30            # theta (eq. 25) minimum relation strength to be retained
TOP_K_SHAP            = 5               # k    (eq. 46) dominant symptoms kept for explanation
EPOCHS                = 100
BATCH_SIZE            = 32
USE_LLM_API           = False           # set True + configure ANTHROPIC_API_KEY to use real LLM calls
RELATION_TYPES        = ["co-occurring", "causal", "differential", "unrelated"]

np.random.seed(RANDOM_STATE)
tf.random.set_seed(RANDOM_STATE)

# Minimal evidence snippets for the Evidence-Grounded LLM Recommendation module (3.9.1).
# Extend this dictionary with authoritative guideline text for all 11 disease classes.
CLINICAL_EVIDENCE_DB = {
    "Dengue":              "WHO guidance: monitor for warning signs (persistent vomiting, abdominal pain, "
                            "bleeding), ensure careful fluid management, avoid NSAIDs/aspirin.",
    "Malaria":              "WHO guidance: confirm parasitologically via RDT/microscopy, start artemisinin-based "
                            "combination therapy (ACT) promptly, monitor for severe-malaria danger signs.",
    "Chikungunya":          "CDC guidance: supportive care, rest, fluids, and analgesics; avoid NSAIDs until "
                            "dengue is excluded due to bleeding risk.",
    "Zika":                 "CDC guidance: supportive care; special counselling and monitoring required for "
                            "pregnant patients due to congenital risk.",
    "Yellow Fever":         "WHO guidance: supportive care, monitor liver/renal function, vaccination is the "
                            "primary prevention strategy.",
    "Rift Valley Fever":    "WHO guidance: supportive care, monitor for hemorrhagic and ocular complications.",
    "West Nile Fever":      "CDC guidance: supportive care; monitor for neuro-invasive progression in older or "
                            "immunocompromised patients.",
    "Tungiasis":            "Guidance: mechanical removal of embedded sand fleas, wound care, tetanus prophylaxis "
                            "as indicated.",
    "Japanese Encephalitis":"WHO guidance: supportive/critical care for neurological symptoms; vaccination is "
                            "the primary prevention strategy.",
    "Plague":                "WHO guidance: immediate antibiotic therapy (e.g., streptomycin/gentamicin), strict "
                            "isolation for pneumonic plague.",
    "Lyme disease":          "CDC guidance: antibiotic therapy (e.g., doxycycline) is generally curative when "
                            "started early.",
}


# ==========================================================================
# 3.2  DATA COLLECTION & PREPROCESSING
# ==========================================================================
def load_and_preprocess(csv_path):
    """Sections 3.1-3.2: ingestion, schema profiling, identifier removal,
    binary validation, mode imputation, duplicate removal, label encoding."""
    df = pd.read_csv(csv_path)

    # 3.2.2 identifier removal
    id_cols = [c for c in df.columns if c.lower() in ("id", "patient_id")]
    df = df.drop(columns=id_cols, errors="ignore")

    label_col = "prognosis" if "prognosis" in df.columns else df.columns[-1]
    symptom_cols = [c for c in df.columns if c != label_col]

    # 3.2.2 binary-value verification
    for c in symptom_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # 3.2.3 missing-value detection + conditional mode imputation (eq. 7-8)
    for c in symptom_cols:
        if df[c].isna().any():
            mode_val = df[c].mode(dropna=True)
            mode_val = mode_val.iloc[0] if len(mode_val) else 0
            df[c] = df[c].fillna(mode_val)
    df[symptom_cols] = (df[symptom_cols] > 0.5).astype(int)

    # 3.2.4 duplicate removal (eq. 9)
    df = df.drop_duplicates(subset=symptom_cols + [label_col]).reset_index(drop=True)

    # 3.2.5 disease-label standardization & encoding (eq. 10-11)
    le = LabelEncoder()
    df["label_idx"] = le.fit_transform(df[label_col].astype(str).str.strip())

    X = df[symptom_cols].values.astype(np.float32)
    y = df["label_idx"].values.astype(np.int64)
    return df, X, y, symptom_cols, le


# ==========================================================================
# 3.3  HIERARCHICAL PATIENT SYMPTOM PROFILING & CLINICAL ONTOLOGY
# ==========================================================================
def compute_cooccurrence(X):
    """eq. 13: raw symptom co-occurrence via outer-product accumulation."""
    C = X.T @ X
    return C


def normalize_association(C):
    """eq. 14: PMI-style normalized pairwise association strength."""
    diag = np.diag(C)
    eps = 1e-8
    denom = np.sqrt(np.outer(diag, diag)) + eps
    A = C / denom
    np.fill_diagonal(A, 0.0)
    return A


def build_clinical_ontology_groups(symptom_cols, n_groups=8, random_state=RANDOM_STATE):
    """eq. 15: symptom-to-clinical-category membership matrix M.
    (Deterministic surrogate grouping — replace with a real clinical
    ontology mapping, e.g. UMLS/SNOMED categories, if available.)"""
    n = len(symptom_cols)
    rng = np.random.RandomState(random_state)
    group_ids = rng.randint(0, n_groups, size=n)
    M = np.zeros((n, n_groups))
    for i, g in enumerate(group_ids):
        M[i, g] = 1.0
    return M, group_ids


def hierarchical_profile(X, M, H=None):
    """eq. 16-17: patient-level clinical-category profile, optionally
    compressed further with a hierarchy matrix H."""
    P = X @ M
    if H is not None:
        P = P @ H
    return P


# ==========================================================================
# 3.4  LLM-GUIDED SEMANTIC SYMPTOM RELATION MAPPING (LLM-SSRM)
# ==========================================================================
def llm_relation_classifier_proxy(A_norm, i, j):
    """Deterministic proxy for eq. 22-23 (LLM relation-probability vector),
    driven by normalized co-occurrence strength."""
    strength = A_norm[i, j]
    if strength > 0.5:
        probs = [0.70, 0.15, 0.10, 0.05]
    elif strength > 0.2:
        probs = [0.40, 0.30, 0.20, 0.10]
    else:
        probs = [0.15, 0.15, 0.20, 0.50]
    probs = np.array(probs) / sum(probs)
    return probs


def llm_relation_classifier_api(sym_i, sym_j, client, model="claude-sonnet-4-6"):
    """Real LLM call implementing eq. 21-23: structured clinical prompt ->
    relation-probability vector. Requires the `anthropic` package and a
    configured client. Falls back to None on any failure so the caller can
    use the proxy instead."""
    prompt = (
        f"Classify the clinical relationship between the symptoms '{sym_i}' and "
        f"'{sym_j}' as exactly one of {RELATION_TYPES}. Respond with only the label."
    )
    try:
        resp = client.messages.create(
            model=model, max_tokens=10,
            messages=[{"role": "user", "content": prompt}],
        )
        label = resp.content[0].text.strip().lower()
        probs = np.array([1.0 if r == label else 0.0 for r in RELATION_TYPES])
        if probs.sum() == 0:
            return None
        # small smoothing so downstream confidence isn't exactly 0/1
        return 0.9 * probs + 0.1 * (1 - probs) / (len(RELATION_TYPES) - 1)
    except Exception:
        return None


def llm_ssrm(X, symptom_cols, tau=ACTIVATION_THRESHOLD, theta=RELATION_THRESHOLD,
             use_api=False, api_client=None):
    """Full LLM-SSRM pipeline: eq. 18-27."""
    n_patients, n_sym = X.shape
    freq = X.mean(axis=0)                                   # eq. 18
    relevant_idx = np.where(freq >= tau)[0]                  # eq. 19

    C = compute_cooccurrence(X)
    A_norm = normalize_association(C)                        # eq. 14

    S = np.zeros((n_sym, n_sym))
    pairs = list(itertools.combinations(relevant_idx, 2))    # eq. 20
    for i, j in pairs:
        probs = None
        if use_api and api_client is not None:
            probs = llm_relation_classifier_api(symptom_cols[i], symptom_cols[j], api_client)
        if probs is None:
            probs = llm_relation_classifier_proxy(A_norm, i, j)

        conf = probs[int(np.argmax(probs))]                  # eq. 23
        strength = 0.5 * conf + 0.5 * A_norm[i, j]            # eq. 24
        if strength >= theta:                                 # eq. 25
            S[i, j] = strength
            S[j, i] = strength                                # symmetry: S_ij = S_ji
    return S, A_norm, freq


def relation_aware_representation(X, S):
    """eq. 27: relation-aware patient representation (residual propagation
    through the semantic relation matrix)."""
    return X + X @ S


# ==========================================================================
# 3.5  LLM-ENHANCED DISEASE PROTOTYPE CONSTRUCTION (LLM-DPC)
# ==========================================================================
def build_disease_prototypes(X_rel, y, S, n_classes):
    """eq. 28-32."""
    d = X_rel.shape[1]
    prototypes = np.zeros((n_classes, d))
    for c in range(n_classes):
        idx = np.where(y == c)[0]                             # eq. 28
        if len(idx) == 0:
            continue
        mean_vec = X_rel[idx].mean(axis=0)                    # eq. 29
        w = np.abs(mean_vec) / (np.abs(mean_vec).sum() + 1e-8)  # eq. 30
        weighted = mean_vec * w
        enhanced = weighted + weighted @ S                    # eq. 31
        norm = np.linalg.norm(enhanced) + 1e-8
        prototypes[c] = enhanced / norm                        # eq. 32
    return prototypes


# ==========================================================================
# 3.6  PROTOTYPE-GUIDED DISEASE SIMILARITY ENCODING (PGDSE)
# ==========================================================================
def pgdse_encode(X_rel, prototypes, scaler=None, fit_scaler=True):
    """eq. 33-36."""
    X_norm = X_rel / (np.linalg.norm(X_rel, axis=1, keepdims=True) + 1e-8)
    P_norm = prototypes / (np.linalg.norm(prototypes, axis=1, keepdims=True) + 1e-8)

    sim = X_norm @ P_norm.T                                    # eq. 33-34
    fused = np.concatenate([X_rel, sim], axis=1)                # eq. 35

    if scaler is None:
        scaler = StandardScaler()
    fused_norm = scaler.fit_transform(fused) if fit_scaler else scaler.transform(fused)  # eq. 36
    return fused_norm, sim, scaler


# ==========================================================================
# 3.7  GRU-CDeepNN DISEASE PREDICTION
# ==========================================================================
def build_gru_cdeepnn(input_dim, n_classes):
    """eq. 37-43: GRU dependency learning + deep nonlinear transform + softmax."""
    inp = layers.Input(shape=(input_dim, 1))
    g = layers.GRU(128, return_sequences=True)(inp)             # eq. 38-40
    g = layers.GRU(64)(g)
    d = layers.Dense(128, activation="relu")(g)                  # eq. 41
    d = layers.Dropout(0.3)(d)
    d = layers.Dense(64, activation="relu")(d)
    out = layers.Dense(n_classes, activation="softmax")(d)       # eq. 42-43
    model = models.Model(inp, out)
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


def to_seq(X):
    return X.reshape(*X.shape, 1)


# ==========================================================================
# 3.8  SHAP-COUNTERFACTUAL DISEASE EXPLANATION (SHAP-CDE)
# ==========================================================================
def shap_cde_explain(model, X_background, X_instance, class_idx, alt_class_idx,
                      top_k=TOP_K_SHAP, feature_names=None):
    """eq. 44-50: SHAP attribution + sparsity-constrained counterfactual."""

    def predict_fn(x):
        return model.predict(to_seq(x), verbose=0)

    background = shap.sample(X_background, min(50, len(X_background)))
    explainer = shap.KernelExplainer(predict_fn, background)
    shap_vals = explainer.shap_values(X_instance.reshape(1, -1), nsamples=100)  # eq. 44

    phi = np.array(shap_vals)[class_idx].flatten()
    importance = np.abs(phi) / (np.abs(phi).sum() + 1e-8)        # eq. 45
    dominant_idx = np.argsort(-importance)[:top_k]                # eq. 46

    phi_alt = np.array(shap_vals)[alt_class_idx].flatten()
    direction = np.sign(phi_alt - phi)                             # eq. 47

    # eq. 48: sparse, constrained counterfactual generation
    counterfactual = X_instance.copy()
    for idx in dominant_idx:
        counterfactual[idx] = np.clip(counterfactual[idx] + 0.5 * direction[idx], 0, 1)

    cf_pred = predict_fn(counterfactual.reshape(1, -1))[0]
    cf_class = int(np.argmax(cf_pred))                              # eq. 49

    explanation = {                                                  # eq. 50
        "predicted_class": int(class_idx),
        "dominant_symptoms": [feature_names[i] if feature_names else int(i) for i in dominant_idx],
        "shap_values": phi[dominant_idx].tolist(),
        "counterfactual_target_class": int(alt_class_idx),
        "counterfactual_achieved_class": cf_class,
        "counterfactual_success": cf_class == alt_class_idx,
    }
    return explanation


# ==========================================================================
# 3.9  EVIDENCE-GROUNDED LLM CLINICAL RECOMMENDATION
# ==========================================================================
def evidence_grounded_recommendation(disease_name, explanation, client=None, model="claude-sonnet-4-6"):
    """eq. 51-52: retrieve relevant clinical evidence, condition LLM
    generation on prediction + explanation + evidence."""
    evidence = CLINICAL_EVIDENCE_DB.get(disease_name, "General vector-borne disease precaution guidance.")
    prompt = (
        f"Predicted disease: {disease_name}\n"
        f"Key contributing symptoms (SHAP-CDE): {explanation['dominant_symptoms']}\n"
        f"Relevant clinical evidence: {evidence}\n\n"
        f"Provide a short, safe, clinical decision-support note (NOT a diagnosis) "
        f"summarizing next steps a clinician should consider, emphasizing professional "
        f"assessment before any treatment decision."
    )
    if client is not None:
        try:
            resp = client.messages.create(
                model=model, max_tokens=300,
                messages=[{"role": "user", "content": prompt}],
            )
            return resp.content[0].text
        except Exception as e:
            return f"[LLM call failed: {e}] Retrieved evidence: {evidence}"
    return f"[No LLM client configured — returning retrieved evidence only]\n{evidence}"


# ==========================================================================
# ABLATION HELPER — trains a restricted variant of the pipeline
# ==========================================================================
def train_variant(X, y, n_classes, use_ssrm=True, use_pgdse=True, epochs=20):
    S, _, _ = llm_ssrm(X, [str(i) for i in range(X.shape[1])]) if use_ssrm else (
        np.zeros((X.shape[1], X.shape[1])), None, None)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=RANDOM_STATE)

    X_train_rel = relation_aware_representation(X_train, S) if use_ssrm else X_train
    X_test_rel = relation_aware_representation(X_test, S) if use_ssrm else X_test

    if use_pgdse:
        prototypes = build_disease_prototypes(X_train_rel, y_train, S, n_classes)
        X_train_final, _, scaler = pgdse_encode(X_train_rel, prototypes)
        X_test_final, _, _ = pgdse_encode(X_test_rel, prototypes, scaler=scaler, fit_scaler=False)
    else:
        scaler = StandardScaler()
        X_train_final = scaler.fit_transform(X_train_rel)
        X_test_final = scaler.transform(X_test_rel)

    model = build_gru_cdeepnn(X_train_final.shape[1], n_classes)
    model.fit(to_seq(X_train_final), y_train, epochs=epochs, batch_size=BATCH_SIZE, verbose=0)
    preds = np.argmax(model.predict(to_seq(X_test_final), verbose=0), axis=1)
    return accuracy_score(y_test, preds)


# ==========================================================================
# END-TO-END PIPELINE
# ==========================================================================
def run_pipeline(csv_path=CSV_PATH, epochs=EPOCHS, batch_size=BATCH_SIZE, use_llm_api=USE_LLM_API):
    api_client = None
    if use_llm_api:
        try:
            import anthropic
            api_client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from env
        except Exception as e:
            print(f"LLM API unavailable, using proxy relation classifier instead: {e}")

    # ---- 3.1-3.2 Data ----
    df, X, y, symptom_cols, le = load_and_preprocess(csv_path)
    n_classes = len(le.classes_)

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=RANDOM_STATE)          # eq. 2
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=RANDOM_STATE)

    # ---- 3.4 LLM-SSRM ----
    S, A_norm, freq = llm_ssrm(X_train, symptom_cols, use_api=use_llm_api, api_client=api_client)
    X_train_rel = relation_aware_representation(X_train, S)
    X_val_rel = relation_aware_representation(X_val, S)
    X_test_rel = relation_aware_representation(X_test, S)

    # ---- 3.5 LLM-DPC ----
    prototypes = build_disease_prototypes(X_train_rel, y_train, S, n_classes)

    # ---- 3.6 PGDSE ----
    X_train_p, _, scaler = pgdse_encode(X_train_rel, prototypes)
    X_val_p, _, _ = pgdse_encode(X_val_rel, prototypes, scaler=scaler, fit_scaler=False)
    X_test_p, _, _ = pgdse_encode(X_test_rel, prototypes, scaler=scaler, fit_scaler=False)

    # ---- 3.7 GRU-CDeepNN ----
    model = build_gru_cdeepnn(X_train_p.shape[1], n_classes)
    history = model.fit(
        to_seq(X_train_p), y_train,
        validation_data=(to_seq(X_val_p), y_val),
        epochs=epochs, batch_size=batch_size, verbose=1,
    )

    y_pred = np.argmax(model.predict(to_seq(X_test_p), verbose=0), axis=1)
    acc = accuracy_score(y_test, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_test, y_pred, average="weighted", zero_division=0)
    print(f"\nTest Accuracy: {acc:.4f} | Precision: {prec:.4f} | Recall: {rec:.4f} | F1: {f1:.4f}")
    print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))

    # ---- 4.1 Training curves ----
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history["accuracy"], label="train")
    plt.plot(history.history["val_accuracy"], label="val")
    plt.title("Accuracy"); plt.xlabel("Epoch"); plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(history.history["loss"], label="train")
    plt.plot(history.history["val_loss"], label="val")
    plt.title("Loss"); plt.xlabel("Epoch"); plt.legend()
    plt.tight_layout()
    plt.savefig("training_curves.png", dpi=150)
    plt.close()
    print("Saved training_curves.png")

    # ---- 3.8 SHAP-CDE on one test instance ----
    inst_idx = 0
    pred_class = int(y_pred[inst_idx])
    inst_probs = model.predict(to_seq(X_test_p[inst_idx:inst_idx + 1]), verbose=0)[0]
    alt_class = int(np.argsort(-inst_probs)[1])
    explanation = shap_cde_explain(
        model, X_train_p, X_test_p[inst_idx], pred_class, alt_class,
        feature_names=symptom_cols + list(le.classes_),
    )
    print("\nSHAP-CDE Explanation:\n", json.dumps(explanation, indent=2))

    # ---- 3.9 Evidence-grounded recommendation ----
    disease_name = le.inverse_transform([pred_class])[0]
    recommendation = evidence_grounded_recommendation(disease_name, explanation, client=api_client)
    print("\nEvidence-Grounded Clinical Recommendation:\n", recommendation)

    # ---- 4.3 5-fold cross-validation ----
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    fold_scores = []
    for fold, (tr_idx, te_idx) in enumerate(skf.split(X, y), 1):
        S_f, _, _ = llm_ssrm(X[tr_idx], symptom_cols, use_api=False)
        Xf_train_rel = relation_aware_representation(X[tr_idx], S_f)
        Xf_test_rel = relation_aware_representation(X[te_idx], S_f)
        protos_f = build_disease_prototypes(Xf_train_rel, y[tr_idx], S_f, n_classes)
        Xf_train_p, _, scaler_f = pgdse_encode(Xf_train_rel, protos_f)
        Xf_test_p, _, _ = pgdse_encode(Xf_test_rel, protos_f, scaler=scaler_f, fit_scaler=False)

        m = build_gru_cdeepnn(Xf_train_p.shape[1], n_classes)
        m.fit(to_seq(Xf_train_p), y[tr_idx], epochs=20, batch_size=batch_size, verbose=0)
        preds = np.argmax(m.predict(to_seq(Xf_test_p), verbose=0), axis=1)
        fold_acc = accuracy_score(y[te_idx], preds)
        fold_scores.append(fold_acc)
        print(f"Fold {fold}: accuracy = {fold_acc:.4f}")

    ci = stats.t.interval(0.95, len(fold_scores) - 1, loc=np.mean(fold_scores), scale=stats.sem(fold_scores))
    print(f"\n5-Fold CV Accuracy: mean={np.mean(fold_scores):.4f}, std={np.std(fold_scores):.4f}")
    print(f"95% Confidence Interval: ({ci[0]:.4f}, {ci[1]:.4f})")

    # ---- 4.5 Ablation study ----
    print("\nAblation study (subset of training data for speed):")
    n_sub = min(2000, X.shape[0])
    idx_sub = np.random.RandomState(RANDOM_STATE).choice(X.shape[0], n_sub, replace=False)
    X_sub, y_sub = X[idx_sub], y[idx_sub]

    ablation_results = {
        "Full ELG-GCF (LLM-SSRM + PGDSE)": train_variant(X_sub, y_sub, n_classes, True, True),
        "Without LLM-SSRM (raw symptoms only)": train_variant(X_sub, y_sub, n_classes, False, True),
        "Without PGDSE (no prototype similarity)": train_variant(X_sub, y_sub, n_classes, True, False),
        "Without LLM-SSRM and PGDSE (baseline)": train_variant(X_sub, y_sub, n_classes, False, False),
    }
    for name, a in ablation_results.items():
        print(f"  {name}: accuracy = {a:.4f}")

    return {
        "model": model, "history": history.history,
        "test_accuracy": acc, "precision": prec, "recall": rec, "f1": f1,
        "cv_scores": fold_scores, "ci_95": ci,
        "ablation_results": ablation_results,
        "S": S, "prototypes": prototypes, "label_encoder": le,
        "symptom_cols": symptom_cols,
    }


if __name__ == "__main__":
    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(
            f"'{CSV_PATH}' not found. Download the Kaggle 'Vector Borne Disease Prediction' "
            f"dataset (trainn.csv) and place it alongside this script, or update CSV_PATH."
        )
    results = run_pipeline(csv_path=CSV_PATH, epochs=EPOCHS, batch_size=BATCH_SIZE, use_llm_api=USE_LLM_API)