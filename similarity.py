"""
Document Similarity using a Vector Space Model (TF-IDF + Cosine Similarity)

Pipeline:
  1. Load documents (.txt, .pdf, .docx) from a folder
  2. Preprocess text (lowercase, remove punctuation/numbers, remove stop words)
  3. Build vocabulary
  4. Build TF-IDF vectors (implemented manually with numpy)
  5. Compute cosine similarity between every pair of documents
  6. Verify results against scikit-learn
  7. Save similarity matrix (CSV), heatmap (PNG) and a ranked list of pairs

Usage:
    python similarity.py                  # reads documents from ./docs
    python similarity.py --folder mydocs  # reads from another folder
"""

import argparse
import math
import os
import re
from collections import Counter

import matplotlib

matplotlib.use("Agg")  # works without a display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from docx import Document
from pypdf import PdfReader
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

SUPPORTED = (".txt", ".pdf", ".docx")


# --------------------------------------------------------------------------
# STEP 1: Load documents
# --------------------------------------------------------------------------
def read_txt(path):
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def read_pdf(path):
    reader = PdfReader(path)
    pages = [(page.extract_text() or "") for page in reader.pages]
    return "\n".join(pages)


def read_docx(path):
    doc = Document(path)
    return "\n".join(p.text for p in doc.paragraphs)


def load_documents(folder):
    """Return (names, raw_texts) for all supported files in the folder."""
    names, texts = [], []
    for filename in sorted(os.listdir(folder)):
        path = os.path.join(folder, filename)
        ext = os.path.splitext(filename)[1].lower()
        if ext not in SUPPORTED:
            if ext == ".doc":
                print(f"[skip] {filename}: .doc not supported, please save it as .docx")
            continue
        try:
            if ext == ".txt":
                text = read_txt(path)
            elif ext == ".pdf":
                text = read_pdf(path)
            else:
                text = read_docx(path)
        except Exception as e:
            print(f"[skip] {filename}: could not read ({e})")
            continue

        if not text.strip():
            print(f"[skip] {filename}: no extractable text (scanned PDF?)")
            continue
        names.append(filename)
        texts.append(text)
        print(f"[ok]   {filename}: {len(text.split())} words")
    return names, texts


# --------------------------------------------------------------------------
# STEP 2: Preprocess
# --------------------------------------------------------------------------
def preprocess(text):
    """Lowercase, keep alphabetic tokens (2+ letters), remove stop words."""
    text = text.lower()
    tokens = re.findall(r"[a-z]{2,}", text)
    return [t for t in tokens if t not in ENGLISH_STOP_WORDS]


# --------------------------------------------------------------------------
# STEP 3: Vocabulary
# --------------------------------------------------------------------------
def build_vocabulary(tokenized_docs):
    vocab = sorted({tok for doc in tokenized_docs for tok in doc})
    return vocab, {word: i for i, word in enumerate(vocab)}


# --------------------------------------------------------------------------
# STEP 4: TF-IDF vectors (manual)
# --------------------------------------------------------------------------
def compute_tf(tokenized_docs, word_index):
    """Term frequency matrix: rows = documents, columns = vocabulary words."""
    tf = np.zeros((len(tokenized_docs), len(word_index)))
    for i, doc in enumerate(tokenized_docs):
        for word, count in Counter(doc).items():
            tf[i, word_index[word]] = count
    return tf


def compute_idf(tf):
    """Smoothed inverse document frequency: ln((1 + N) / (1 + df)) + 1"""
    n_docs = tf.shape[0]
    df = (tf > 0).sum(axis=0)  # number of documents containing each word
    return np.log((1 + n_docs) / (1 + df)) + 1


def compute_tfidf(tf, idf):
    tfidf = tf * idf
    norms = np.linalg.norm(tfidf, axis=1, keepdims=True)
    norms[norms == 0] = 1  # avoid dividing by zero
    return tfidf / norms  # L2-normalise each document vector


# --------------------------------------------------------------------------
# STEP 5: Cosine similarity (manual)
# --------------------------------------------------------------------------
def cosine_sim(a, b):
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denom) if denom else 0.0


def similarity_matrix(vectors):
    n = vectors.shape[0]
    sim = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            sim[i, j] = cosine_sim(vectors[i], vectors[j])
    return sim


# --------------------------------------------------------------------------
# Outputs
# --------------------------------------------------------------------------
def short_name(name, maxlen=22):
    base = os.path.splitext(name)[0]
    return base if len(base) <= maxlen else base[: maxlen - 3] + "..."


def save_heatmap(sim, names, path):
    labels = [short_name(n) for n in names]
    fig, ax = plt.subplots(figsize=(max(6, len(names) * 0.9), max(5, len(names) * 0.8)))
    im = ax.imshow(sim, cmap="YlGnBu", vmin=0, vmax=1)
    ax.set_xticks(range(len(names)))
    ax.set_yticks(range(len(names)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    for i in range(len(names)):
        for j in range(len(names)):
            color = "white" if sim[i, j] > 0.6 else "black"
            ax.text(j, i, f"{sim[i, j]:.2f}", ha="center", va="center", color=color, fontsize=8)
    ax.set_title("Cosine Similarity Between Documents (TF-IDF)")
    fig.colorbar(im, ax=ax, label="Similarity")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def ranked_pairs(sim, names):
    rows = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            rows.append((names[i], names[j], sim[i, j]))
    rows.sort(key=lambda r: r[2], reverse=True)
    return pd.DataFrame(rows, columns=["Document 1", "Document 2", "Similarity"])


def top_terms(vectors, vocab, k=5):
    """Highest-weighted TF-IDF words for each document (nice for the report)."""
    result = []
    for row in vectors:
        idx = np.argsort(row)[::-1][:k]
        result.append([vocab[i] for i in idx if row[i] > 0])
    return result


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Document similarity (TF-IDF + cosine)")
    parser.add_argument("--folder", default="docs", help="folder containing documents")
    parser.add_argument("--out", default="output", help="folder for results")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", None)

    # 1. Load
    print("\n=== Loading documents ===")
    names, raw_texts = load_documents(args.folder)
    if len(names) < 2:
        raise SystemExit("Need at least 2 readable documents in the folder.")

    # 2. Preprocess
    tokenized = [preprocess(t) for t in raw_texts]
    print("\n=== Preprocessing ===")
    for n, toks in zip(names, tokenized):
        print(f"{n}: {len(toks)} tokens after cleaning")

    # 3. Vocabulary
    vocab, word_index = build_vocabulary(tokenized)
    print(f"\nVocabulary size: {len(vocab)} unique words")

    # 4. TF-IDF
    tf = compute_tf(tokenized, word_index)
    idf = compute_idf(tf)
    tfidf = compute_tfidf(tf, idf)
    print(f"TF-IDF matrix shape: {tfidf.shape} (documents x words)")

    print("\n=== Top TF-IDF terms per document ===")
    for n, terms in zip(names, top_terms(tfidf, vocab)):
        print(f"{n}: {', '.join(terms)}")

    # 5. Cosine similarity
    sim = similarity_matrix(tfidf)
    labels = [short_name(n) for n in names]
    sim_df = pd.DataFrame(sim, index=labels, columns=labels).round(3)
    print("\n=== Similarity matrix ===")
    print(sim_df)

    # 6. Verify against scikit-learn
    vec = TfidfVectorizer(stop_words="english", lowercase=True, token_pattern=r"(?u)\b[a-zA-Z]{2,}\b")
    sk_sim = cosine_similarity(vec.fit_transform(raw_texts))
    max_diff = np.abs(sim - sk_sim).max()
    print(f"\n=== Verification vs scikit-learn ===\nMax difference: {max_diff:.10f}")
    print("Match!" if max_diff < 1e-6 else "Results differ slightly (check preprocessing).")

    # 7. Save results
    sim_df.to_csv(os.path.join(args.out, "similarity_matrix.csv"))
    pairs = ranked_pairs(sim, names)
    pairs.to_csv(os.path.join(args.out, "ranked_pairs.csv"), index=False)
    save_heatmap(sim, names, os.path.join(args.out, "similarity_heatmap.png"))

    print("\n=== Most similar pairs ===")
    print(pairs.head(5).round(3).to_string(index=False))
    print("\n=== Least similar pairs ===")
    print(pairs.tail(3).round(3).to_string(index=False))
    print(f"\nSaved results to '{args.out}/'")


if __name__ == "__main__":
    main()
