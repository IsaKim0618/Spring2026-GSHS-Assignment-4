
import math

def tokenize(text: str) -> list[str]:
    return text.lower().split()

def compute_tf(document: str) -> dict:
    tf = {}
    tokens = tokenize(document)
    total = len(tokens)
    if total == 0:
        return tf

    # count words
    counts = {}
    for w in tokens:
        counts[w] = counts.get(w, 0) + 1

    # TF(w) = count(w) / total_words
    for w, c in counts.items():
        tf[w] = c / total

    return tf

def compute_idf(docs: list[str]) -> dict:
    idf = {}
    N = len(docs)
    all_words = set()

    for doc in docs:
        all_words.update(tokenize(doc))

    # document frequency: in how many docs does each word appear?
    df = {w: 0 for w in all_words}
    for doc in docs:
        unique_words = set(tokenize(doc))
        for w in unique_words:
            df[w] += 1

    # IDF(w) = log(N / df(w))
    for w in all_words:
        idf[w] = math.log(N / df[w]) if df[w] > 0 else 0.0

    return idf

def compute_tf_idf(document: str, idf: dict) -> dict:
    tf_idf = {}
    tf = compute_tf(document)

    # TF-IDF(w) = TF(w) * IDF(w)
    for w, tf_val in tf.items():
        tf_idf[w] = tf_val * idf.get(w, 0.0)

    return tf_idf
