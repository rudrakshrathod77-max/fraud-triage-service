import json

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

with open("fraud_policies.json") as f:
    POLICIES = json.load(f)

_vectorizer = TfidfVectorizer()
_matrix = _vectorizer.fit_transform([p["text"] for p in POLICIES])

# Highest out-of-scope score is 0.4251, lowest in-scope score is 0.4862.
MIN_SCORE = 0.45


def retrieve(query):
    sims = cosine_similarity(_vectorizer.transform([query]), _matrix)[0]
    i = int(sims.argmax())
    return POLICIES[i]["policy_id"], POLICIES[i]["text"], float(sims[i])


if __name__ == "__main__":
    with open("retrieval_queries.json") as f:
        queries = json.load(f)

    hits, in_scores, out_scores = 0, [], []
    print(f"{'query':<68} {'expected':<9} {'retrieved':<10} score")
    for q in queries:
        pid, _, score = retrieve(q["query"])
        exp = q["expected_policy"]
        print(f"{q['query']:<68} {str(exp):<9} {pid:<10} {score:.4f}")
        if exp is None:
            out_scores.append(score)
        else:
            in_scores.append(score)
            hits += pid == exp

    print(f"\nin-scope top-1 hits: {hits} of {len(in_scores)}")
    print(f"highest out-of-scope score: {max(out_scores):.4f}")
    print(f"lowest in-scope score:      {min(in_scores):.4f}")
    print(f"MIN_SCORE chosen: {MIN_SCORE}")