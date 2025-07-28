from sentence_transformers import CrossEncoder

cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

def rerank_with_cross_encoder(query: str, results: list, top_n: int = 3):
    pairs = [(query, r.payload.get("text", "")) for r in results]
    scores = cross_encoder.predict(pairs)
    for i, r in enumerate(results):
        r.score = float(scores[i])
    reranked = sorted(results, key=lambda x: x.score, reverse=True)
    return reranked[:top_n]
