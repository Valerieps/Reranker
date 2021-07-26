from reranker import RerankerForInference
rk = RerankerForInference.from_pretrained("Luyu/bert-base-mdoc-bm25")  # load checkpoint

inputs = rk.tokenize('weather in new york', 'it is cold today in new york', return_tensors='pt')
score = rk(inputs).logits