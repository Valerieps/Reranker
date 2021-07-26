#abrir o msmarco-dcos.tsv

import pandas

file = "data/msmarco-docs.tsv"

dados = pandas.read_csv(file, sep="\t", chunksize=5)
chunk = dados.get_chunk()
print(chunk)
print(type(chunk))
print(len(chunk))