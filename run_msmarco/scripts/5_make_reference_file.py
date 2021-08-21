import io
from tqdm import tqdm

path = "D:\\DOCUMENTOS\\ufmg\\tcc\\TCC_RERANKER\\data\\dev.d100.tsv"

new_lines = list()


with io.open(path, "r", encoding='utf8') as f:
    for line in tqdm(f):
        fields = line.split("\t")
        query_id = fields[0]
        doc_id = fields[2]
        new_line = f"{query_id}\t0\t{doc_id}\n"
        new_lines.append(new_line)
print(len(new_lines))

output = "D:\\DOCUMENTOS\\ufmg\\tcc\\TCC_RERANKER\\data_train\\reference_file.tsv"
with io.open(output,'w',encoding='utf8') as f:
    for line in new_lines:
        f.write(line)