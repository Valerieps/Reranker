# from datasets import load_dataset
# dataset = load_dataset('csv', data_files='teste2.csv', delimiter='\t')



import pandas as pd

pd.read_csv("teste2.csv", names=None, prefix=None)



# dataset = load_dataset('text', data_files=['teste2.csv'])
# dataset = load_dataset('csv', data_files=['teste2.csv'])
#
# print(dataset)
# # print(datasets.list_datasets())
# # columns = ['did', 'url', 'title', 'body']
# train_doc_collection = load_dataset(
#     'csv',
#     data_files=["teste2.csv"],
#     # script_version="master",
#     # column_names=columns,
#     # delimiter='\t',
#     # ignore_verifications=True,
# )["train"]