"""
This module computes evaluation metrics for MSMARCO dataset on the ranking task.
Command line:
python msmarco_eval_ranking.py <path_to_reference_file> <path_to_candidate_file>

Creation Date : 06/12/2018
Last Modified : 1/21/2019
Authors : Daniel Campos <dacamp@microsoft.com>, Rutger van Haasteren <ruvanh@microsoft.com>
"""
import sys
import statistics

from collections import Counter

MaxMRRRank = 10


def load_reference_from_stream(f):
    """Load Reference reference relevant passages
    Args:f (stream): stream to load.
    Returns:qids_to_relevant_passageids (dict): dictionary mapping from query_id (int) to relevant passages (list of ints).
    """
    qids_to_relevant_passageids = {}
    for l in f:
        try:
            l = l.strip().split('\t')
            qid = int(l[0])


            if qid in qids_to_relevant_passageids:
                pass
            else:
                qids_to_relevant_passageids[qid] = []
            something = int(l[2][1:])
            qids_to_relevant_passageids[qid].append(something)

        except Exception as e:
            print(e)
            raise IOError('\"%s\" is not valid format' % l)
    return qids_to_relevant_passageids


def load_reference(path_to_reference):
    """Load Reference reference relevant passages
    Args:path_to_reference (str): path to a file to load.
    Returns:qids_to_relevant_passageids (dict): dictionary mapping from query_id (int) to relevant passages (list of ints).
    """
    with open(path_to_reference, 'r') as f:
        qids_to_relevant_passageids = load_reference_from_stream(f)
    return qids_to_relevant_passageids


def load_candidate_from_stream(f):
    """
    ACHO QUE O ERRO TA AQUI

    Load candidate data_train from a stream.
    Args:f (stream): stream to load.
    Returns:qid_to_ranked_candidate_passages (dict):
    dictionary mapping from query_id (int) to a list of 1000 passage ids(int) ranked by relevance and importance
    """
    query_to_doc_candidate = {}
    for line in f:
        try:
            line = line.strip().split('\t')
            qid = int(line[0])
            pid = int(line[1][1:]) # remove o char 'D' do inicio do documento
            rank = int(line[2])
            if qid in query_to_doc_candidate:
                pass
            else:
                # By default, all PIDs in the list of 1000 are 0. Only override those that are given
                tmp = [0] * 100
                query_to_doc_candidate[qid] = tmp
            query_to_doc_candidate[qid][rank - 1] = pid
        except:
            raise IOError('\"%s\" is not valid format' % line)
    return query_to_doc_candidate


def load_candidate(path_to_candidate):
    """Load candidate data_train from a file.
    Args:path_to_candidate (str): path to file to load.
    Returns:qid_to_ranked_candidate_passages (dict): dictionary mapping from query_id (int) to a list of 1000 passage ids(int) ranked by relevance and importance
    """

    with open(path_to_candidate, 'r') as f:
        query_to_doc_candidate = load_candidate_from_stream(f)
    return query_to_doc_candidate


def quality_checks_qids(qids_to_relevant_passageids, qids_to_ranked_candidate_passages):
    """Perform quality checks on the dictionaries

    Args:
    p_qids_to_relevant_passageids (dict): dictionary of query-passage mapping
        Dict as read in with load_reference or load_reference_from_stream
    p_qids_to_ranked_candidate_passages (dict): dictionary of query-passage candidates
    Returns:
        bool,str: Boolean whether allowed, message to be shown in case of a problem
    """
    message = ''
    allowed = True

    # Create sets of the QIDs for the submitted and reference queries
    candidate_set = set(qids_to_ranked_candidate_passages.keys())
    ref_set = set(qids_to_relevant_passageids.keys())

    # Check that we do not have multiple passages per query
    for qid in qids_to_ranked_candidate_passages:
        # Remove all zeros from the candidates
        duplicate_pids = set(
            [item for item, count in Counter(qids_to_ranked_candidate_passages[qid]).items() if count > 1])

        if len(duplicate_pids - set([0])) > 0:
            message = "Cannot rank a passage multiple times for a single query. QID={qid}, PID={pid}".format(
                qid=qid, pid=list(duplicate_pids)[0])
            allowed = False

    return allowed, message


def compute_metrics(query_to_doc_ground_truth, query_to_docs_candiadte):
    """Compute MRR metric
    Args:
    p_qids_to_relevant_passageids (dict): dictionary of query-passage mapping
        Dict as read in with load_reference or load_reference_from_stream
    p_qids_to_ranked_candidate_passages (dict): dictionary of query-passage candidates
    Returns:
        dict: dictionary of metrics {'MRR': <MRR Score>}
    """
    print("inside compute metrics")
    # print(query_to_doc_ground_truth)
    # print(len(query_to_doc_ground_truth[174249]))
    print(query_to_docs_candiadte)

    all_scores = {}
    total_MRR = 0
    qids_with_relevant_passages = 0
    ranking = []
    for query_id in query_to_docs_candiadte:
        if query_id in query_to_doc_ground_truth:
            ranking.append(0)

            # Resgata a lista de documentos corretos
            ordered_docs_ground_truth = query_to_doc_ground_truth[query_id]

            # Resgata a lista de documentos candidatos
            ordered_docs_candidate = query_to_docs_candiadte[query_id]

            #
            for idx in range(0, MaxMRRRank): # 0 a 9
                probable_relevant_doc = ordered_docs_candidate[idx]
                print(f"checking doc {probable_relevant_doc}")
                if probable_relevant_doc in ordered_docs_ground_truth:
                    real_position = ordered_docs_ground_truth.index(probable_relevant_doc) + 1
                    print(f"Doc {probable_relevant_doc} is in position {real_position}")
                    if real_position <= 10:
                        this_MRR = 1 / real_position
                        print(this_MRR)
                        ranking.pop()
                        ranking.append(idx)
                        break
                # print(f"MRR parcial: {this_MRR}")
    if len(ranking) == 0:
        raise IOError("No matching QIDs found. Are you sure you are scoring the evaluation set?")

    MRR = total_MRR / len(query_to_doc_ground_truth)
    all_scores['MRR @10'] = MRR
    all_scores['QueriesRanked'] = len(query_to_docs_candiadte)
    return all_scores


def compute_metrics_from_files(path_to_reference, path_to_candidate, perform_checks=True):
    """Compute MRR metric
    Args:
    p_path_to_reference_file (str): path to reference file.
        Reference file should contain lines in the following format:
            QUERYID\tPASSAGEID
            Where PASSAGEID is a relevant passage for a query. Note QUERYID can repeat on different lines with different PASSAGEIDs
    p_path_to_candidate_file (str): path to candidate file.
        Candidate file sould contain lines in the following format:
            QUERYID\tPASSAGEID1\tRank
            If a user wishes to use the TREC format please run the script with a -t flag at the end. If this flag is used the expected format is
            QUERYID\tITER\tDOCNO\tRANK\tSIM\tRUNID
            Where the values are separated by tabs and ranked in order of relevance
    Returns:
        dict: dictionary of metrics {'MRR': <MRR Score>}
    """

    query_to_doc_ground_truth = load_reference(path_to_reference)
    query_to_doc_candidate = load_candidate(path_to_candidate)
    if perform_checks:
        allowed, message = quality_checks_qids(query_to_doc_ground_truth, query_to_doc_candidate)
        if message != '': print(message)

    return compute_metrics(query_to_doc_ground_truth, query_to_doc_candidate)


def main():
    """Command line:
    python msmarco_eval_ranking.py <path_to_reference_file> <path_to_candidate_file>
    """

    if len(sys.argv) == 3:
        path_to_reference = sys.argv[1]
        path_to_candidate = sys.argv[2]
        metrics = compute_metrics_from_files(path_to_reference, path_to_candidate)
        print('#####################')
        for metric in sorted(metrics):
            print('{}: {}'.format(metric, metrics[metric]))
        print('#####################')

    else:
        print('Usage: msmarco_eval_ranking.py <reference ranking> <candidate ranking>')
        exit()


if __name__ == '__main__':
    main()