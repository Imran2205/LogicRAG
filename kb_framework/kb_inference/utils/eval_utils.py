from sklearn.metrics import f1_score, precision_score, recall_score, classification_report
import numpy as np


def get_scores(all_answers_, all_annotations_):
    tot_q = 0
    tot_cor = 0
    for i_ans, ans in enumerate(all_answers_):
        tot_q += 1
        if ans == all_annotations_[i_ans]:
            tot_cor += 1

    y_true = np.array(all_annotations_)
    y_pred = np.array(all_answers_)

    f1_ = f1_score(y_true, y_pred)

    precision_ = precision_score(y_true, y_pred)
    recall_ = recall_score(y_true, y_pred)

    accuracy_ = tot_cor / tot_q

    return accuracy_, f1_, precision_, recall_
