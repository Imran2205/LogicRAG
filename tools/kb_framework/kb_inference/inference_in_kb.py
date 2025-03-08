import pandas as pd
import logging
try:
    from .utils.inference_engine import query_kb
    from .utils.eval_utils import get_scores
except ImportError:
    from utils.inference_engine import query_kb
    from utils.eval_utils import get_scores


def lrag_inference(
        questions_csv_path,
        fol_mapping_csv_path,
        base_kb_path,
        base_tracker_out_path,
        logger=None):
    if not logger:
        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger(__name__)

    questions_df = pd.read_csv(questions_csv_path)
    fol_mapping_df = pd.read_csv(fol_mapping_csv_path)

    # Create a mapping of questions to FOL queries
    fol_dict = dict(zip(fol_mapping_df['Question'], fol_mapping_df['FOL Query']))

    frame_groups = []

    # tot_q = 0
    # tot_cor = 0
    seq_all_ans = []
    seq_all_annos = []

    # Group by Video and Frame
    groups = questions_df.groupby(['Video', 'Frame'], dropna=False)

    # print(frame_groups)
    # Generate function calls for each frame sequence
    for (vid, frame_range), group in groups:
        video = f"{int(vid):04d}"

        logger.info(f"Processing video {video}, frames {frame_range}")

        questions = group['Questions'].tolist()
        annotations = group['Human'].tolist()
        # Parse frame range
        # start_frame, end_frame = map(int, frame_seq.split('-'))
        start_frame, end_frame = frame_range.split('-')

        # Generate FOL queries list
        fol_queries = []
        for question in questions:
            if question in fol_dict:
                fol_queries.append(fol_dict[question])
            else:
                print(f"Warning: No FOL query found for question: {question}")

        # Generate paths
        kb_file_path = f"{base_kb_path}/{video}/kb_window_{start_frame}_{end_frame}.txt"
        obj_info_file_path = f"{base_tracker_out_path}/{video}/vehicles_prop.txt"

        # Generate and print the function call
        # print(f"\n# Video: {video_id}, Frame sequence: {frame_seq}")
        # print(fol_queries)

        answers = query_kb(
            kb_file_path,
            obj_info_file_path,
            fol_queries,
            logger=logger
        )

        seq_all_ans += answers
        seq_all_annos += annotations

        # print("GT:   ", annotations)
        # print("PRED: ", answers)

        # for i_ans, ans in enumerate(answers):
        #     tot_q += 1
        #     if ans == annotations[i_ans]:
        #         tot_cor += 1

        # print(f"# Questions in this sequence ({frame_seq}):")
        # for i, q in enumerate(questions):
        #     print(f"#  {i+1}. {q} (Ground truth: {annotations[i]})")
        # # print(f"kb_file_path = \"{kb_file_path}\"")
        # # print(f"obj_info_file_path = \"{obj_info_file_path}\"")
        # print("fol_queries = [")
        # for query in fol_queries:
        #     print(f"    \"{query}\",")
        # print("]")
        # print(f"ground_truth = {annotations}")
        # print(f"ans = {answers}")

    return seq_all_ans, seq_all_annos  # (tot_cor/tot_q)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Process video frame questions with Claude")
    parser.add_argument("--csv", required=True, help="Path to input CSV file with questions")
    parser.add_argument("--fol_trans_csv", required=True, help="Path to input CSV file with FOL translation")
    parser.add_argument("--kb_dir", required=True, help="Directory containing KB files")
    parser.add_argument("--tracker_dir", required=True, help="Directory containing tracker trajectories")
    parser.add_argument("--output", required=True, help="Path to output CSV file")

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    base_logger = logging.getLogger(__name__)

    df = pd.read_csv(args.csv)
    # vid_ids = [str(vid_).zfill(4) for vid_ in list(set(df['Video'].tolist()))]
    # print(vid_ids)

    # vid_ids = args.vid_ids.split(',')  # ['0000', '0001', '0002', '0003', '0004', '0005', '0008', '0010', '0012']

    all_answers, all_annotations = lrag_inference(
        args.csv, args.fol_trans_csv,
        args.kb_dir, args.tracker_dir,
        logger=base_logger
    )

    acc, f1, precision, recall = get_scores(all_answers, all_annotations)

    print(f"Overall Accuracy: {acc:.2f}, F1: {f1:.2f}, Prec: {precision:.2f}, Rec: {recall:.2f}")

    df['LogicRAG'] = all_answers
    df.to_csv(args.output, index=False)
