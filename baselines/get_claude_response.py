import pandas as pd
import anthropic
import time
import ast
import re

try:
    from .utils.baseline_utils import *
except ImportError:
    from utils.baseline_utils import *


def generate_answers(images, questions, client, logger, model_name):
    """Query Claude with the images and questions to get answers"""
    image_contents = [encode_image(img) for img in images]

    prompt = f"""
    Here is a sequence of 5 frames extracted from a video. The frames are consecutive frames from a video and thus contain temporal information. The camera is positioned at the center in first-person view. 
    Your task is to:
    1. Analyze the images provided and answer the given question.
    2. Answer each question accurately based on the images.
    3. Provide concise and clear answers.
    Questions:
    {questions}
    Please provide your answers in a list format, where each answer corresponds to the question in the same order. The answer should be either Yes or No.
    Your response should be a Python list of strings, each string being an answer. For example:
    ["Answer to question 1", "Answer to question 2", "Answer to question 3"]
    The example of value would be: ["Yes", "No", "Yes"]
    Respond with only the Python list of answers, no additional text.
    """

    logger.info(f"Sending prompt to Claude with {len(questions)} questions: {questions}")

    max_retries = 3
    retries = 0

    while retries < max_retries:
        try:
            message = client.messages.create(
                model=model_name,
                max_tokens=2000,
                temperature=0,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            *[{"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": img}}
                              for img in image_contents]
                        ]
                    }
                ]
            )

            content = message.content[0].text
            logger.info(f"Claude's raw response: {content}")

            # Try to evaluate as a Python list
            try:
                # First, try direct eval
                answers = ast.literal_eval(content)
                if isinstance(answers, list) and all(isinstance(a, str) for a in answers):
                    logger.info(f"Successfully parsed answers: {answers}")
                    return answers
                else:
                    raise ValueError("Not a list of strings")
            except:
                # If that fails, try regex to extract a list pattern
                list_pattern = r'\[(.*?)\]'
                match = re.search(list_pattern, content, re.DOTALL)
                if match:
                    items = match.group(1).split(',')
                    answers = [item.strip().strip('"\'') for item in items]
                    if len(answers) == len(questions):
                        logger.info(f"Extracted answers using regex: {answers}")
                        return answers

                logger.warning(f"Retry {retries + 1}: Could not parse Claude's response.")
                retries += 1
                time.sleep(2)  # Wait a bit before retrying

        except anthropic.APIError as e:
            logger.error(f"Anthropic API Error: {str(e)}")
            retries += 1
            time.sleep(5)  # Longer wait for API errors

    # If we've exhausted retries, return empty list
    logger.error("Failed to get valid response after maximum retries")
    return []


def get_answers_claude3(csv_path, frames_dir, output_csv_path, log_dir, model_name):
    """Process all questions in the CSV file"""
    # Set up logging
    logger = setup_logging(log_dir, model_name)
    logger.info(f"Starting processing of questions from {csv_path}")

    # Read the CSV file
    df = pd.read_csv(csv_path)
    logger.info(f"Loaded CSV with {len(df)} rows")

    # Initialize Anthropic client
    client = anthropic.Anthropic()

    # Add Claude column if it doesn't exist
    if model_name not in df.columns:
        df[model_name] = ""  # np.nan
        logger.info(f"Added '{model_name}' column to DataFrame")

    # Group by Video and Frame
    groups = df.groupby(['Video', 'Frame'], dropna=False)
    logger.info(f"Found {len(groups)} unique video/frame groups")

    for (vid, frame_range), group in groups:
        video = f"{int(vid):04d}"
        if pd.isna(video) or pd.isna(frame_range):
            continue

        logger.info(f"Processing video {video}, frames {frame_range}")

        # Get the questions for this frame range
        questions = group['Questions'].tolist()
        logger.info(f"Questions: {questions}")

        # Get frame paths
        frame_paths = get_frames_for_range(video, frame_range, frames_dir, logger=logger)

        # print(vid, frame_range, frame_paths)
        # continue

        if not frame_paths or len(frame_paths) == 0:
            logger.error(f"No frames found for video {video}, frame range {frame_range}")
            continue

        logger.info(f"Using {len(frame_paths)} frames: {[os.path.basename(p) for p in frame_paths]}")

        # Generate answers
        answers = generate_answers(frame_paths, questions, client, logger, model_name)

        if not answers or len(answers) != len(questions):
            logger.error(f"Error: Did not get valid answers for video {video}, frame range {frame_range}")
            continue

        # Convert to binary
        binary_answers = convert_to_binary(answers, logger)

        # Update the DataFrame
        idx = group.index
        df.loc[idx, model_name] = [str(int(b_a)) for b_a in binary_answers]

        # Save after each group to preserve progress
        df.to_csv(output_csv_path, index=False)
        logger.info(f"Saved progress to {output_csv_path}")

        logger.info(f"Completed {video} {frame_range}. Answers: {answers}")
        logger.info(f"Binary: {binary_answers}")

        # Wait a bit to avoid rate limits
        time.sleep(1)

    # Final save
    df['Video'] = [f"{int(vd):04d}" for vd in df['Video']]
    df.to_csv(output_csv_path, index=False)
    logger.info(f"All processing complete. Results saved to {output_csv_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Process video frame questions with Claude")
    parser.add_argument("--csv", required=True, help="Path to input CSV file with questions")
    parser.add_argument("--frames_dir", required=True, help="Directory containing video frames")
    parser.add_argument("--output", required=True, help="Path to output CSV file")
    parser.add_argument("--log_dir", required=True, help="Directory for saving the logs")

    args = parser.parse_args()

    MODEL_NAME = "claude-3-5-sonnet-20240620"
    # MODEL_NAME = "claude-3-5-sonnet-20241022"
    # MODEL_NAME = "claude-3-7-sonnet-20250219"

    get_answers_claude3(args.csv, args.frames_dir, args.output, args.log_dir, MODEL_NAME)
