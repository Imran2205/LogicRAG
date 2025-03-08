import os
import pandas as pd
import requests
import time
import ast
import re

try:
    from .utils.baseline_utils import *
except ImportError:
    from utils.baseline_utils import *


def generate_answers_gpt4(images, questions, logger, model_name):
    """Query GPT-4 Vision with the images and questions to get answers"""
    image_contents = [encode_image(img) for img in images]

    prompt = f"""
    Here is a sequence of 10 frames extracted from a video. The frames are consecutive frames from a video and thus contain temporal information. The camera is positioned at the center in first-person view. 
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

    logger.info(f"Sending prompt to GPT-4 Vision with {len(questions)} questions and {len(images)} frames")

    max_retries = 3
    retries = 0

    while retries < max_retries:
        try:
            payload = {
                "model": model_name,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            *[{"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img}"}} for img in
                              image_contents]
                        ]
                    }
                ],
                # "max_tokens": 2000,
            }

            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {os.environ['OPENAI_API']}"
            }

            response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)

            if response.status_code == 200:
                content = response.json()['choices'][0]['message']['content']
                logger.info(f"GPT-4's raw response: {content}")

                try:
                    # First, try direct eval
                    answers = ast.literal_eval(content)
                    if isinstance(answers, list) and all(isinstance(a, str) for a in answers):
                        logger.info(f"Successfully parsed GPT-4 answers: {answers}")
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
                            logger.info(f"Extracted GPT-4 answers using regex: {answers}")
                            return answers

                    logger.warning(f"Retry {retries + 1}: Could not parse GPT-4's response.")
            else:
                logger.error(f"GPT-4 API Error: {response.status_code}, {response.text}")

            retries += 1
            time.sleep(5)  # Wait a bit before retrying

        except Exception as e:
            logger.error(f"Error with GPT-4 request: {str(e)}")
            retries += 1
            time.sleep(5)

    # If we've exhausted retries, return empty list
    logger.error("Failed to get valid response from GPT-4 after maximum retries")
    return []


def get_answers_gpt4(csv_path, frames_dir, output_csv_path, log_dir, model_name):
    # Set up logging
    logger = setup_logging(log_dir, model_name)
    logger.info(f"Starting processing of questions from {csv_path}")

    # Read the CSV file
    df = pd.read_csv(csv_path)
    logger.info(f"Loaded CSV with {len(df)} rows")

    # Add GPT4 column if it doesn't exist
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

        # Get ALL frames for the window (instead of just 5)
        frame_paths = get_frames_for_range(video, frame_range, frames_dir, num_frames=10, logger=logger)

        if not frame_paths or len(frame_paths) == 0:
            logger.error(f"No frames found for video {video}, frame range {frame_range}")
            continue

        logger.info(f"Using {len(frame_paths)} frames: {[os.path.basename(p) for p in frame_paths]}")

        # Generate answers from GPT-4 with all frames
        gpt4_answers = generate_answers_gpt4(frame_paths, questions, logger, model_name)

        if gpt4_answers and len(gpt4_answers) == len(questions):
            # Convert to binary
            gpt4_binary = convert_to_binary(gpt4_answers, logger)

            # Update the DataFrame
            idx = group.index
            df.loc[idx, model_name] = [str(int(b_a)) for b_a in gpt4_binary]

            logger.info(f"GPT-4 answers: {gpt4_answers}")
            logger.info(f"GPT-4 binary: {gpt4_binary}")

            # Save after each group to preserve progress
            df.to_csv(output_csv_path, index=False)
            logger.info(f"Saved progress to {output_csv_path}")
        else:
            logger.error(f"Error: Did not get valid answers from GPT-4 for video {video}, frame range {frame_range}")

        # Wait a bit to avoid rate limits
        time.sleep(1)

    # Final save
    df['Video'] = [f"{int(vd):04d}" if not pd.isna(vd) else vd for vd in df['Video']]
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

    # MODEL_NAME = "gpt-4-turbo" # gpt-4-turbo-2024-04-09
    MODEL_NAME = "gpt-4o"  # gpt-4o-2024-08-06
    # MODEL_NAME = "gpt-4.5-preview" # gpt-4.5-preview-2025-02-27
    # MODEL_NAME = "o1"  # o1-2024-12-17

    get_answers_gpt4(args.csv, args.frames_dir, args.output, args.log_dir, MODEL_NAME)
