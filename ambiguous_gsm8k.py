"""
Generate an ambiguous version of the GSM-8k dataset.
"""

import re
import random
import argparse
import logging
from datasets import load_dataset, Dataset
from huggingface_hub import create_repo

logging.basicConfig(level=logging.INFO)


def extract_numbers_and_context(question, num_context_words=3):
    """
    Extract numbers from the question with their surrounding context.

    Args:
        question (str): The original question text.
        num_context_words (int): Number of context words to extract before/after.

    Returns:
        list: Tuples of (number, words_before, words_after).
    """
    words = question.split()
    numbers_with_context = []
    for i, word in enumerate(words):
        if re.match(r"\d+", word):
            before = words[max(0, i - num_context_words) : i]
            after = words[i + 1 : i + 1 + num_context_words]
            numbers_with_context.append((word, before, after))
    return numbers_with_context


def generate_unique_keywords(existing_keywords, before, after):
    """
    Generate unique keywords for an obscured number, avoiding overlap with existing keyword sets.

    Args:
        existing_keywords (list): List of previously used keyword lists.
        before (list): Words before the number.
        after (list): Words after the number.

    Returns:
        list: Unique keywords for the current number.
    """
    keywords = before[-2:] + after[:2]  # Default to 2 before, 2 after
    while any(set(keywords) <= set(existing) for existing in existing_keywords):
        if before:
            keywords.insert(0, before.pop(0))
        elif after:
            keywords.append(after.pop(0))
        else:
            break
    return keywords


def load_ambiguous_gsm8k():
    """
    Generate an ambiguous version of the GSM-8k dataset with varied obscurations.

    Returns:
        list: List of dictionaries containing ambiguous questions, missing info, and answers.
    """
    dataset = load_dataset("gsm8k", "main")["train"]
    logging.info(len(dataset))
    ambiguous_dataset = []

    for item in dataset:
        question = item["question"]
        answer = item["answer"]
        numbers_with_context = extract_numbers_and_context(question)

        # Skip if no numbers are found
        if not numbers_with_context:
            continue

        # Randomly decide to obscure 1 or 2 numbers (if possible)
        num_to_obscure = random.randint(1, min(2, len(numbers_with_context)))
        numbers_to_obscure = random.sample(numbers_with_context, num_to_obscure)

        # Store missing information as a list of dictionaries
        missing_info_list = []
        existing_keywords = []
        ambiguous_question = question

        for idx, (num, before, after) in enumerate(numbers_to_obscure, start=1):
            placeholder = f"some number"
            ambiguous_question = ambiguous_question.replace(num, placeholder, 1)

            # Ensure unique keywords for each obscured number
            keywords = generate_unique_keywords(
                existing_keywords, before.copy(), after.copy()
            )
            existing_keywords.append(keywords)
            missing_info_list.append({"keywords": keywords, "value": num})

        # Add to the dataset
        ambiguous_dataset.append(
            {
                "question": ambiguous_question,
                "missing_info": missing_info_list,  # List of dicts with string keys
                "answer": answer,
            }
        )

    logging.info(f"Generated {len(ambiguous_dataset)} ambiguous questions.")

    return ambiguous_dataset


def save_to_huggingface(dataset, repo_name):
    """
    Save the ambiguous dataset to Hugging Face.

    Args:
        dataset (list): List of dictionaries with the ambiguous dataset.
        repo_name (str): Name of the Hugging Face repository (e.g., "username/repo").
    """
    # Convert to a Hugging Face Dataset object
    ambiguous_dataset = Dataset.from_list(dataset)

    # Create the repository if it doesn't exist
    create_repo(repo_name, exist_ok=True)

    # Upload the dataset
    ambiguous_dataset.push_to_hub(repo_name)
    logging.info(ambiguous_dataset)
    logging.info(f"Dataset uploaded to: https://huggingface.co/datasets/{repo_name}")


if __name__ == "__main__":
    # CLI arguments
    parser = argparse.ArgumentParser(
        description="Generate an ambiguous version of the GSM-8k dataset."
    )
    parser.add_argument(
        "--repo_name",
        type=str,
        default="Technoculture/ambiguous_gsm8k",
        help="Name of the Hugging Face repository (e.g., 'username/repo').",
    )
    args = parser.parse_args()

    # Generate the ambiguous dataset
    ambiguous_dataset = load_ambiguous_gsm8k()

    # Save to Hugging Face (replace with your details)
    repo_name = args.repo_name
    save_to_huggingface(ambiguous_dataset, repo_name)
