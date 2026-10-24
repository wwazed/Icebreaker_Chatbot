"""Main script for running the Icebreaker Bot."""

import sys
import time
import logging
import argparse

from modules.data_extraction import extract_linkedin_profile
from modules.data_processing import (
    split_profile_data,
    create_vector_database,
    verify_embeddings,
)
from modules.query_engine import (
    generate_initial_facts,
    answer_user_query,
)
from modules.llm_interface import change_llm_model
import config


# Set up logging.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(stream=sys.stdout)
    ],
)

logger = logging.getLogger(__name__)


def process_linkedin(linkedin_url, api_key=None, mock=False):
    """Process profile data and start the command-line chatbot.

    Args:
        linkedin_url: LinkedIn profile URL to process.
        api_key: ProxyCurl API key. Required if mock is False.
        mock: Whether to use the prepared mock profile data.
    """
    try:
        # Extract the profile data.
        profile_data = extract_linkedin_profile(
            linkedin_url,
            api_key,
            mock=mock,
        )

        if not profile_data:
            logger.error("Failed to retrieve profile data.")
            return

        # Split the data into nodes.
        nodes = split_profile_data(profile_data)

        if not nodes:
            logger.error("Failed to process profile data into nodes.")
            return

        # Create the vector index.
        vectordb_index = create_vector_database(nodes)

        if vectordb_index is None:
            logger.error("Failed to create vector database.")
            return

        # Verify embeddings.
        if not verify_embeddings(vectordb_index):
            logger.warning("Some embeddings may be missing or invalid.")

        # Generate and display the initial facts.
        initial_facts = generate_initial_facts(vectordb_index)

        if initial_facts == "Failed to generate initial facts.":
            logger.error(initial_facts)
            return

        print("\nHere are 3 interesting facts about this person:")
        print(initial_facts)

        # Start the chatbot interface.
        chatbot_interface(vectordb_index)

    except Exception:
        logger.exception("Error while processing the profile")


def chatbot_interface(index):
    """Provide a command-line interface for asking profile questions.

    Args:
        index: VectorStoreIndex containing the profile data.
    """
    print(
        "\nYou can now ask more in-depth questions about this person. "
        "Type 'exit', 'quit', or 'bye' to quit."
    )

    while True:
        try:
            user_query = input("You: ").strip()

            if user_query.lower() in ["exit", "quit", "bye"]:
                print("Bot: Goodbye!")
                break

            if not user_query:
                continue

            print("Bot is typing...", end="", flush=True)
            time.sleep(1)
            print("\r" + " " * 20 + "\r", end="", flush=True)

            response = answer_user_query(index, user_query)

            # Handle response objects and error strings.
            answer = getattr(response, "response", response)

            if answer is None:
                answer = "No answer was returned. Please try another question."

            print(f"Bot: {str(answer).strip()}\n")

        except (KeyboardInterrupt, EOFError):
            print("\nBot: Goodbye!")
            break

        except Exception:
            logger.exception("Error while answering the question")
            print("Bot: Something went wrong. Please try again.\n")


def main():
    """Parse command-line arguments and run the Icebreaker Bot."""
    parser = argparse.ArgumentParser(
        description="Icebreaker Bot - LinkedIn Profile Analyzer"
    )

    parser.add_argument(
        "--url",
        type=str,
        help="LinkedIn profile URL",
    )

    parser.add_argument(
        "--api-key",
        type=str,
        help="ProxyCurl API key",
    )

    parser.add_argument(
        "--mock",
        action="store_true",
        help="Use mock data instead of the API",
    )

    parser.add_argument(
        "--model",
        type=str,
        help=(
            "LLM model to use "
            '(e.g., "ibm/granite-4-h-small")'
        ),
    )

    args = parser.parse_args()

    # Skip the URL prompt when mock mode is explicitly requested.
    if args.mock:
        linkedin_url = (args.url or "").strip()
        use_mock = True
    else:
        linkedin_url = (
            args.url
            or input(
                "Enter LinkedIn profile URL "
                "(or press Enter to use mock data): "
            )
        ).strip()

        use_mock = not linkedin_url

    if args.model:
        change_llm_model(args.model)

    api_key = args.api_key or config.PROXYCURL_API_KEY

    if not use_mock and not api_key:
        api_key = input("Enter ProxyCurl API key: ").strip()

    # Use the default profile URL for mock mode.
    if use_mock and not linkedin_url:
        linkedin_url = "https://www.linkedin.com/in/leonkatsnelson/"

    process_linkedin(
        linkedin_url,
        api_key,
        mock=use_mock,
    )


if __name__ == "__main__":
    main()