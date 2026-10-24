"""Gradio web interface for the Icebreaker Bot."""

import sys
import logging
import uuid

import gradio as gr

from modules.data_extraction import extract_linkedin_profile
from modules.data_processing import (
    split_profile_data,
    create_vector_database,
    verify_embeddings,
)
from modules.llm_interface import change_llm_model
from modules.query_engine import (
    generate_initial_facts,
    answer_user_query,
)
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

# Dictionary to store active conversations.
active_indices = {}


def process_profile(linkedin_url, api_key, use_mock, selected_model):
    """Process a LinkedIn profile and generate initial facts.

    Args:
        linkedin_url: LinkedIn profile URL to process.
        api_key: ProxyCurl API key.
        use_mock: Whether to use mock data.
        selected_model: LLM model to use.

    Returns:
        Initial facts or an error message, and a session ID or None.
    """
    try:
        if selected_model != config.LLM_MODEL_ID:
            change_llm_model(selected_model)

        if use_mock and not linkedin_url:
            linkedin_url = "https://www.linkedin.com/in/leonkatsnelson/"

        profile_data = extract_linkedin_profile(
            linkedin_url,
            api_key if not use_mock else None,
            mock=use_mock,
        )

        if not profile_data:
            return (
                "Failed to retrieve profile data. "
                "Check the terminal logs for details.",
                None,
            )

        nodes = split_profile_data(profile_data)

        if not nodes:
            return "Failed to process profile data into nodes.", None

        index = create_vector_database(nodes)

        if index is None:
            return "Failed to create vector database.", None

        if not verify_embeddings(index):
            logger.warning("Some embeddings may be missing or invalid")

        facts = generate_initial_facts(index)

        if facts == "Failed to generate initial facts.":
            return facts, None

        session_id = str(uuid.uuid4())
        active_indices[session_id] = index

        result = (
            "Profile processed successfully!\n\n"
            "Here are 3 interesting facts about this person:\n\n"
            f"{facts}"
        )

        return result, session_id

    except Exception as e:
        logger.exception("Error in process_profile")
        return f"Error: {e}", None


def chat_with_profile(session_id, user_query, chat_history):
    """Answer a question about a processed profile.

    Args:
        session_id: Session ID for this conversation.
        user_query: User's question.
        chat_history: Existing chat history.

    Returns:
        Updated chat history.
    """
    chat_history = list(chat_history or [])
    user_query = (user_query or "").strip()

    if not user_query:
        return chat_history

    if not session_id:
        return chat_history + [
            [
                user_query,
                "No profile loaded. Please process a LinkedIn profile first.",
            ]
        ]

    if session_id not in active_indices:
        return chat_history + [
            [
                user_query,
                "Session expired. Please process the LinkedIn profile again.",
            ]
        ]

    try:
        index = active_indices[session_id]
        response = answer_user_query(index, user_query)

        # The query function returns an object on success or a string on failure.
        answer = getattr(response, "response", response)

        if answer is None:
            answer = "No answer was returned. Please try another question."

        return chat_history + [[user_query, str(answer)]]

    except Exception as e:
        logger.exception("Error in chat_with_profile")
        return chat_history + [[user_query, f"Error: {e}"]]


def create_gradio_interface():
    """Create the Gradio interface for the Icebreaker Bot."""
    available_models = [
        "ibm/granite-4-h-small",
        "meta-llama/llama-3-2-11b-vision-instruct",
    ]

    # Ensure the configured default is included in the dropdown.
    if config.LLM_MODEL_ID not in available_models:
        available_models.append(config.LLM_MODEL_ID)

    with gr.Blocks(title="LinkedIn Icebreaker Bot") as demo:
        gr.Markdown("# LinkedIn Icebreaker Bot")
        gr.Markdown(
            "Generate personalized icebreakers and chat about "
            "LinkedIn profiles."
        )

        with gr.Tab("Process LinkedIn Profile"):
            with gr.Row():
                with gr.Column():
                    linkedin_url = gr.Textbox(
                        label="LinkedIn Profile URL",
                        placeholder="https://www.linkedin.com/in/username/",
                    )

                    api_key = gr.Textbox(
                        label="ProxyCurl API Key (leave empty for mock data)",
                        placeholder="Your ProxyCurl API Key",
                        type="password",
                        value=config.PROXYCURL_API_KEY,
                    )

                    use_mock = gr.Checkbox(
                        label="Use Mock Data",
                        value=True,
                    )

                    model_dropdown = gr.Dropdown(
                        choices=available_models,
                        label="Select LLM Model",
                        value=config.LLM_MODEL_ID,
                    )

                    process_btn = gr.Button("Process Profile")

                with gr.Column():
                    result_text = gr.Textbox(
                        label="Initial Facts",
                        lines=10,
                    )

                    session_id = gr.Textbox(
                        label="Session ID",
                        visible=False,
                    )

            process_btn.click(
                fn=process_profile,
                inputs=[
                    linkedin_url,
                    api_key,
                    use_mock,
                    model_dropdown,
                ],
                outputs=[result_text, session_id],
            )

        with gr.Tab("Chat"):
            gr.Markdown("Chat with the processed LinkedIn profile.")

            chatbot = gr.Chatbot(height=500)

            chat_input = gr.Textbox(
                label="Ask a question about the profile",
                placeholder="What is this person's current job title?",
            )

            chat_btn = gr.Button("Send")

            chat_btn.click(
                fn=chat_with_profile,
                inputs=[session_id, chat_input, chatbot],
                outputs=[chatbot],
            )

            chat_input.submit(
                fn=chat_with_profile,
                inputs=[session_id, chat_input, chatbot],
                outputs=[chatbot],
            )

    return demo


if __name__ == "__main__":
    demo = create_gradio_interface()
    demo.launch(
        server_name="0.0.0.0",
        server_port=5000,
    )