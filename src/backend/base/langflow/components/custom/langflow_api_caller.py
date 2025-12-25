"""LangFlow API Caller component for Langflow."""

import time

import requests
from loguru import logger

from langflow.custom import Component
from langflow.io import HandleInput, MessageTextInput, Output, SecretStrInput
from langflow.schema import Data


class LangFlowApiCaller(Component):
    """Call another LangFlow instance via API.

    Use this component to send messages to another LangFlow flow,
    typically for human takeover scenarios.

    Typical flow:
    HTTP Endpoint → Google Chat Message Parser → LangFlow API Caller

    This sends the operator's message to LangFlow 1, which then
    processes it and sends the response back to the customer.
    """

    display_name = "LangFlow API Caller"
    description = "Send messages to another LangFlow flow via API"
    icon = "Send"

    inputs = [
        HandleInput(
            name="message_input",
            display_name="Message",
            info="Message containing text and session_id",
            input_types=["Message"],
            required=True,
        ),
        MessageTextInput(
            name="langflow_url",
            display_name="LangFlow URL",
            info="Base URL of the target LangFlow instance (e.g., http://localhost:7861)",
            required=True,
        ),
        MessageTextInput(
            name="target_flow_id",
            display_name="Target Flow ID",
            info="Target flow ID to call",
            required=True,
        ),
        SecretStrInput(
            name="api_key",
            display_name="API Key",
            info="API key for authentication (leave empty if PUBLIC flow)",
            required=False,
        ),
    ]

    outputs = [
        Output(display_name="Result", name="result", method="call_langflow_api"),
    ]

    def call_langflow_api(self) -> Data:
        """Call LangFlow API with the message."""
        logger.info("=== LangFlow API Caller: Starting ===")
        start_time = time.time()

        if not self.message_input:
            logger.warning("No message provided")
            return Data(data={"success": False, "error": "No message provided"})

        # Get message text and session_id
        message_text = self.message_input.text if hasattr(self.message_input, "text") else ""
        session_id = getattr(self.message_input, "session_id", "")

        if not message_text:
            logger.warning("No message text found")
            return Data(data={"success": False, "error": "No message text found"})

        # Build API request
        url = f"{self.langflow_url.strip().rstrip('/')}/api/v1/run/{self.target_flow_id}"

        headers = {
            "Content-Type": "application/json",
        }

        # Add API key if provided
        if self.api_key and self.api_key.strip():
            headers["x-api-key"] = self.api_key.strip()
            # Debug: Show first/last 4 chars of API key
            key_preview = f"{self.api_key[:4]}...{self.api_key[-4:]}" if len(self.api_key) > 8 else "***"
            logger.info(f"Using API key: {key_preview}")
        else:
            logger.info("No API key provided (PUBLIC flow expected)")

        payload = {
            "input_value": message_text,
            "output_type": "chat",
            "input_type": "chat",
        }

        # Add session_id if available
        if session_id:
            payload["session_id"] = session_id

        logger.info(f"Calling LangFlow API: {url}")
        logger.info(f"Payload: {payload}")
        logger.info(f"Headers: {list(headers.keys())}")

        try:
            response = requests.post(
                url,
                headers=headers,
                json=payload,
                timeout=30,
            )
            response.raise_for_status()
            result = response.json()

            logger.info(f"API call successful in {time.time() - start_time:.2f}s")
            logger.info(f"Response: {result}")

            return Data(
                data={
                    "success": True,
                    "response": result,
                    "session_id": session_id,
                }
            )

        except requests.exceptions.RequestException as e:
            logger.error(f"LangFlow API error: {e}")
            if hasattr(e, "response") and e.response is not None:
                logger.error(f"Response: {e.response.text}")

            return Data(
                data={
                    "success": False,
                    "error": str(e),
                    "session_id": session_id,
                }
            )
