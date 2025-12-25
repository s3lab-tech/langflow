"""Google Chat Message Parser component for Langflow."""

from loguru import logger

from langflow.custom import Component
from langflow.io import HandleInput, Output
from langflow.schema import Data


class GoogleChatMessageParser(Component):
    """Parse incoming Google Chat message events.

    Use this component after HTTP Endpoint to extract message details
    from Google Chat webhook payloads.

    Typical flow:
    HTTP Endpoint → Google Chat Message Parser → LangFlow 1 API Caller

    Extracts:
    - message_text: The operator's message
    - thread_name: Full thread name (spaces/xxx/threads/yyy)
    - thread_key: threadKey if available (matches session_id)
    - sender_name: Display name of the sender
    - sender_email: Email of the sender
    - space_name: Space name where message was sent
    """

    display_name = "Google Chat Message Parser"
    description = "Parse Google Chat message events from webhook"
    icon = "MessageSquare"

    inputs = [
        HandleInput(
            name="data_input",
            display_name="Data",
            info="Data from HTTP Endpoint (Google Chat webhook payload)",
            input_types=["Data"],
            required=True,
        ),
    ]

    outputs = [
        Output(display_name="Parsed Data", name="parsed_data", method="parse_message"),
    ]

    def parse_message(self) -> Data:
        """Parse Google Chat message event."""
        logger.info("=== Google Chat Message Parser: Starting ===")

        if not self.data_input:
            logger.warning("No data provided")
            return Data(data={})

        # Get raw data
        raw_data = self.data_input.data if hasattr(self.data_input, "data") else {}

        # Parse Google Chat event
        event_type = raw_data.get("type", "")
        logger.info(f"Event type: {event_type}")

        # Only process MESSAGE events
        if event_type != "MESSAGE":
            logger.warning(f"Skipping non-MESSAGE event: {event_type}")
            return Data(
                data={
                    "skip": True,
                    "event_type": event_type,
                    "message": f"Event type {event_type} is not supported",
                }
            )

        # Extract message data
        message = raw_data.get("message", {})
        thread = message.get("thread", {})
        sender = message.get("sender", {})
        space = raw_data.get("space", {})

        # Extract relevant fields
        message_text = message.get("text", "").strip()
        thread_name = thread.get("name", "")
        thread_key = thread.get("threadKey", "")  # This matches session_id
        sender_name = sender.get("displayName", "Unknown")
        sender_email = sender.get("email", "")
        space_name = space.get("name", "")

        # Note: We don't check for mentions here
        # Human Takeover Checker will check if message starts with @manager-ai-dev-chat

        # Extract session_id from thread_key or thread_name
        session_id = ""
        if thread_key:
            # If thread was created with threadKey, use it as session_id
            session_id = thread_key
        elif thread_name:
            # Try to extract from thread_name (if we need to reverse-lookup)
            # For now, we'll leave it empty and rely on thread_name for lookup
            pass

        # Build parsed data
        parsed_data = {
            "message_text": message_text,
            "thread_name": thread_name,
            "thread_key": thread_key,
            "session_id": session_id,
            "sender_name": sender_name,
            "sender_email": sender_email,
            "space_name": space_name,
            "skip": False,
        }

        logger.info(f"return: {parsed_data}")

        logger.info("=== Google Chat Message Parser: Complete ===")
        return Data(data=parsed_data)
