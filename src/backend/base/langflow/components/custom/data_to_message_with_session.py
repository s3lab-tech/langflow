"""Data to Message with Session component for Langflow."""

from loguru import logger

from langflow.custom import Component
from langflow.io import HandleInput, MessageTextInput, Output
from langflow.schema.message import Message


class DataToMessageWithSession(Component):
    """Convert Data to Message and preserve session_id.

    Use this component to convert Data (from Google Chat Message Parser)
    to Message format while preserving the session_id.

    Typical flow:
    Google Chat Message Parser → Data to Message with Session → Firestore Session Logger
    """

    display_name = "Data to Message with Session"
    description = "Convert Data to Message and add session_id attribute"
    icon = "ArrowRightLeft"

    inputs = [
        HandleInput(
            name="data_input",
            display_name="Data",
            info="Data containing message_text and session_id",
            input_types=["Data"],
            required=True,
        ),
        MessageTextInput(
            name="text_field",
            display_name="Text Field",
            info="Field name in data to use as message text",
            value="message_text",
            advanced=True,
        ),
        MessageTextInput(
            name="session_id_field",
            display_name="Session ID Field",
            info="Field name in data to use as session_id",
            value="session_id",
            advanced=True,
        ),
        MessageTextInput(
            name="sender_field",
            display_name="Sender Field",
            info="Field name in data to use as sender name (optional)",
            value="sender_name",
            advanced=True,
        ),
    ]

    outputs = [
        Output(display_name="Message", name="message_output", method="convert_to_message"),
    ]

    def convert_to_message(self) -> Message:
        """Convert Data to Message with session_id."""
        logger.info("=== Data to Message with Session: Starting ===")

        if not self.data_input:
            logger.warning("No data provided")
            return Message(text="")

        # Get data
        data = self.data_input.data if hasattr(self.data_input, "data") else {}

        # Extract fields
        text_field = self.text_field or "message_text"
        session_id_field = self.session_id_field or "session_id"
        sender_field = self.sender_field or "sender_name"

        message_text = data.get(text_field, "")
        session_id = data.get(session_id_field, "")
        sender_name = data.get(sender_field, "")

        # Create Message with session_id
        message = Message(text=message_text)
        message.session_id = session_id

        # Add sender information if available
        if sender_name:
            message.sender = sender_name

        logger.info(f"return: {message}")

        logger.info("=== Data to Message with Session: Complete ===")
        return message
