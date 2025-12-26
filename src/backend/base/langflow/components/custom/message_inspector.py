"""Message Inspector component for Langflow."""

from datetime import datetime, timezone

from loguru import logger

from langflow.custom import Component
from langflow.io import BoolInput, HandleInput, Output
from langflow.schema.message import Message


class MessageInspector(Component):
    """Inspect and normalize Message to ensure required fields exist.

    Use this component right after Chat Input to validate and normalize messages.

    Required fields (will be added if missing):
    - text: Message text (default: "")
    - sender: Sender type (default: "User")
    - sender_name: Sender display name (default: "User")
    - session_id: Session ID (from graph if not present)
    - timestamp: UTC timestamp (current time if not present)

    Input:
    - Message from Chat Input (or any Message source)

    Output:
    - Normalized Message with all required fields
    - Logs all Message attributes for debugging

    Typical flow (LangFlow 1):
    Chat Input → Message Inspector → Human Takeover Checker → ...
    """

    display_name = "Message Inspector"
    description = "Inspect and normalize Message (ensure required fields exist)"
    icon = "Bug"

    inputs = [
        HandleInput(
            name="message_input",
            display_name="Message",
            info="Message to inspect and normalize",
            input_types=["Message"],
            required=True,
        ),
        BoolInput(
            name="normalize",
            display_name="Normalize Message",
            info="Add missing required fields with defaults",
            value=True,
            advanced=True,
        ),
    ]

    outputs = [
        Output(
            display_name="Message",
            name="message_output",
            method="inspect_message",
        ),
    ]

    def inspect_message(self) -> Message:
        """Inspect the message and log its attributes."""
        logger.info("=== Message Inspector: Starting ===")

        logger.info(f"message_input: {self.message_input}")

        # Type and basic info
        # logger.info(f"Type: {type(self.message_input)}")
        # logger.info(f"Class: {self.message_input.__class__.__name__}")

        # Standard Message fields
        # logger.info("\n--- Standard Fields ---")
        # logger.info(f"text: {getattr(self.message_input, 'text', 'NOT_FOUND')}")
        # logger.info(f"sender: {getattr(self.message_input, 'sender', 'NOT_FOUND')}")
        # logger.info(f"sender_name: {getattr(self.message_input, 'sender_name', 'NOT_FOUND')}")
        # logger.info(f"session_id: {getattr(self.message_input, 'session_id', 'NOT_FOUND')}")
        # logger.info(f"timestamp: {getattr(self.message_input, 'timestamp', 'NOT_FOUND')}")

        # Custom attributes
        # logger.info("\n--- Custom Attributes ---")
        custom_attrs = [
            attr
            for attr in dir(self.message_input)
            if not attr.startswith("_")
            and attr
            not in [
                "text",
                "sender",
                "sender_name",
                "session_id",
                "timestamp",
                "files",
                "context_id",
                "flow_id",
                "error",
                "edit",
                "properties",
                "category",
                "content_blocks",
                "duration",
                "data",
                "text_key",
            ]
        ]
        for attr in custom_attrs:
            try:
                value = getattr(self.message_input, attr)
                if not callable(value):
                    # logger.info(f"{attr}: {value}")
                    pass
            except Exception as e:
                logger.info(f"{attr}: ERROR - {e}")

        # All attributes (for reference)
        # logger.info("\n--- All Attributes ---")
        # all_attrs = [attr for attr in dir(self.message_input) if not attr.startswith("_")]
        # logger.info(f"Available attributes: {all_attrs}")

        # Normalize message if enabled
        if self.normalize:
            # logger.info("--- Normalizing Message ---")

            # Ensure text exists
            if not hasattr(self.message_input, "text") or not self.message_input.text:
                logger.warning("Missing or empty 'text' field - setting to empty string")
                self.message_input.text = ""

            # Ensure sender exists
            if not hasattr(self.message_input, "sender") or not self.message_input.sender:
                logger.warning("Missing or empty 'sender' field - setting to 'User'")
                self.message_input.sender = "User"

            # Ensure sender_name exists
            if not hasattr(self.message_input, "sender_name") or not self.message_input.sender_name:
                logger.warning("Missing or empty 'sender_name' field - setting to 'User'")
                self.message_input.sender_name = "User"

            # Ensure session_id exists
            if not hasattr(self.message_input, "session_id") or not self.message_input.session_id:
                # Try to get from graph
                if hasattr(self, "graph") and self.graph:
                    session_id = getattr(self.graph, "session_id", "")
                    if session_id:
                        logger.warning(f"Missing 'session_id' - using graph.session_id: {session_id}")
                        self.message_input.session_id = session_id
                    else:
                        logger.error("Missing 'session_id' and graph.session_id not available")
                        self.message_input.session_id = ""
                else:
                    logger.error("Missing 'session_id' and graph not available")
                    self.message_input.session_id = ""

            # Ensure timestamp exists
            if not hasattr(self.message_input, "timestamp") or not self.message_input.timestamp:
                current_time = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S %Z")
                logger.warning(f"Missing 'timestamp' field - setting to current time: {current_time}")
                self.message_input.timestamp = current_time

            #logger.info("Normalization complete")
            #logger.info(f"Final text: {self.message_input.text[:100]}...")
            #logger.info(f"Final sender: {self.message_input.sender}")
            #logger.info(f"Final sender_name: {self.message_input.sender_name}")
            #logger.info(f"Final session_id: {self.message_input.session_id}")
            #logger.info(f"Final timestamp: {self.message_input.timestamp}")

        logger.info(f"return: {self.message_input}")
        logger.info("=== Message Inspector: Complete ===")

        # Return message with only essential fields
        return self.message_input
