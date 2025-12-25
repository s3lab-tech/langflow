"""Session ID Preserver component for Langflow."""

from loguru import logger

from langflow.custom import Component
from langflow.io import HandleInput, Output
from langflow.schema.message import Message


class SessionIdPreserver(Component):
    """Preserve session_id from original message to LLM output.

    Use this component after LLM to add session_id back to the response message.

    Typical flow:
    Chat Input → LLM → Session ID Preserver → Google Chat Sender
                 ↓
              (session_id lost in LLM output)
                        ↓
              (session_id restored from graph)
    """

    display_name = "Session ID Preserver"
    description = "Add session_id to message from graph context"
    icon = "Key"

    inputs = [
        HandleInput(
            name="message_input",
            display_name="Message",
            info="Message to add session_id to (typically from LLM)",
            input_types=["Message"],
            required=True,
        ),
    ]

    outputs = [
        Output(display_name="Message", name="message_output", method="preserve_session_id"),
    ]

    def _get_session_id(self) -> str:
        """Get session_id from graph context."""
        session_id = ""

        # Get from graph.session_id
        if hasattr(self, "graph") and self.graph:
            session_id = getattr(self.graph, "session_id", "") or ""

        return session_id.strip() if session_id else ""

    def preserve_session_id(self) -> Message:
        """Add session_id to the message."""
        logger.info("=== Session ID Preserver: Starting ===")

        # Get session_id from graph
        session_id = self._get_session_id()

        if not session_id:
            logger.warning("No session_id found in graph context")
            return self.message_input

        # Check if message already has session_id
        existing_session_id = getattr(self.message_input, "session_id", "")
        if existing_session_id:
            logger.info(f"Message already has session_id: {existing_session_id}")
            return self.message_input

        # Add session_id to message
        logger.info(f"Adding session_id to message: {session_id}")
        self.message_input.session_id = session_id

        logger.info("=== Session ID Preserver: Complete ===")
        return self.message_input
