"""Operator Takeover Checker component for Langflow."""

from loguru import logger

from langflow.custom import Component
from langflow.io import HandleInput, MessageTextInput, Output
from langflow.schema.message import Message


class OperatorTakeover(Component):
    """Check if an operator has taken over the conversation.

    This component checks if the message starts with a specific mention prefix
    to determine if an operator is responding instead of the AI.

    Typical flow (LangFlow 1):
    Chat Input → Operator Takeover Checker → Sender Router
                                                ├─ (Operator) → Chat Output
                                                └─ (User) → LLM → Chat Output

    For customer messages, text won't start with the mention prefix, so operator_takeover = False.
    For operator messages, text will start with the mention prefix, so operator_takeover = True.
    """

    display_name = "Operator Takeover Checker"
    description = "Check if operator has taken over the conversation (skip AI if true)"
    icon = "UserCheck"

    inputs = [
        HandleInput(
            name="message_input",
            display_name="Message",
            info="Message from Chat Input",
            input_types=["Message"],
            required=True,
        ),
        MessageTextInput(
            name="operator_mention_prefix",
            display_name="Operator Mention Prefix",
            info="Mention prefix that indicates operator message (e.g., @manager-ai-dev-chat)",
            value="@manager-ai-dev-chat",
        ),
        MessageTextInput(
            name="operator_sender",
            display_name="Operator Sender",
            info="Sender type when operator takes over (e.g., 'Operator')",
            value="Operator",
            advanced=True,
        ),
        MessageTextInput(
            name="operator_sender_name",
            display_name="Operator Sender Name",
            info="Sender name when operator takes over (leave empty to keep original)",
            value="",
            advanced=True,
        ),
    ]

    outputs = [
        Output(
            display_name="Message",
            name="message_output",
            method="check_takeover",
        ),
    ]

    def check_takeover(self) -> Message:
        """Return message only if operator has taken over, otherwise return empty message.

        - Operator messages: Return message (skip AI)
        - User messages: Return empty message (continue to AI)
        """
        msg = self._check_and_flag()

        if msg.operator_takeover:
            logger.info("Returning operator message")
            return msg

        logger.info("Returning empty message (not operator message)")
        return Message(text="")

    def _check_and_flag(self) -> Message:
        """Check if message starts with mention prefix (operator takeover) and pass through the message."""
        logger.info("=== Operator Takeover Checker: Starting ===")

        # Get message text
        message_text = ""
        if hasattr(self.message_input, "text") and self.message_input.text:
            message_text = self.message_input.text
        elif hasattr(self.message_input, "content") and self.message_input.content:
            message_text = self.message_input.content

        # Get operator mention prefix
        operator_mention_prefix = (
            getattr(self, "operator_mention_prefix", "@manager-ai-dev-chat") or "@manager-ai-dev-chat"
        )

        # Check if message starts with operator mention prefix
        operator_takeover = message_text.startswith(operator_mention_prefix)

        # Add flag to message for downstream components
        self.message_input.operator_takeover = operator_takeover

        if operator_takeover:
            logger.info(
                f"Message starts with '{operator_mention_prefix}' - Operator has taken over (AI should be skipped)"
            )

            # Remove mention prefix and following space from text
            cleaned_text = message_text[len(operator_mention_prefix) :].lstrip()
            self.message_input.text = cleaned_text
            logger.info(f"Removed mention prefix. New text: {cleaned_text[:100]}...")

            # Set operator sender
            operator_sender = getattr(self, "operator_sender", "Operator") or "Operator"
            self.message_input.sender = operator_sender
            logger.info(f"Set sender to '{operator_sender}'")

            # Set operator sender_name if specified
            operator_sender_name = getattr(self, "operator_sender_name", "")
            if operator_sender_name and operator_sender_name.strip():
                self.message_input.sender_name = operator_sender_name.strip()
                logger.info(f"Set sender_name to '{operator_sender_name.strip()}'")
            else:
                logger.info(f"Keeping original sender_name: {getattr(self.message_input, 'sender_name', 'N/A')}")
        else:
            logger.info(f"Message doesn't start with '{operator_mention_prefix}' - AI is in control")

        logger.info(f"return: {self.message_input}")
        logger.info("=== Operator Takeover Checker: Complete ===")

        return self.message_input
