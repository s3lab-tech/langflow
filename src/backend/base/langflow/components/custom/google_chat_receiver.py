"""Google Chat Receiver component for Langflow using Pub/Sub."""

import json
import time

from google.oauth2 import service_account
from loguru import logger

from langflow.custom import Component
from langflow.io import BoolInput, HandleInput, IntInput, MessageTextInput, Output, SecretStrInput
from langflow.schema import Data
from langflow.schema.message import Message

# Constants
DEFAULT_TIMEOUT = 30
MAX_MESSAGES_DEFAULT = 10


class GoogleChatReceiver(Component):
    """Receive human responses from Google Chat via Pub/Sub subscription."""

    display_name = "Google Chat Receiver"
    description = "Receive human responses from Google Chat via Pub/Sub"
    icon = "MessageSquare"

    inputs = [
        SecretStrInput(
            name="service_account_json",
            display_name="Service Account JSON",
            info="Google Cloud Service Account JSON with Pub/Sub access",
            required=True,
        ),
        MessageTextInput(
            name="project_id",
            display_name="Project ID",
            info="Google Cloud Project ID",
            required=True,
        ),
        MessageTextInput(
            name="subscription_name",
            display_name="Subscription Name",
            info="Pub/Sub subscription name (e.g., 'google-chat-sub')",
            required=True,
        ),
        HandleInput(
            name="sender_result",
            display_name="Sender Result",
            info="Connect the Result output from Google Chat Sender to filter by thread",
            input_types=["Data"],
            required=False,
        ),
        MessageTextInput(
            name="thread_name_filter",
            display_name="Thread Name Filter",
            info="Only return messages from this thread (optional)",
            required=False,
            advanced=True,
        ),
        IntInput(
            name="max_messages",
            display_name="Max Messages",
            info="Maximum number of messages to pull from Pub/Sub",
            value=MAX_MESSAGES_DEFAULT,
        ),
        IntInput(
            name="timeout",
            display_name="Timeout (seconds)",
            info="Timeout for Pub/Sub pull operation",
            value=DEFAULT_TIMEOUT,
            advanced=True,
        ),
        BoolInput(
            name="acknowledge_messages",
            display_name="Acknowledge Messages",
            info="Acknowledge messages after reading (removes from queue)",
            value=True,
            advanced=True,
        ),
        MessageTextInput(
            name="bot_email",
            display_name="Bot Email",
            info="Service account email to filter out bot messages",
            required=False,
            advanced=True,
        ),
    ]

    outputs = [
        Output(display_name="Human Messages", name="human_messages", method="get_human_messages"),
        Output(display_name="Has Response", name="has_response", method="check_has_response"),
        Output(display_name="Latest Response", name="latest_response", method="get_latest_response"),
    ]

    def _get_pubsub_client(self, service_account_info: dict):
        """Create Pub/Sub subscriber client."""
        try:
            from google.cloud import pubsub_v1
        except ImportError as e:
            msg = "google-cloud-pubsub is not installed. Run: pip install google-cloud-pubsub"
            raise ImportError(msg) from e

        credentials = service_account.Credentials.from_service_account_info(
            service_account_info,
        )

        return pubsub_v1.SubscriberClient(credentials=credentials)

    def _get_thread_name_filter(self) -> str:
        """Get thread name filter from sender result or manual input."""
        # Try to get from sender result first
        if self.sender_result and hasattr(self.sender_result, "data") and self.sender_result.data:
            thread_name = self.sender_result.data.get("thread_name", "")
            if thread_name:
                logger.info(f"Using thread filter from sender result: {thread_name}")
                return thread_name

        # Fall back to manual input
        manual = getattr(self, "thread_name_filter", "")
        if manual and manual.strip():
            logger.info(f"Using manual thread filter: {manual}")
            return manual.strip()

        logger.info("No thread filter - will return all messages")
        return ""

    def _parse_chat_message(self, pubsub_message) -> dict | None:
        """Parse Google Chat message from Pub/Sub message."""
        try:
            data = pubsub_message.data.decode("utf-8")
            message_data = json.loads(data)

            # Google Chat Pub/Sub format
            # The message contains the Chat event
            event_type = message_data.get("type", "")

            # We're interested in MESSAGE events
            if event_type != "MESSAGE":
                logger.debug(f"Skipping non-MESSAGE event: {event_type}")
                return None

            message = message_data.get("message", {})
            sender = message.get("sender", {})
            thread = message.get("thread", {})
            space = message_data.get("space", {})

            return {
                "text": message.get("text", ""),
                "message_name": message.get("name", ""),
                "sender_name": sender.get("displayName", "Unknown"),
                "sender_email": sender.get("email", ""),
                "sender_type": sender.get("type", ""),
                "thread_name": thread.get("name", ""),
                "space_name": space.get("name", ""),
                "create_time": message.get("createTime", ""),
                "event_type": event_type,
            }
        except (json.JSONDecodeError, AttributeError) as e:
            logger.warning(f"Failed to parse Pub/Sub message: {e}")
            return None

    def _fetch_messages(self) -> list[dict]:
        """Fetch messages from Pub/Sub subscription."""
        logger.info("=== Google Chat Receiver: Fetching from Pub/Sub ===")
        start_time = time.time()

        # Get thread filter
        thread_filter = self._get_thread_name_filter()

        # Parse service account
        if not self.service_account_json or not self.service_account_json.strip():
            msg = "Service Account JSON is required"
            raise ValueError(msg)

        try:
            service_account_info = json.loads(self.service_account_json, strict=False)
        except json.JSONDecodeError as e:
            msg = f"Invalid Service Account JSON: {e}"
            raise ValueError(msg) from e

        bot_email = self.bot_email or service_account_info.get("client_email", "")
        logger.debug(f"Bot email filter: {bot_email}")

        # Create Pub/Sub client
        subscriber = self._get_pubsub_client(service_account_info)

        # Build subscription path
        project_id = self.project_id.strip()
        subscription_name = self.subscription_name.strip()
        subscription_path = subscriber.subscription_path(project_id, subscription_name)

        logger.info(f"Pulling from: {subscription_path}")
        logger.info(f"Thread filter: {thread_filter or 'None'}")
        logger.info(f"Max messages: {self.max_messages}")

        # Pull messages
        try:
            response = subscriber.pull(
                request={
                    "subscription": subscription_path,
                    "max_messages": self.max_messages,
                },
                timeout=self.timeout,
            )
        except Exception as e:  # noqa: BLE001
            logger.error(f"Pub/Sub pull error: {e}")
            # Return empty list if no messages or error
            self._messages = []
            return []

        received_messages = response.received_messages
        logger.info(f"Received {len(received_messages)} messages from Pub/Sub")

        # Parse and filter messages
        filtered_messages = []
        ack_ids = []

        for received_message in received_messages:
            parsed = self._parse_chat_message(received_message.message)

            if parsed is None:
                # Still acknowledge non-parseable messages
                if self.acknowledge_messages:
                    ack_ids.append(received_message.ack_id)
                continue

            # Skip bot messages
            if bot_email and bot_email in parsed.get("sender_email", ""):
                logger.debug(f"Skipping bot message from: {parsed.get('sender_email')}")
                if self.acknowledge_messages:
                    ack_ids.append(received_message.ack_id)
                continue

            # Filter by thread if specified
            if thread_filter:
                msg_thread = parsed.get("thread_name", "")
                if msg_thread != thread_filter:
                    logger.debug(f"Skipping message from different thread: {msg_thread}")
                    if self.acknowledge_messages:
                        ack_ids.append(received_message.ack_id)
                    continue

            filtered_messages.append(parsed)
            if self.acknowledge_messages:
                ack_ids.append(received_message.ack_id)

        # Acknowledge processed messages
        if ack_ids and self.acknowledge_messages:
            try:
                subscriber.acknowledge(
                    request={
                        "subscription": subscription_path,
                        "ack_ids": ack_ids,
                    }
                )
                logger.info(f"Acknowledged {len(ack_ids)} messages")
            except Exception as e:  # noqa: BLE001
                logger.warning(f"Failed to acknowledge messages: {e}")

        logger.info(f"Filtered to {len(filtered_messages)} human messages in {time.time() - start_time:.2f}s")
        self._messages = filtered_messages
        return filtered_messages

    def get_human_messages(self) -> list[Data]:
        """Get all human messages from Pub/Sub."""
        messages = self._fetch_messages()

        data_list = []
        for msg in messages:
            data = Data(
                text=msg.get("text", ""),
                data={
                    "message_name": msg.get("message_name"),
                    "sender_name": msg.get("sender_name"),
                    "sender_email": msg.get("sender_email"),
                    "sender_type": msg.get("sender_type"),
                    "create_time": msg.get("create_time"),
                    "thread_name": msg.get("thread_name"),
                    "space_name": msg.get("space_name"),
                },
            )
            data_list.append(data)

        return data_list

    def check_has_response(self) -> Data:
        """Check if there are any human responses."""
        messages = getattr(self, "_messages", None)
        if messages is None:
            messages = self._fetch_messages()

        has_response = len(messages) > 0

        return Data(
            text=str(has_response),
            data={
                "has_response": has_response,
                "message_count": len(messages),
            },
        )

    def get_latest_response(self) -> Message:
        """Get the latest human response."""
        messages = getattr(self, "_messages", None)
        if messages is None:
            messages = self._fetch_messages()

        if not messages:
            return Message(text="", sender="System", sender_name="No Response")

        # Get the most recent message (last in list from Pub/Sub)
        latest = messages[-1] if messages else {}

        return Message(
            text=latest.get("text", ""),
            sender=latest.get("sender_name", "Human"),
            sender_name=latest.get("sender_name", "Human"),
        )
