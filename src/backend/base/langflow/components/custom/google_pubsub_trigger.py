"""Pub/Sub Trigger component for Langflow - waits for messages."""

import json
import time

from google.oauth2 import service_account
from loguru import logger

from langflow.custom import Component
from langflow.io import BoolInput, HandleInput, IntInput, MessageTextInput, Output, SecretStrInput
from langflow.schema.message import Message

# Constants
DEFAULT_WAIT_TIMEOUT = 300  # 5 minutes


class GooglePubSubTrigger(Component):
    """Wait for and receive messages from Google Cloud Pub/Sub.

    Use this component to wait for messages (e.g., from Google Chat)
    published to a Pub/Sub topic. The component blocks until a message
    arrives or timeout is reached.

    Typical flow:
    Google Chat Sender → Pub/Sub Trigger → (process response)

    Features:
    - Waits for messages with configurable timeout
    - Filters messages by thread (using sender_result connection)
    - Filters out bot messages
    - Parses Google Chat message format

    Output:
    - Returns message content if received
    - Returns empty message if timeout (no message)
    """

    display_name = "Google Pub/Sub Trigger"
    description = "Wait for messages from Google Cloud Pub/Sub. Blocks until message arrives or timeout."
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
            info="Connect GoogleChatSender Result to filter by thread",
            input_types=["Data"],
            required=False,
        ),
        HandleInput(
            name="customer_message",
            display_name="Customer Message",
            info="Customer message from Chat Input (passed through if no human response)",
            input_types=["Message"],
            required=False,
        ),
        IntInput(
            name="wait_timeout",
            display_name="Wait Timeout (seconds)",
            info="Maximum time to wait for a message (0 = wait forever)",
            value=10,  # Short default for interrupt window
        ),
        MessageTextInput(
            name="thread_filter",
            display_name="Thread Filter",
            info="Only accept messages from this thread (optional, for Google Chat)",
            required=False,
            advanced=True,
        ),
        MessageTextInput(
            name="bot_email",
            display_name="Bot Email Filter",
            info="Ignore messages from this email (e.g., bot's service account email)",
            required=False,
            advanced=True,
        ),
        BoolInput(
            name="parse_google_chat",
            display_name="Parse as Google Chat",
            info="Parse messages as Google Chat format",
            value=True,
            advanced=True,
        ),
        BoolInput(
            name="acknowledge_message",
            display_name="Acknowledge Message",
            info="Acknowledge message after receiving (removes from queue)",
            value=True,
            advanced=True,
        ),
    ]

    outputs = [
        Output(display_name="Response", name="response", method="get_response"),
    ]

    def _get_subscriber_client(self, service_account_info: dict):
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

    def _parse_google_chat_message(self, data: dict) -> dict | None:
        """Parse Google Chat message format."""
        event_type = data.get("type", "")

        # We're interested in MESSAGE events
        if event_type != "MESSAGE":
            logger.debug(f"Skipping non-MESSAGE event: {event_type}")
            return None

        message = data.get("message", {})
        sender = message.get("sender", {})
        thread = message.get("thread", {})
        space = data.get("space", {})

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

    def _get_thread_filter(self) -> str:
        """Get thread filter from sender result or manual input."""
        # Try to get from sender result first
        if self.sender_result and hasattr(self.sender_result, "data") and self.sender_result.data:
            thread_name = self.sender_result.data.get("thread_name", "")
            if thread_name:
                logger.info(f"Using thread filter from sender: {thread_name}")
                return thread_name

        # Fall back to manual input
        manual = getattr(self, "thread_filter", "") or ""
        if manual and manual.strip():
            return manual.strip()

        return ""

    def _should_accept_message(self, parsed: dict) -> bool:
        """Check if message passes filters."""
        # Filter by bot email
        bot_email = getattr(self, "bot_email", "") or ""
        if bot_email and bot_email.strip():
            sender_email = parsed.get("sender_email", "")
            if bot_email.strip() in sender_email:
                logger.debug(f"Skipping bot message from: {sender_email}")
                return False

        # Filter by thread (from sender result or manual input)
        thread_filter = self._get_thread_filter()
        if thread_filter:
            msg_thread = parsed.get("thread_name", "")
            if msg_thread != thread_filter:
                logger.debug(f"Skipping message from different thread: {msg_thread}")
                return False

        return True

    def _wait_for_message_sync(self) -> dict | None:
        """Wait for a message using synchronous pull with polling."""
        logger.info("=== Pub/Sub Trigger: Starting ===")
        start_time = time.time()

        # Parse service account
        if not self.service_account_json or not self.service_account_json.strip():
            msg = "Service Account JSON is required"
            logger.error(msg)
            raise ValueError(msg)

        try:
            service_account_info = json.loads(self.service_account_json, strict=False)
            logger.info(f"Service account: {service_account_info.get('client_email', 'unknown')}")
        except json.JSONDecodeError as e:
            msg = f"Invalid Service Account JSON: {e}"
            logger.error(msg)
            raise ValueError(msg) from e

        # Create subscriber client
        logger.info("Creating Pub/Sub subscriber client...")
        subscriber = self._get_subscriber_client(service_account_info)
        logger.info("Subscriber client created")

        # Build subscription path
        project_id = self.project_id.strip()
        subscription_name = self.subscription_name.strip()
        subscription_path = subscriber.subscription_path(project_id, subscription_name)

        thread_filter = self._get_thread_filter()
        bot_email = getattr(self, "bot_email", "") or service_account_info.get("client_email", "")

        logger.info("=== Pub/Sub Trigger Config ===")
        logger.info(f"  Subscription: {subscription_path}")
        logger.info(f"  Wait timeout: {self.wait_timeout}s")
        logger.info(f"  Thread filter: {thread_filter or 'None'}")
        logger.info(f"  Bot email filter: {bot_email or 'None'}")
        logger.info(f"  Parse as Google Chat: {self.parse_google_chat}")
        logger.info(f"  Acknowledge messages: {self.acknowledge_message}")
        logger.info("=== Waiting for messages... ===")

        wait_timeout = self.wait_timeout or 0
        poll_interval = 3  # Check every 3 seconds
        poll_count = 0

        while True:
            poll_count += 1
            # Check timeout
            elapsed = time.time() - start_time
            if wait_timeout > 0 and elapsed >= wait_timeout:
                logger.warning(f"Timeout after {elapsed:.1f}s - no message received (polled {poll_count} times)")
                return None

            # Pull messages
            logger.info(f"[Poll #{poll_count}] Pulling messages... (elapsed: {elapsed:.1f}s)")
            try:
                response = subscriber.pull(
                    request={
                        "subscription": subscription_path,
                        "max_messages": 10,
                    },
                    timeout=poll_interval,
                )
                received_count = len(response.received_messages)
                logger.info(f"[Poll #{poll_count}] Pull returned {received_count} messages")
            except Exception as e:  # noqa: BLE001
                error_name = type(e).__name__
                # DeadlineExceeded means no messages within timeout - this is normal
                if "DeadlineExceeded" in error_name or "deadline" in str(e).lower():
                    logger.info(f"[Poll #{poll_count}] No messages yet (deadline), continuing...")
                    continue
                # Other errors
                logger.error(f"[Poll #{poll_count}] Pub/Sub pull error: {error_name}: {e}")
                time.sleep(poll_interval)
                continue

            # Process received messages
            for i, received_message in enumerate(response.received_messages):
                logger.info(f"[Poll #{poll_count}] Processing message {i + 1}/{received_count}")
                try:
                    raw_data = received_message.message.data.decode("utf-8")
                    logger.info(f"[Poll #{poll_count}] Raw data: {raw_data[:500]}...")
                    message_data = json.loads(raw_data)
                    logger.info(f"[Poll #{poll_count}] Parsed JSON keys: {list(message_data.keys())}")
                except (json.JSONDecodeError, UnicodeDecodeError) as e:
                    logger.warning(f"[Poll #{poll_count}] Failed to parse message: {e}")
                    # Acknowledge bad messages to remove them
                    if self.acknowledge_message:
                        subscriber.acknowledge(
                            request={
                                "subscription": subscription_path,
                                "ack_ids": [received_message.ack_id],
                            }
                        )
                    continue

                # Parse if Google Chat format
                if self.parse_google_chat:
                    event_type = message_data.get("type", "unknown")
                    logger.info(f"[Poll #{poll_count}] Event type: {event_type}")

                    parsed = self._parse_google_chat_message(message_data)
                    if parsed is None:
                        logger.info(f"[Poll #{poll_count}] Skipping non-MESSAGE event: {event_type}")
                        # Not a MESSAGE event, acknowledge and skip
                        if self.acknowledge_message:
                            subscriber.acknowledge(
                                request={
                                    "subscription": subscription_path,
                                    "ack_ids": [received_message.ack_id],
                                }
                            )
                        continue

                    sender = parsed.get("sender_name")
                    thread = parsed.get("thread_name")
                    logger.info(f"[Poll #{poll_count}] Parsed message: sender={sender}, thread={thread}")
                    logger.info(f"[Poll #{poll_count}] Message text: {parsed.get('text', '')[:100]}")

                    # Check filters
                    if not self._should_accept_message(parsed):
                        logger.info(f"[Poll #{poll_count}] Message filtered out (bot or wrong thread)")
                        if self.acknowledge_message:
                            subscriber.acknowledge(
                                request={
                                    "subscription": subscription_path,
                                    "ack_ids": [received_message.ack_id],
                                }
                            )
                        continue

                    # Found a valid message!
                    logger.info("=== MESSAGE RECEIVED ===")
                    logger.info(f"  From: {parsed.get('sender_name')}")
                    logger.info(f"  Text: {parsed.get('text', '')}")
                    logger.info(f"  Thread: {parsed.get('thread_name')}")
                    logger.info(f"  Time elapsed: {elapsed:.1f}s")

                    if self.acknowledge_message:
                        subscriber.acknowledge(
                            request={
                                "subscription": subscription_path,
                                "ack_ids": [received_message.ack_id],
                            }
                        )
                        logger.info("Message acknowledged")

                    self._received_message = parsed
                    self._raw_event = message_data
                    return parsed

                # Raw message (non-Google Chat)
                logger.info("=== RAW MESSAGE RECEIVED ===")
                logger.info(f"  Data: {raw_data[:200]}")
                logger.info(f"  Time elapsed: {elapsed:.1f}s")

                if self.acknowledge_message:
                    subscriber.acknowledge(
                        request={
                            "subscription": subscription_path,
                            "ack_ids": [received_message.ack_id],
                        }
                    )

                self._received_message = {"text": raw_data, "raw": message_data}
                self._raw_event = message_data
                return self._received_message

            # No valid messages in this batch, continue polling
            logger.info(f"[Poll #{poll_count}] No valid messages in batch, sleeping 1s...")
            time.sleep(1)

    def get_response(self) -> Message:
        """Wait for human response and return it.

        Returns:
            - Human's message if they responded within timeout
            - Empty message (text="") if no response (timeout)

        Use If-Else to check if text is empty:
            - Operator: "not equals"
            - Match Text: "" (empty)
            - True = human responded
            - False = no response, AI should handle
        """
        result = self._wait_for_message_sync()

        if result is None:
            # No human response - return empty message
            return Message(
                text="",
                sender="System",
                sender_name="No Response",
            )

        # Human responded - return their message
        return Message(
            text=result.get("text", ""),
            sender=result.get("sender_name", "Human"),
            sender_name=result.get("sender_name", "Human"),
        )
