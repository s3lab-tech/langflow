"""Human Takeover component - Manages AI/Human control of conversation."""

import json
import time
from datetime import datetime, timezone

from google.oauth2 import service_account
from loguru import logger

from langflow.custom import Component
from langflow.io import HandleInput, IntInput, MessageTextInput, Output, SecretStrInput
from langflow.schema.message import Message


class GoogleChatHumanTakeover(Component):
    """Check for operator responses from Google Chat via Pub/Sub.

    Use this component to check if a support operator has responded
    in Google Chat. The operator can take over the conversation by
    sending any message, and hand back to AI by sending '@ai' or '/handoff'.

    Typical flow:
    Chat Input → Thread Key Generator → Google Chat Sender
                         ↓
                 Human Takeover → If-Else
                       ↓              ├→ (has message) → Chat Output
                 (auto-lookup         └→ (empty) → LLM → Chat Output
                  thread_key
                  from Firestore)

    Features:
    - Auto-looks up thread_key from Firestore (no connection needed from Thread Key Generator)
    - Uses session_id to find the corresponding thread_key
    - Stores control state (AI/Operator) in Firestore
    - Auto-returns to AI after operator timeout

    Output:
    - Returns operator's message if found
    - Returns empty message if no operator response (or handoff command)
    """

    display_name = "Google Chat Human Takeover"
    description = "Check for operator messages via Pub/Sub. Operator sends '@ai' or '/handoff' to return control to AI."
    icon = "MessageSquare"

    inputs = [
        SecretStrInput(
            name="service_account_json",
            display_name="Service Account JSON",
            info="Google Cloud Service Account JSON",
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
            info="Pub/Sub subscription name for human messages",
            required=True,
        ),
        MessageTextInput(
            name="thread_keys_collection",
            display_name="Thread Keys Collection",
            info="Firestore collection where Thread Key Generator stores keys",
            value="thread_keys",
            advanced=True,
        ),
        MessageTextInput(
            name="control_state_collection",
            display_name="Control State Collection",
            info="Firestore collection for storing control state",
            value="conversation_control",
            advanced=True,
        ),
        HandleInput(
            name="customer_message",
            display_name="Customer Message",
            info="Customer message from Chat Input",
            input_types=["Message"],
            required=True,
        ),
        IntInput(
            name="check_timeout",
            display_name="Check Timeout (seconds)",
            info="How long to check for human messages",
            value=3,
        ),
        IntInput(
            name="human_timeout_minutes",
            display_name="Human Timeout (minutes)",
            info="Auto-handoff to AI after this many minutes of human inactivity",
            value=5,
        ),
        MessageTextInput(
            name="handoff_keywords",
            display_name="Handoff Keywords",
            info="Keywords for human to hand back to AI (comma-separated)",
            value="@ai, /handoff, /ai",
            advanced=True,
        ),
        MessageTextInput(
            name="bot_email",
            display_name="Bot Email Filter",
            info="Ignore messages from this email",
            required=False,
            advanced=True,
        ),
    ]

    outputs = [
        Output(
            display_name="Operator Message", name="operator_message", method="get_operator_message", group_outputs=True
        ),
        Output(
            display_name="Customer Message",
            name="customer_message_output",
            method="get_customer_message",
            group_outputs=True,
        ),
    ]

    def _debug_service_account(self):
        """Debug: Log service account JSON info."""
        logger.info("=== DEBUG: Service Account JSON ===")
        logger.info(f"Type: {type(self.service_account_json)}")
        logger.info(f"Length: {len(self.service_account_json) if self.service_account_json else 0}")
        if self.service_account_json:
            # Show first 100 chars (safe, no secrets)
            preview_limit = 100
            preview = (
                self.service_account_json[:preview_limit]
                if len(self.service_account_json) > preview_limit
                else self.service_account_json
            )
            logger.info(f"Preview: {preview}...")
            # Check if it looks like JSON
            if self.service_account_json.strip().startswith("{"):
                logger.info("Looks like JSON: YES")
            else:
                logger.info("Looks like JSON: NO - might be variable reference or encrypted")
        else:
            logger.info("Value: None or empty")
        logger.info("=== END DEBUG ===")

    def _get_firestore_client(self, service_account_info: dict):
        """Create Firestore client."""
        try:
            from google.cloud import firestore
        except ImportError as e:
            msg = "google-cloud-firestore is not installed"
            raise ImportError(msg) from e

        credentials = service_account.Credentials.from_service_account_info(
            service_account_info,
        )
        project_id = service_account_info.get("project_id")
        return firestore.Client(project=project_id, credentials=credentials)

    def _get_pubsub_client(self, service_account_info: dict):
        """Create Pub/Sub client."""
        try:
            from google.cloud import pubsub_v1
        except ImportError as e:
            msg = "google-cloud-pubsub is not installed"
            raise ImportError(msg) from e

        credentials = service_account.Credentials.from_service_account_info(
            service_account_info,
        )
        return pubsub_v1.SubscriberClient(credentials=credentials)

    def _get_session_id(self) -> str:
        """Get session_id from customer message or flow context."""
        logger.info("=== DEBUG HumanTakeover: Getting session_id ===")

        # Method 1: From customer message
        method1_id = ""
        if self.customer_message:
            method1_id = getattr(self.customer_message, "session_id", "") or ""
            logger.info(f"Method 1 (customer_message.session_id): '{method1_id[:20] if method1_id else 'None'}...'")
        else:
            logger.info("Method 1: customer_message is None")

        # Method 2: From graph.session_id
        method2_id = ""
        if hasattr(self, "graph") and self.graph:
            method2_id = getattr(self.graph, "session_id", "") or ""
            logger.info(f"Method 2 (graph.session_id): '{method2_id[:20] if method2_id else 'None'}...'")
        else:
            logger.info("Method 2: graph is None or not available")

        # Use first available
        session_id = method1_id or method2_id
        logger.info(f"Final session_id: '{session_id[:20] if session_id else 'None'}...'")
        logger.info(f"Source: {'Method 1' if method1_id else 'Method 2' if method2_id else 'None'}")
        logger.info("=== END DEBUG HumanTakeover ===")

        return session_id.strip() if session_id else ""

    def _get_thread_key(self, firestore_client) -> str:
        """Look up thread_key from Firestore using session_id."""
        session_id = self._get_session_id()

        if not session_id:
            logger.warning("No session_id found, cannot look up thread_key")
            return ""

        # Look up in thread_keys collection
        collection = self.thread_keys_collection or "thread_keys"
        doc_ref = firestore_client.collection(collection).document(session_id)

        try:
            doc = doc_ref.get()
            if doc.exists:
                thread_key = doc.to_dict().get("thread_key", "")
                if thread_key:
                    logger.info(f"Found thread_key: {thread_key} (session: {session_id[:8]}...)")
                    return thread_key
        except Exception as e:  # noqa: BLE001
            logger.warning(f"Failed to look up thread_key: {e}")

        logger.warning(f"No thread_key found for session: {session_id[:8]}...")
        return ""

    def _get_control_state(self, firestore_client, thread_key: str) -> dict:
        """Get current control state from Firestore."""
        collection = self.control_state_collection or "conversation_control"
        doc_ref = firestore_client.collection(collection).document(thread_key)
        doc = doc_ref.get()

        if doc.exists:
            return doc.to_dict()

        # Default: AI is in control
        return {
            "controller": "AI",
            "last_human_activity": None,
            "thread_key": thread_key,
        }

    def _set_control_state(self, firestore_client, thread_key: str, controller: str):
        """Update control state in Firestore."""
        collection = self.control_state_collection or "conversation_control"
        doc_ref = firestore_client.collection(collection).document(thread_key)

        data = {
            "controller": controller,
            "thread_key": thread_key,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }

        if controller == "HUMAN":
            data["last_human_activity"] = datetime.now(timezone.utc).isoformat()

        doc_ref.set(data, merge=True)
        logger.info(f"Control state updated: {thread_key} → {controller}")

    def _check_human_timeout(self, state: dict) -> bool:
        """Check if human has timed out (auto-handoff to AI)."""
        last_activity = state.get("last_human_activity")
        if not last_activity:
            return False

        try:
            last_time = datetime.fromisoformat(last_activity.replace("Z", "+00:00"))
            elapsed_minutes = (datetime.now(timezone.utc) - last_time).total_seconds() / 60

            if elapsed_minutes > self.human_timeout_minutes:
                logger.info(f"Human timeout: {elapsed_minutes:.1f} min > {self.human_timeout_minutes} min")
                return True
        except (ValueError, TypeError) as e:
            logger.warning(f"Failed to parse last_human_activity: {e}")

        return False

    def _is_handoff_message(self, text: str) -> bool:
        """Check if message is a handoff command."""
        keywords = [k.strip().lower() for k in self.handoff_keywords.split(",")]
        text_lower = text.lower().strip()

        for keyword in keywords:
            if keyword and keyword in text_lower:
                logger.info(f"Handoff keyword detected: {keyword}")
                return True

        return False

    def _check_for_human_message(self, subscriber, subscription_path: str, bot_email: str) -> dict | None:
        """Quick check for human messages in Pub/Sub."""
        logger.info(f"Checking for human messages (timeout: {self.check_timeout}s)...")

        start_time = time.time()

        while time.time() - start_time < self.check_timeout:
            try:
                response = subscriber.pull(
                    request={
                        "subscription": subscription_path,
                        "max_messages": 10,
                    },
                    timeout=2,
                )

                for received in response.received_messages:
                    try:
                        data = json.loads(received.message.data.decode("utf-8"))

                        if data.get("type") != "MESSAGE":
                            subscriber.acknowledge(
                                request={
                                    "subscription": subscription_path,
                                    "ack_ids": [received.ack_id],
                                }
                            )
                            continue

                        message = data.get("message", {})
                        sender = message.get("sender", {})
                        sender_email = sender.get("email", "")

                        # Skip bot messages
                        if bot_email and bot_email in sender_email:
                            subscriber.acknowledge(
                                request={
                                    "subscription": subscription_path,
                                    "ack_ids": [received.ack_id],
                                }
                            )
                            continue

                        text = message.get("text", "")
                        sender_name = sender.get("displayName", "Human")

                        logger.info(f"Human message found: {sender_name}: {text[:50]}...")

                        subscriber.acknowledge(
                            request={
                                "subscription": subscription_path,
                                "ack_ids": [received.ack_id],
                            }
                        )

                    except (json.JSONDecodeError, KeyError, TypeError) as e:
                        logger.warning(f"Failed to parse message: {e}")
                        continue
                    else:
                        return {
                            "text": text,
                            "sender_name": sender_name,
                            "sender_email": sender_email,
                        }

            except Exception as e:  # noqa: BLE001
                if "DeadlineExceeded" not in str(type(e).__name__):
                    logger.debug(f"Pull error: {e}")
                continue

        logger.info("No human messages found")
        return None

    def _route(self) -> tuple[str, Message | None]:
        """Determine routing and return (route, human_message)."""
        logger.info("=== Human Takeover: Checking control state ===")

        # Debug service account
        self._debug_service_account()

        # Parse service account
        try:
            sa_info = json.loads(self.service_account_json, strict=False)
        except json.JSONDecodeError as e:
            logger.error(f"JSON parse error: {e}")
            msg = f"Invalid Service Account JSON: {e}"
            raise ValueError(msg) from e

        bot_email = self.bot_email or sa_info.get("client_email", "")

        # Get Firestore client
        firestore_client = self._get_firestore_client(sa_info)

        # Look up thread_key from Firestore
        thread_key = self._get_thread_key(firestore_client)
        if not thread_key:
            logger.warning("No thread_key found, routing to AI")
            return ("AI", None)

        # Get control state
        state = self._get_control_state(firestore_client, thread_key)
        current_controller = state.get("controller", "AI")

        logger.info(f"Thread Key: {thread_key}")
        logger.info(f"Current controller: {current_controller}")

        # Check for human timeout
        if current_controller == "HUMAN" and self._check_human_timeout(state):
            logger.info("Human timed out, switching to AI")
            self._set_control_state(firestore_client, thread_key, "AI")
            current_controller = "AI"

        # Get Pub/Sub client
        subscriber = self._get_pubsub_client(sa_info)
        subscription_path = subscriber.subscription_path(
            self.project_id.strip(),
            self.subscription_name.strip(),
        )

        # Check for new human message
        human_msg = self._check_for_human_message(subscriber, subscription_path, bot_email)

        if human_msg:
            text = human_msg.get("text", "")

            # Check for handoff command
            if self._is_handoff_message(text):
                logger.info("Human handed off to AI")
                self._set_control_state(firestore_client, thread_key, "AI")
                # Don't show handoff message to customer, route to AI
                return ("AI", None)

            # Human is taking over or continuing
            if current_controller != "HUMAN":
                logger.info("Human taking over conversation")
            self._set_control_state(firestore_client, thread_key, "HUMAN")

            return (
                "HUMAN",
                Message(
                    text=text,
                    sender=human_msg.get("sender_name", "Support"),
                    sender_name=human_msg.get("sender_name", "Support"),
                ),
            )

        # No human message
        if current_controller == "HUMAN":
            # Human is in control but no new message - still route to human path
            # (customer message will be forwarded to Google Chat)
            logger.info("Human in control, forwarding to human path")
            return ("HUMAN", None)

        # AI is in control
        logger.info("AI in control, routing to AI")
        return ("AI", None)

    def get_operator_message(self) -> Message:
        """Get operator message from Pub/Sub.

        Returns:
            - Operator's message if found (text has content)
            - Empty message if no operator message (text is "")

        Use If-Else to check:
            - text not empty → show operator message to customer
            - text empty → route customer message to AI
        """
        if not hasattr(self, "_route_result"):
            self._route_result = self._route()

        _route, operator_message = self._route_result

        if operator_message:
            # Operator responded - return their message
            return operator_message

        # No operator message - return empty
        return Message(text="", sender="System", sender_name="No Message")

    def get_customer_message(self) -> Message:
        """Pass through the customer message."""
        if self.customer_message and hasattr(self.customer_message, "text"):
            return Message(
                text=self.customer_message.text,
                sender=getattr(self.customer_message, "sender", "User"),
                sender_name=getattr(self.customer_message, "sender_name", "User"),
                session_id=getattr(self.customer_message, "session_id", ""),
            )
        return Message(text="", sender="System", sender_name="No Input")
