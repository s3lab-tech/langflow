"""Thread Key Generator component for Langflow."""

import json
import random
import time
from datetime import datetime, timezone

from google.oauth2 import service_account
from loguru import logger

from langflow.custom import Component
from langflow.io import BoolInput, HandleInput, IntInput, MessageTextInput, Output, SecretStrInput
from langflow.schema.message import Message


class ThreadKeyGenerator(Component):
    """Generate a short, memorable thread key for Google Chat conversations.

    Use this component to create a unique thread key for each customer conversation.
    The thread key groups all messages from the same session in one Google Chat thread.

    Typical flow:
    Thread Key Generator ──(Thread Key)──→ Google Chat Sender (thread_key)
           │
           └── session_id auto-detected from flow context

    Features:
    - Auto-detects session_id from flow context (no input connection needed)
    - Persists thread key in Firestore (keyed by session_id)
    - First message: generates new thread key (e.g., 'A3F8K2')
    - Subsequent messages: retrieves same thread key from Firestore
    - Customizable length, prefix/suffix, character sets

    Usage:
    1. Add Thread Key Generator to your flow (no input connection required)
    2. Connect Thread Key Generator (Thread Key) → Google Chat Sender (Thread Key)

    Example outputs: 'A3F8K2', 'TKT-B7M2P9', 'Q-X4N8'
    """

    display_name = "Thread Key Generator"
    description = "Generate a short thread key (e.g., 'A3F8K2'). Auto-detects session, persists in Firestore."
    icon = "Hash"

    inputs = [
        HandleInput(
            name="input_message",
            display_name="Input Message",
            info="Connect from Chat Input. Session ID is extracted automatically.",
            input_types=["Message"],
            required=True,
        ),
        SecretStrInput(
            name="service_account_json",
            display_name="Service Account JSON",
            info="Google Cloud Service Account JSON for Firestore",
            required=True,
        ),
        MessageTextInput(
            name="firestore_collection",
            display_name="Firestore Collection",
            info="Firestore collection for storing thread keys",
            value="thread_keys",
            advanced=True,
        ),
        MessageTextInput(
            name="prefix",
            display_name="Prefix",
            info="Prefix for the key (e.g., 'TKT-', 'Q-')",
            value="",
        ),
        IntInput(
            name="key_length",
            display_name="Key Length",
            info="Number of characters (excluding prefix)",
            value=6,
        ),
        BoolInput(
            name="include_uppercase",
            display_name="Include Uppercase (A-Z)",
            info="Include uppercase letters",
            value=True,
        ),
        BoolInput(
            name="include_lowercase",
            display_name="Include Lowercase (a-z)",
            info="Include lowercase letters",
            value=False,
        ),
        BoolInput(
            name="include_digits",
            display_name="Include Digits (0-9)",
            info="Include numbers",
            value=True,
        ),
        BoolInput(
            name="exclude_ambiguous",
            display_name="Exclude Ambiguous Characters",
            info="Exclude confusing characters (0/O, 1/I/L)",
            value=True,
            advanced=True,
        ),
        MessageTextInput(
            name="suffix",
            display_name="Suffix",
            info="Suffix for the key (optional)",
            value="",
            advanced=True,
        ),
    ]

    outputs = [
        Output(display_name="Message", name="message_output", method="get_message_output"),
    ]

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

    def _get_session_id(self) -> str:
        """Get session_id from input message or flow context."""
        logger.info("=== DEBUG: Getting session_id ===")

        # Method 1: From input message
        method1_id = ""
        if self.input_message:
            method1_id = getattr(self.input_message, "session_id", "") or ""
            logger.info(f"Method 1 (input_message.session_id): '{method1_id[:20] if method1_id else 'None'}...'")
        else:
            logger.info("Method 1: input_message is None")

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
        logger.info("=== END DEBUG ===")

        return session_id.strip() if session_id else ""

    def _build_character_set(self) -> str:
        """Build character set based on options."""
        chars = ""

        if self.include_uppercase:
            if self.exclude_ambiguous:
                chars += "ABCDEFGHJKLMNPQRSTUVWXYZ"  # pragma: allowlist secret
            else:
                chars += "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

        if self.include_lowercase:
            if self.exclude_ambiguous:
                chars += "abcdefghjkmnpqrstuvwxyz"  # pragma: allowlist secret
            else:
                chars += "abcdefghijklmnopqrstuvwxyz"

        if self.include_digits:
            if self.exclude_ambiguous:
                chars += "23456789"
            else:
                chars += "0123456789"

        if not chars:
            chars = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"  # pragma: allowlist secret

        return chars

    def _generate_key(self) -> str:
        """Generate a thread key."""
        chars = self._build_character_set()
        random.seed(time.time_ns())
        random_part = "".join(random.choices(chars, k=self.key_length))  # noqa: S311

        prefix = self.prefix.strip() if self.prefix else ""
        suffix = self.suffix.strip() if self.suffix else ""

        return f"{prefix}{random_part}{suffix}"

    def _get_or_create_key(self) -> str:
        """Get existing key from Firestore or create new one."""
        session_id = self._get_session_id()

        if not session_id:
            logger.warning("No session_id found, generating thread key without persistence")
            return self._generate_key()

        logger.info(f"Session ID: {session_id[:8]}...")

        # Parse service account
        try:
            sa_info = json.loads(self.service_account_json, strict=False)
        except json.JSONDecodeError as e:
            logger.error(f"Invalid Service Account JSON: {e}")
            return self._generate_key()

        # Get Firestore client
        try:
            firestore_client = self._get_firestore_client(sa_info)
        except Exception as e:  # noqa: BLE001
            logger.error(f"Failed to create Firestore client: {e}")
            return self._generate_key()

        # Check for existing thread key
        collection = self.firestore_collection or "thread_keys"
        doc_ref = firestore_client.collection(collection).document(session_id)

        try:
            doc = doc_ref.get()
            if doc.exists:
                existing_key = doc.to_dict().get("thread_key", "")
                if existing_key:
                    logger.info(f"Found existing thread key: {existing_key}")
                    return existing_key
        except Exception as e:  # noqa: BLE001
            logger.warning(f"Failed to read from Firestore: {e}")

        # Generate new thread key
        new_key = self._generate_key()
        logger.info(f"Generated new thread key: {new_key}")

        # Save to Firestore
        try:
            doc_ref.set(
                {
                    "thread_key": new_key,
                    "session_id": session_id,
                    "created_at": datetime.now(timezone.utc).isoformat(),
                }
            )
            logger.info("Saved thread key to Firestore")
        except Exception as e:  # noqa: BLE001
            logger.warning(f"Failed to save to Firestore: {e}")

        return new_key

    def get_message_output(self) -> Message:
        """Create/save thread key and pass through the input message."""
        # Create and save thread key (side effect)
        thread_key = self._get_or_create_key()
        logger.info(f"Thread key for this session: {thread_key}")

        # Pass through the input message
        if self.input_message and hasattr(self.input_message, "text"):
            return Message(
                text=self.input_message.text,
                sender=getattr(self.input_message, "sender", "User"),
                sender_name=getattr(self.input_message, "sender_name", "User"),
                session_id=getattr(self.input_message, "session_id", ""),
            )
        return Message(text="", sender="System", sender_name="No Input")
