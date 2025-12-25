"""Firestore Session Logger component for Langflow."""

import datetime
import json

from google.cloud import firestore
from google.oauth2 import service_account
from loguru import logger

from langflow.custom import Component
from langflow.io import HandleInput, MessageTextInput, Output, SecretStrInput
from langflow.schema.message import Message


class FirestoreSessionLogger(Component):
    """Save session and message data to Firestore."""

    display_name = "Google Firestore Session Logger"
    description = "Save session information and messages to Google Cloud Firestore"
    icon = "Database"

    inputs = [
        HandleInput(
            name="message",
            display_name="Message",
            info="Message to save to Firestore",
            input_types=["Message"],
            required=True,
        ),
        SecretStrInput(
            name="service_account_json",
            display_name="Service Account JSON",
            info="Google Cloud Service Account JSON with Firestore access",
            required=True,
        ),
        MessageTextInput(
            name="project_id",
            display_name="Project ID",
            info="Google Cloud Project ID",
            required=True,
        ),
        MessageTextInput(
            name="session_collection",
            display_name="Session Collection",
            info="Firestore collection name for sessions",
            value="sessions",
            advanced=True,
        ),
        MessageTextInput(
            name="messages_collection",
            display_name="Messages Collection",
            info="Firestore collection name for messages (subcollection of sessions)",
            value="messages",
            advanced=True,
        ),
        MessageTextInput(
            name="session_expiry_hours",
            display_name="Session Expiry (hours)",
            info="Session expiry time in hours (default: 24)",
            value="24",
            advanced=True,
        ),
    ]

    outputs = [
        Output(display_name="Message", name="output_message", method="save_to_firestore"),
    ]

    def _get_firestore_client(self, service_account_info: dict) -> firestore.Client:
        """Create Firestore client."""
        credentials = service_account.Credentials.from_service_account_info(
            service_account_info,
        )
        return firestore.Client(
            project=self.project_id.strip(),
            credentials=credentials,
        )

    def _parse_service_account(self) -> dict:
        """Parse service account JSON."""
        if not self.service_account_json or not self.service_account_json.strip():
            msg = "Service Account JSON is required"
            raise ValueError(msg)

        try:
            return json.loads(self.service_account_json, strict=False)
        except json.JSONDecodeError as e:
            msg = f"Invalid Service Account JSON: {e}"
            raise ValueError(msg) from e

    def _save_session(self, db: firestore.Client, session_id: str) -> None:
        """Save or update session information."""
        session_ref = db.collection(self.session_collection).document(session_id)
        session_doc = session_ref.get()

        # Session expiry time
        try:
            expiry_hours = int(self.session_expiry_hours)
        except (ValueError, AttributeError):
            expiry_hours = 24

        expires_at = datetime.datetime.utcnow() + datetime.timedelta(hours=expiry_hours)

        if not session_doc.exists:
            # Create new session
            session_data = {
                "created_at": firestore.SERVER_TIMESTAMP,
                "updated_at": firestore.SERVER_TIMESTAMP,
                "expires_at": expires_at,
                "message_count": 1,
            }
            session_ref.set(session_data)
            logger.info(f"[Firestore] New session created: {session_id}")
        else:
            # Update existing session
            session_ref.update(
                {
                    "updated_at": firestore.SERVER_TIMESTAMP,
                    "message_count": firestore.Increment(1),
                }
            )
            logger.info(f"[Firestore] Session updated: {session_id}")

    def _save_message(
        self,
        db: firestore.Client,
        session_id: str,
        message: Message,
    ) -> None:
        """Save message to Firestore."""
        # Messages subcollection under session
        messages_ref = db.collection(self.session_collection).document(session_id).collection(self.messages_collection)

        # Message data
        message_data = {
            "text": message.text or "",
            "sender": message.sender or "Unknown",
            "sender_name": message.sender_name or "Unknown",
            "session_id": session_id,
            "timestamp": firestore.SERVER_TIMESTAMP,
            "created_at": firestore.SERVER_TIMESTAMP,
        }

        # Add optional fields
        if hasattr(message, "flow_id") and message.flow_id:
            message_data["flow_id"] = str(message.flow_id)

        # Convert properties to dict if present
        if hasattr(message, "properties") and message.properties:
            try:
                # Try to convert to dict
                if hasattr(message.properties, "model_dump"):
                    message_data["properties"] = message.properties.model_dump()
                elif hasattr(message.properties, "dict"):
                    message_data["properties"] = message.properties.dict()
                else:
                    # Skip if can't convert
                    logger.warning("Cannot convert properties to dict, skipping")
            except Exception as e:
                logger.warning(f"Error converting properties: {e}")

        # Save message
        doc_ref = messages_ref.add(message_data)
        logger.info(f"[Firestore] Message saved: {doc_ref[1].id}")

    def save_to_firestore(self) -> Message:
        """Save session and message to Firestore."""
        logger.info("=== Firestore Session Logger: Starting ===")

        # Get message and session_id
        message = self.message
        if not message:
            msg = "Message is required"
            raise ValueError(msg)

        # Get session_id from message or graph
        session_id = getattr(message, "session_id", "") or ""
        if not session_id and hasattr(self, "graph") and self.graph:
            session_id = getattr(self.graph, "session_id", "") or ""

        if not session_id:
            msg = "Session ID is required (either in message or from graph)"
            raise ValueError(msg)

        logger.info(f"Session ID: {session_id[:8]}...")
        logger.info(f"Message: {message.text[:100] if message.text else '(empty)'}...")

        # Parse service account
        service_account_info = self._parse_service_account()
        logger.info(f"Service account: {service_account_info.get('client_email', 'unknown')}")

        # Create Firestore client
        db = self._get_firestore_client(service_account_info)
        logger.info("Firestore client created")

        # Save session
        self._save_session(db, session_id)

        # Save message
        self._save_message(db, session_id, message)

        logger.info("=== Firestore Session Logger: Complete ===")

        # Return message as-is for chaining
        return message
