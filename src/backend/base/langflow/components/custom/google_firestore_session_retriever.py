"""Google Firestore Session Retriever component for Langflow."""

import json

from google.cloud import firestore
from google.oauth2 import service_account
from loguru import logger

from langflow.custom import Component
from langflow.io import HandleInput, MessageTextInput, Output, SecretStrInput
from langflow.schema import Data


class FirestoreSessionRetriever(Component):
    """Retrieve session data from Google Firestore.

    Use this component to fetch session information stored by
    Google Firestore Session Logger.

    Typical flow:
    Google Chat Message Parser → Firestore Session Retriever → LangFlow API Caller

    This retrieves the session data from Firestore using the thread_key (session_id)
    to verify that the session exists and get additional context if needed.
    """

    display_name = "Google Firestore Session Retriever"
    description = "Retrieve session data from Firestore using session_id"
    icon = "Database"

    inputs = [
        HandleInput(
            name="data_input",
            display_name="Data",
            info="Data containing session_id or thread_key",
            input_types=["Data"],
            required=True,
        ),
        SecretStrInput(
            name="service_account_json",
            display_name="Service Account JSON",
            info="Google Cloud Service Account JSON with Firestore access",
            required=True,
        ),
        MessageTextInput(
            name="session_collection",
            display_name="Session Collection",
            info="Firestore collection name for sessions",
            value="sessions",
        ),
        MessageTextInput(
            name="session_id_field",
            display_name="Session ID Field",
            info="Field name in input data containing session_id",
            value="thread_key",
            advanced=True,
        ),
    ]

    outputs = [
        Output(display_name="Enriched Data", name="enriched_data", method="retrieve_session"),
    ]

    def _get_firestore_client(self, service_account_info: dict) -> firestore.Client:
        """Create Firestore client."""
        credentials = service_account.Credentials.from_service_account_info(
            service_account_info,
        )
        project_id = service_account_info.get("project_id")
        return firestore.Client(project=project_id, credentials=credentials)

    def retrieve_session(self) -> Data:
        """Retrieve session data from Firestore."""
        logger.info("=== Firestore Session Retriever: Starting ===")

        if not self.data_input:
            logger.warning("No data provided")
            return Data(data={})

        # Get input data
        if hasattr(self.data_input, "data"):
            data = self.data_input.data.copy()  # Copy to avoid modifying original
        else:
            data = {}

        # Check if we should skip
        if data.get("skip", False):
            logger.info("Skipping (skip=True)")
            return Data(data=data)

        # Get session_id from specified field
        session_id_field = self.session_id_field or "thread_key"
        session_id = data.get(session_id_field, "")

        # Fallback to session_id field if thread_key is empty
        if not session_id:
            session_id = data.get("session_id", "")

        if not session_id:
            logger.warning(f"No session_id found in field '{session_id_field}'")
            data["session_exists"] = False
            data["session_id"] = ""
            return Data(data=data)

        logger.info(f"Looking up session: {session_id}")

        try:
            # Parse service account JSON
            service_account_info = json.loads(self.service_account_json, strict=False)

            # Get Firestore client
            db = self._get_firestore_client(service_account_info)

            # Retrieve session document
            collection_name = self.session_collection or "sessions"
            session_ref = db.collection(collection_name).document(session_id)
            session_doc = session_ref.get()

            if session_doc.exists:
                session_data = session_doc.to_dict()
                logger.info(f"Session found: {session_id}")
                logger.info(f"Session data: {session_data}")

                # Enrich input data with session info
                data["session_exists"] = True
                data["session_id"] = session_id
                data["session_data"] = session_data
                data["created_at"] = session_data.get("created_at")
                data["updated_at"] = session_data.get("updated_at")
                data["expires_at"] = session_data.get("expires_at")
                data["message_count"] = session_data.get("message_count", 0)

            else:
                logger.warning(f"Session not found: {session_id}")
                data["session_exists"] = False
                data["session_id"] = session_id

        except Exception as e:
            logger.error(f"Error retrieving session: {e}")
            data["session_exists"] = False
            data["session_id"] = session_id
            data["error"] = str(e)

        logger.info("=== Firestore Session Retriever: Complete ===")
        return Data(data=data)
