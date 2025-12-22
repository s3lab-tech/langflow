"""Firestore Logger component for Langflow."""

import json
import time
from datetime import datetime, timezone
from typing import TYPE_CHECKING

from google.oauth2 import service_account
from loguru import logger

from langflow.custom import Component
from langflow.io import HandleInput, IntInput, MessageTextInput, Output, SecretStrInput
from langflow.schema import Data

if TYPE_CHECKING:
    from google.cloud import firestore

# Default timeout in seconds
DEFAULT_TIMEOUT = 30


class GoogleFirestoreLogger(Component):
    """Log data to Google Cloud Firestore."""

    display_name = "Google Firestore Logger"
    description = "Save logs and data to Google Cloud Firestore"
    icon = "database"

    inputs = [
        SecretStrInput(
            name="service_account_json",
            display_name="Service Account JSON",
            info="Google Cloud Service Account JSON key content",
            required=True,
        ),
        MessageTextInput(
            name="database_name",
            display_name="Database Name",
            info="Firestore database name (leave empty for default)",
            value="",
            required=False,
        ),
        MessageTextInput(
            name="collection_name",
            display_name="Collection Name",
            info="Firestore collection name",
            value="langflow_logs",
            required=True,
        ),
        MessageTextInput(
            name="document_id",
            display_name="Document ID",
            info="Optional document ID (auto-generated if empty)",
            required=False,
        ),
        HandleInput(
            name="data_input",
            display_name="Data Input",
            info="Data to log (Message, Data, or text)",
            input_types=["Message", "Data"],
            required=False,
        ),
        MessageTextInput(
            name="log_message",
            display_name="Log Message",
            info="Custom log message",
            required=False,
        ),
        MessageTextInput(
            name="log_level",
            display_name="Log Level",
            info="Log level (INFO, WARNING, ERROR, DEBUG)",
            value="INFO",
        ),
        IntInput(
            name="timeout",
            display_name="Timeout (seconds)",
            info="Timeout for Firestore operations",
            value=DEFAULT_TIMEOUT,
            advanced=True,
        ),
    ]

    outputs = [
        Output(display_name="Result", name="result", method="log_to_firestore"),
    ]

    def _get_firestore_client(self) -> "firestore.Client":
        """Create Firestore client with service account credentials."""
        logger.debug("=== Firestore Logger: Creating client ===")
        start_time = time.time()

        try:
            from google.cloud import firestore
        except ImportError as e:
            msg = "google-cloud-firestore is not installed. Run: pip install google-cloud-firestore"
            raise ImportError(msg) from e

        if not self.service_account_json or not self.service_account_json.strip():
            msg = "Service Account JSON is required"
            raise ValueError(msg)

        logger.debug("Parsing Service Account JSON...")
        try:
            service_account_info = json.loads(self.service_account_json, strict=False)
        except json.JSONDecodeError as e:
            msg = f"Invalid Service Account JSON: {e}"
            raise ValueError(msg) from e

        project_id = service_account_info.get("project_id")
        client_email = service_account_info.get("client_email", "unknown")
        logger.debug(f"Project ID: {project_id}")
        logger.debug(f"Service Account: {client_email}")

        if not project_id:
            msg = "project_id not found in Service Account JSON"
            raise ValueError(msg)

        logger.debug("Creating credentials...")
        credentials = service_account.Credentials.from_service_account_info(service_account_info)
        logger.debug(f"Credentials created in {time.time() - start_time:.2f}s")

        logger.debug("Creating Firestore client...")
        client_start = time.time()

        # Get database name (use default if not specified)
        database = self.database_name.strip() if self.database_name else "(default)"
        logger.debug(f"Database: {database}")

        client = firestore.Client(
            project=project_id,
            credentials=credentials,
            database=database,
        )
        logger.debug(f"Firestore client created in {time.time() - client_start:.2f}s")
        logger.debug(f"Total client setup time: {time.time() - start_time:.2f}s")

        return client

    def log_to_firestore(self) -> Data:
        """Log data to Firestore collection."""
        logger.info("=== Firestore Logger: Starting ===")
        database = self.database_name.strip() if self.database_name else "(default)"
        logger.info(f"Database: {database}")
        logger.info(f"Collection: {self.collection_name}")
        logger.info(f"Document ID: {self.document_id or '(auto-generated)'}")
        logger.info(f"Timeout: {self.timeout}s")

        total_start = time.time()

        try:
            client = self._get_firestore_client()
        except Exception as e:
            logger.error(f"Failed to create Firestore client: {e}")
            raise

        logger.debug("Getting collection reference...")
        collection_ref = client.collection(self.collection_name)

        # Build log document
        logger.debug("Building log document...")
        log_data = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "log_level": self.log_level or "INFO",
            "message": self.log_message or "",
        }

        # Add input data if provided
        if self.data_input:
            logger.debug(f"Adding input data (type: {type(self.data_input).__name__})")
            if hasattr(self.data_input, "data"):
                log_data["data"] = self.data_input.data
            if hasattr(self.data_input, "text"):
                log_data["text"] = self.data_input.text
            if hasattr(self.data_input, "get_text"):
                log_data["content"] = self.data_input.get_text()

        logger.debug(f"Log data keys: {list(log_data.keys())}")

        # Create or update document
        write_start = time.time()
        try:
            if self.document_id and self.document_id.strip():
                logger.info(f"Setting document: {self.document_id}")
                doc_ref = collection_ref.document(self.document_id)
                doc_ref.set(log_data, merge=True, timeout=self.timeout)
                doc_id = self.document_id
                logger.info(f"Document updated in {time.time() - write_start:.2f}s")
            else:
                logger.info("Adding new document...")
                doc_ref = collection_ref.add(log_data, timeout=self.timeout)
                doc_id = doc_ref[1].id
                logger.info(f"Document created: {doc_id} in {time.time() - write_start:.2f}s")
        except Exception as e:
            logger.error(f"Firestore write failed after {time.time() - write_start:.2f}s: {e}")
            logger.error(f"Error type: {type(e).__name__}")
            raise

        logger.info(f"=== Firestore Logger: Complete in {time.time() - total_start:.2f}s ===")

        return Data(
            text=f"Logged to Firestore: {self.collection_name}/{doc_id}",
            data={
                "collection": self.collection_name,
                "document_id": doc_id,
                "log_level": log_data["log_level"],
                "timestamp": log_data["timestamp"],
                "success": True,
            },
        )
