"""
Document Processing Service (Docling-first with legacy Unstructured API fallback)

This service now uses the open-source Docling converter to process PDFs, Word, and ReqIF files
locally so layout-heavy artifacts (images, tables) remain intact for downstream GraphRAG usage.
The legacy Unstructured Serverless API implementation is preserved as a fallback that can be
explicitly enabled via environment variable.
"""

import os
import shutil
from typing import List, Dict, Any, Optional, Tuple, Union
import json
import time
import warnings
import tempfile

# Always import requests (needed for fallback and type hints)
import requests
from urllib3.util.retry import Retry
from requests.adapters import HTTPAdapter
try:
    from urllib3.exceptions import InsecureRequestWarning
except ImportError:
    # Fallback for older urllib3 versions or when urllib3 is bundled with requests
    try:
        import urllib3
        InsecureRequestWarning = getattr(urllib3.exceptions, 'InsecureRequestWarning', None)
    except (ImportError, AttributeError):
        InsecureRequestWarning = None

# Try to import Docling (preferred open-source processor)
try:
    from docling.document_converter import DocumentConverter
    HAS_DOCLING = True
except ImportError:
    HAS_DOCLING = False

# Hugging Face cache helpers (used to repair corrupted model downloads)
# Priority: Environment variable > huggingface_hub default > fallback
HF_HUB_CACHE = os.getenv("HF_HUB_CACHE")
if not HF_HUB_CACHE:
    try:
        from huggingface_hub.constants import HF_HUB_CACHE as _HF_HUB_CACHE
        HF_HUB_CACHE = _HF_HUB_CACHE
    except ImportError:
        # Fallback: use environment variable HF_HOME or default location
        HF_HOME = os.getenv("HF_HOME")
        if HF_HOME:
            HF_HUB_CACHE = os.path.join(HF_HOME, "hub")
        else:
            HF_HUB_CACHE = os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "hub")

DOCLING_LAYOUT_MODEL_REPO = "docling-project/docling-layout-heron"

# Try to import the official SDK, fall back to requests if not available
try:
    from unstructured_client import UnstructuredClient
    from unstructured_client.models import shared, errors
    USE_OFFICIAL_SDK = True
except ImportError:
    USE_OFFICIAL_SDK = False

# Try to import httpx for error handling (used by SDK)
try:
    import httpx
    HAS_HTTPX = True
except ImportError:
    HAS_HTTPX = False

# API URL - should use the URL provided when account was created
# Default is https://api.unstructuredapp.io/general/v0/general according to docs
# But we allow override via environment variable
UNSTRUCTURED_API_URL = os.getenv(
    "UNSTRUCTURED_API_URL",
    "https://api.unstructured.io/general/v0/general"  # Fallback to common URL
)
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB in bytes
USE_UNSTRUCTURED_API_FALLBACK = os.getenv("REQVIBE_USE_UNSTRUCTURED_API", "").lower() in {"1", "true", "yes"}

# Supported document extensions for Docling ingestion (keep lowercase with leading dot)
SUPPORTED_UPLOAD_EXTENSIONS = {
    '.pdf', '.docx', '.xlsx', '.pptx',
    '.md', '.markdown', '.adoc', '.asciidoc',
    '.html', '.htm', '.xhtml',
    '.csv',
    '.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp', '.webp',
    '.reqif'
}

IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff', '.gif', '.webp', '.heic'}


def _infer_source_type(file_extension: str) -> str:
    ext = (file_extension or '').lower()
    if ext in IMAGE_EXTENSIONS:
        return 'image'
    if ext in {'.pdf'}:
        return 'pdf'
    if ext in {'.docx'}:
        return 'word'
    if ext in {'.pptx'}:
        return 'presentation'
    if ext in {'.xlsx', '.csv'}:
        return 'spreadsheet'
    if ext in {'.md', '.markdown', '.adoc', '.asciidoc'}:
        return 'markdown'
    if ext in {'.html', '.htm', '.xhtml'}:
        return 'html'
    if ext in {'.reqif'}:
        return 'reqif'
    return 'document'


def _annotate_element_metadata(elements: List[Dict[str, Any]], filename: str) -> None:
    """Ensure every element contains basic provenance metadata."""
    file_extension = os.path.splitext(filename)[1].lower()
    source_type = _infer_source_type(file_extension)
    for element in elements:
        if not isinstance(element, dict):
            continue
        metadata = element.setdefault('metadata', {})
        metadata.setdefault('source_filename', filename)
        metadata.setdefault('source_extension', file_extension or 'unknown')
        metadata.setdefault('source_type', source_type)
        metadata.setdefault('element_type', element.get('type', 'unknown'))


def _prepare_elements(elements: Any, filename: str) -> List[Dict[str, Any]]:
    """Normalize and annotate element structures before returning to callers."""
    if elements is None:
        return []
    if isinstance(elements, dict):
        elements = [elements]
    if not isinstance(elements, list):
        return []
    _annotate_element_metadata(elements, filename)
    return elements


class UnstructuredServiceError(Exception):
    """Custom exception for Unstructured API service errors."""
    pass


class DoclingProcessingError(Exception):
    """Raised when Docling fails to convert a document."""
    pass


_docling_converter: Optional['DocumentConverter'] = None


def setup_cache_directories(project_root: Optional[str] = None, silent: bool = False) -> None:
    """
    Setup cache directories for Docling, Hugging Face, and RapidOCR on Streamlit Cloud.
    
    This function ensures cache directories exist and sets environment variables
    so that models can be downloaded to writable locations. This MUST be called
    before any Docling, RapidOCR, or Hugging Face libraries are initialized.
    
    Args:
        project_root: Optional project root directory. If None, attempts to detect it.
    """
    # Detect project root if not provided
    if not project_root:
        # Try to find project root by looking for app.py or requirements.txt
        current = os.path.dirname(os.path.abspath(__file__))
        for _ in range(6):  # Max 6 levels up
            if os.path.exists(os.path.join(current, 'app.py')) or \
               os.path.exists(os.path.join(current, 'requirements.txt')):
                if os.path.exists(os.path.join(current, 'domain')):
                    project_root = current
                    break
            current = os.path.dirname(current)
        
        # Fallback to current directory
        if not project_root:
            project_root = os.path.dirname(os.path.abspath(__file__))
    
    # Determine cache base directory
    # On Streamlit Cloud, use /mount/src/{repo_name}/.cache
    # Otherwise, use project_root/.cache
    if os.path.exists("/mount/src"):
        # We're on Streamlit Cloud - find repo name
        repo_name = None
        try:
            for item in os.listdir("/mount/src"):
                item_path = os.path.join("/mount/src", item)
                if os.path.isdir(item_path) and os.path.exists(os.path.join(item_path, "app.py")):
                    repo_name = item
                    break
        except (OSError, PermissionError):
            # If we can't read /mount/src, fall back to project_root
            pass
        
        if repo_name:
            cache_base = os.path.join("/mount/src", repo_name, ".cache")
        else:
            # Fallback: use project_root/.cache
            cache_base = os.path.join(project_root, ".cache")
    else:
        # Local development: use project_root/.cache
        cache_base = os.path.join(project_root, ".cache")
    
    # Create cache directories
    cache_dirs = {
        "rapidocr": os.path.join(cache_base, "rapidocr"),
        "huggingface": os.path.join(cache_base, "huggingface"),
        "huggingface_hub": os.path.join(cache_base, "huggingface", "hub"),
    }
    
    # Create directories with proper error handling
    for cache_dir in cache_dirs.values():
        try:
            os.makedirs(cache_dir, exist_ok=True)
        except (OSError, PermissionError) as e:
            print(f"Warning: Could not create cache directory {cache_dir}: {e}")
            # Continue anyway - libraries might handle missing directories
    
    # Set environment variables if not already set
    # This must be done BEFORE any libraries (Docling, RapidOCR, huggingface_hub) are imported
    if not os.getenv("RAPIDOCR_HOME"):
        os.environ["RAPIDOCR_HOME"] = cache_dirs["rapidocr"]
    
    if not os.getenv("HF_HOME"):
        os.environ["HF_HOME"] = cache_dirs["huggingface"]
    
    if not os.getenv("HF_HUB_CACHE"):
        os.environ["HF_HUB_CACHE"] = cache_dirs["huggingface_hub"]
    
    # Update global HF_HUB_CACHE variable
    global HF_HUB_CACHE, _cache_setup_logged
    HF_HUB_CACHE = cache_dirs["huggingface_hub"]
    
    # Log cache configuration only once per Python process (not on every Streamlit rerun)
    # This prevents the warning from appearing on every message
    if not silent and not getattr(setup_cache_directories, '_logged', False):
        print(f"Cache directories configured:")
        print(f"  RAPIDOCR_HOME: {cache_dirs['rapidocr']}")
        print(f"  HF_HOME: {cache_dirs['huggingface']}")
        print(f"  HF_HUB_CACHE: {cache_dirs['huggingface_hub']}")
        setup_cache_directories._logged = True


def _reset_docling_converter() -> None:
    """Clear cached Docling converter instance so models are reloaded on next use."""
    global _docling_converter
    _docling_converter = None


def _clear_docling_model_cache() -> None:
    """
    Delete cached Docling model artifacts when downloads are incomplete/corrupted.
    
    This primarily targets the huggingface hub folder for the layout detector so the
    next conversion attempt re-downloads missing files such as preprocessor configs.
    """
    cache_root = HF_HUB_CACHE
    if not cache_root:
        return
    model_cache_dir = os.path.join(
        cache_root,
        f"models--{DOCLING_LAYOUT_MODEL_REPO.replace('/', '--')}"
    )
    if os.path.exists(model_cache_dir):
        shutil.rmtree(model_cache_dir, ignore_errors=True)
    _reset_docling_converter()


def _get_docling_converter() -> 'DocumentConverter':
    """
    Lazily instantiate a Docling converter so we reuse heavy resources.
    
    Returns:
        DocumentConverter: Shared Docling converter instance.
    """
    global _docling_converter
    if not HAS_DOCLING:
        raise DoclingProcessingError(
            "Docling is not installed. Install it with `pip install docling` "
            "or set REQVIBE_USE_UNSTRUCTURED_API=1 to fall back to the legacy Unstructured API."
        )
    if _docling_converter is None:
        try:
            _docling_converter = DocumentConverter()
        except Exception as exc:
            raise DoclingProcessingError(
                f"Failed to initialize Docling converter: {exc}"
            ) from exc
    return _docling_converter


def _docling_element_to_dict(element: Any) -> Dict[str, Any]:
    """
    Normalize Docling elements so downstream GraphRAG logic can stay untouched.
    
    Args:
        element: Docling element instance (heading, paragraph, table, etc.)
    """
    element_type = (
        getattr(element, "category", None)
        or getattr(element, "type", None)
        or element.__class__.__name__
    )
    text = (
        getattr(element, "text_representation", None)
        or getattr(element, "text", None)
        or ""
    )
    metadata: Dict[str, Any] = {}
    
    page_number = getattr(element, "page_number", None)
    if page_number is not None:
        metadata["page_number"] = page_number
    
    bbox = getattr(element, "bbox", None)
    if bbox is not None:
        metadata["bbox"] = getattr(bbox, "to_list", lambda: bbox)()
    
    element_id = getattr(element, "content_id", None) or getattr(element, "id", None)
    if element_id:
        metadata["id"] = element_id
    
    if hasattr(element, "classification"):
        metadata["classification"] = getattr(element, "classification")
    
    return {
        "type": element_type or "unknown",
        "text": text or "",
        "metadata": metadata
    }


def _has_textual_content(elements: List[Dict[str, Any]]) -> bool:
    """Return True if any element contains non-empty text."""
    for element in elements:
        text = element.get("text")
        if isinstance(text, str) and text.strip():
            return True
    return False


def _export_docling_document_text(document: Any) -> Optional[str]:
    """
    Try multiple exporters to recover textual content when structured elements are empty.
    """
    exporters = [
        "export_to_markdown",
        "export_to_text",
    ]
    for attr in exporters:
        exporter = getattr(document, attr, None)
        if callable(exporter):
            try:
                text = exporter()
            except Exception:
                continue
            if isinstance(text, str) and text.strip():
                return text
    body = getattr(document, "body", None)
    if body is not None:
        text = getattr(body, "text_representation", None)
        if isinstance(text, str) and text.strip():
            return text
    return None


def _build_elements_from_text(
    text: str,
    base_metadata: Optional[Dict[str, Any]] = None,
    include_page_breaks: bool = True
) -> List[Dict[str, Any]]:
    """
    Convert plain text or markdown into pseudo-elements for downstream processing.
    """
    if not text:
        return []
    elements: List[Dict[str, Any]] = []
    metadata_template = base_metadata.copy() if base_metadata else {}
    if include_page_breaks and "\f" in text:
        parts = text.split("\f")
        for idx, part in enumerate(parts, start=1):
            part_text = part.strip()
            if not part_text:
                continue
            metadata = metadata_template.copy()
            metadata["page_number"] = idx
            elements.append({
                "type": "page_text",
                "text": part_text,
                "metadata": metadata
            })
    else:
        elements.append({
            "type": "document_text",
            "text": text.strip(),
            "metadata": metadata_template
        })
    return elements


def process_document_with_docling(file_bytes: bytes, filename: str) -> List[Dict[str, Any]]:
    """
    Convert documents with Docling to preserve layout-aware content such as tables and images.
    
    Args:
        file_bytes: Raw file bytes.
        filename: Original filename for extension detection.
    """
    suffix = os.path.splitext(filename)[1] or ".bin"
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
        tmp_file.write(file_bytes)
        tmp_path = tmp_file.name
    
    try:
        last_exc: Optional[Exception] = None
        file_extension = os.path.splitext(filename)[1].lower()
        base_metadata = {
            "source_filename": filename,
            "source_extension": file_extension or "unknown",
            "source_type": _infer_source_type(file_extension)
        }
        for attempt in range(2):
            try:
                converter = _get_docling_converter()
                result = converter.convert(tmp_path)
                document = getattr(result, "document", None)
                if document is None:
                    raise DoclingProcessingError("Docling returned no document object.")
                
                elements: List[Dict[str, Any]] = []
                raw_elements = getattr(document, "elements", None)
                
                if raw_elements:
                    for element in raw_elements:
                        elements.append(_docling_element_to_dict(element))
                
                if not elements or not _has_textual_content(elements):
                    fallback_text = _export_docling_document_text(document)
                    if fallback_text:
                        elements = _build_elements_from_text(
                            fallback_text,
                            base_metadata=base_metadata
                        )
                
                if not elements or not _has_textual_content(elements):
                    raise DoclingProcessingError(
                        "Docling returned no extractable text for this document."
                    )
                
                return _prepare_elements(elements, filename)
            except Exception as exc:
                last_exc = exc
                error_message = str(exc)
                missing_processor_config = "Missing processor config file" in error_message
                if missing_processor_config and attempt == 0:
                    # Cached download is missing files; clear cache and retry once.
                    _clear_docling_model_cache()
                    continue
                raise DoclingProcessingError(
                    f"Docling failed to process '{filename}': {error_message}"
                ) from exc
        
        if last_exc:
            raise DoclingProcessingError(
                f"Docling failed to process '{filename}' after retry: {last_exc}"
            ) from last_exc
        raise DoclingProcessingError(
            f"Docling failed to process '{filename}' due to an unknown error."
        )
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


def get_unstructured_api_key() -> Optional[str]:
    """
    Get the Unstructured API key from environment variable.
    
    Returns:
        Optional[str]: The API key if set, None otherwise
        
    Raises:
        UnstructuredServiceError: If API key is not set
    """
    api_key = os.getenv("UNSTRUCTURED_API_KEY")
    if not api_key:
        raise UnstructuredServiceError(
            "UNSTRUCTURED_API_KEY environment variable is not set. "
            "Please set it to use document processing functionality."
        )
    return api_key


def validate_file(file_bytes: bytes, filename: str) -> tuple[bool, Optional[str]]:
    """
    Validate uploaded file format and size.
    
    Args:
        file_bytes: The file content as bytes
        filename: The name of the file
        
    Returns:
        tuple[bool, Optional[str]]: (is_valid, error_message)
    """
    # Check file size
    if len(file_bytes) > MAX_FILE_SIZE:
        size_mb = len(file_bytes) / (1024 * 1024)
        return False, f"File '{filename}' is too large ({size_mb:.2f}MB). Maximum size is 10MB."
    
    file_ext = os.path.splitext(filename.lower())[1]
    
    if file_ext not in SUPPORTED_UPLOAD_EXTENSIONS:
        readable_exts = ", ".join(sorted(SUPPORTED_UPLOAD_EXTENSIONS))
        return False, (
            f"File '{filename}' has unsupported format '{file_ext}'. "
            f"Supported formats: {readable_exts}"
        )
    
    return True, None


def _create_unstructured_client(api_key: str, disable_ssl_verify: bool = False):
    """
    Create an UnstructuredClient instance - simplified to match working test file.
    
    Args:
        api_key: The API key for authentication
        disable_ssl_verify: Whether to disable SSL verification (for testing only, not used with SDK)
        
    Returns:
        UnstructuredClient: Configured client instance
    """
    if USE_OFFICIAL_SDK:
        try:
            # Simple initialization matching the working test file
            client = UnstructuredClient(
                api_key_auth=api_key
            )
            return client
        except Exception as e:
            raise UnstructuredServiceError(
                f"Failed to initialize UnstructuredClient: {str(e)}. "
                "Please check the SDK documentation for the correct initialization method."
            )
    else:
        return None


def _create_requests_session(disable_ssl_verify: bool = False) -> requests.Session:
    """
    Create a requests session with retry logic and SSL configuration.
    Fallback method when official SDK is not available.
    
    Returns:
        requests.Session: Configured session with retry logic
    """
    # Always create a session (needed for fallback even if SDK is available)
    session = requests.Session()
    
    # Configure retry strategy
    retry_strategy = Retry(
        total=3,  # Total number of retries
        backoff_factor=1,  # Wait 1, 2, 4 seconds between retries
        status_forcelist=[429, 500, 502, 503, 504],  # HTTP status codes to retry on
        allowed_methods=["POST", "GET"],  # Methods to retry
        raise_on_status=False  # Don't raise on status, handle it manually
    )
    
    # Mount adapter with retry strategy
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    
    if disable_ssl_verify and InsecureRequestWarning is not None:
        warnings.filterwarnings('ignore', category=InsecureRequestWarning)
    
    return session


def process_document(
    file_bytes: bytes,
    filename: str,
    strategy: str = "fast",
    disable_ssl_verify: bool = False
) -> List[Dict[str, Any]]:
    """
    Process a single document using Unstructured Serverless API.
    
    Args:
        file_bytes: The file content as bytes
        filename: The name of the file
        strategy: Processing strategy (default: "fast")
                  Options: "fast", "hi_res", "ocr_only", "auto"
        disable_ssl_verify: Whether to disable SSL verification (for testing only)
        
    Returns:
        List[Dict[str, Any]]: Structured output from the partition pipeline
        
    Raises:
        UnstructuredServiceError: If processing fails
    """
    # Validate file
    is_valid, error_message = validate_file(file_bytes, filename)
    if not is_valid:
        raise UnstructuredServiceError(error_message)
    
    # Prefer Docling for conversion to keep tables and visual blocks intact.
    if HAS_DOCLING:
        try:
            return process_document_with_docling(file_bytes, filename)
        except DoclingProcessingError as exc:
            raise UnstructuredServiceError(str(exc))
    
    if not USE_UNSTRUCTURED_API_FALLBACK:
        raise UnstructuredServiceError(
            "Docling is not available in this environment. Install docling>=2.0.0 "
            "or set REQVIBE_USE_UNSTRUCTURED_API=1 to explicitly enable the legacy Unstructured API."
        )
    
    # Legacy Unstructured API path (kept for backwards compatibility / opt-in fallback).
    api_key = get_unstructured_api_key()
    
    # Use local variable for SSL verification setting (may be updated if SSL errors detected)
    should_disable_ssl = disable_ssl_verify
    
    # Use official SDK if available (recommended)
    use_sdk = USE_OFFICIAL_SDK
    
    if use_sdk:
        try:
            client = _create_unstructured_client(api_key, disable_ssl_verify=should_disable_ssl)
            
            # Call partition API using SDK - matching working test file pattern
            try:
                # Map strategy string to Strategy enum
                strategy_map = {
                    "fast": shared.Strategy.FAST,
                    "hi_res": shared.Strategy.HI_RES,
                    "ocr_only": shared.Strategy.OCR_ONLY,
                    "auto": shared.Strategy.AUTO,
                }
                strategy_enum = strategy_map.get(strategy.lower(), shared.Strategy.AUTO)
                
                # Use dictionary-based request structure matching the working test file
                # The SDK accepts raw bytes directly (not BytesIO)
                # Based on validation errors, it expects bytes, IO, or BufferedReader
                req = {
                    "partition_parameters": {
                        "files": {
                            "content": file_bytes,  # Pass raw bytes directly
                            "file_name": filename,
                        },
                        "strategy": strategy_enum,
                    }
                }
                
                # Call partition method with dictionary request (synchronous)
                result = client.general.partition(request=req)
                
                # Extract elements from response - direct access like test file
                if hasattr(result, 'elements') and result.elements:
                    elements = result.elements
                    # Convert elements to list of dicts
                    element_dicts = []
                    for element in elements:
                        if isinstance(element, dict):
                            element_dicts.append(element)
                        elif hasattr(element, '__dict__'):
                            # Convert object to dict
                            element_dict = vars(element)
                            element_dicts.append(element_dict)
                        else:
                            # Convert to string representation
                            element_dicts.append({"text": str(element)})
                    return _prepare_elements(element_dicts, filename)
                else:
                    return []
                
            except errors.UnstructuredClientError as e:
                # Handle SDK-specific errors
                raise UnstructuredServiceError(
                    f"Unstructured API error: {str(e)}"
                )
            except (AttributeError, TypeError) as e:
                # If partition method structure is different, try alternative
                raise UnstructuredServiceError(
                    f"SDK API structure may have changed: {str(e)}. "
                    f"Please check unstructured-client documentation for the correct usage."
                )
                
        except UnstructuredServiceError:
            raise
        except Exception as e:
            # Check if this is an SSL/connection error from httpx (used by SDK)
            ssl_error_detected = False
            error_str = str(e).lower()
            
            # Check for SSL-related errors
            if HAS_HTTPX:
                # Check if it's an httpx.ConnectError
                if isinstance(e, (httpx.ConnectError, httpx.ConnectTimeout)):
                    ssl_error_detected = True
                # Also check for SSL-related error messages
                elif any(ssl_term in error_str for ssl_term in [
                    'ssl', 'tls', 'certificate', 'unexpected_eof', 
                    'eof occurred', 'protocol violation'
                ]):
                    ssl_error_detected = True
            elif any(ssl_term in error_str for ssl_term in [
                'ssl', 'tls', 'certificate', 'unexpected_eof', 
                'eof occurred', 'protocol violation'
            ]):
                ssl_error_detected = True
            
            # If SDK fails, fall back to requests method
            # Log the error but don't raise yet - try requests method
            if ssl_error_detected:
                print(f"Warning: SSL/connection error with SDK: {str(e)}. Falling back to requests method with SSL verification disabled...")
                # Automatically disable SSL verification for fallback
                should_disable_ssl = True
            else:
                print(f"Warning: SDK failed: {str(e)}. Falling back to requests method...")
            use_sdk = False  # Force fallback
    
    # Fallback to requests method if SDK is not available or failed
    if not use_sdk:
        # Create session for requests - disable SSL verification if needed
        session = _create_requests_session(disable_ssl_verify=should_disable_ssl)
        if session is None:
            raise UnstructuredServiceError(
                f"Failed to create requests session. "
                "Please ensure requests and urllib3 are properly installed."
            )
        
        # Prepare request according to official documentation
        # API key should be passed as 'unstructured-api-key' header, not Authorization Bearer
        # Reference: https://docs.unstructured.io/api-reference/partition/overview
        headers = {
            "unstructured-api-key": api_key,
            "accept": "application/json",
        }
        
        # Determine file type from extension
        file_ext = os.path.splitext(filename.lower())[1]
        content_type_map = {
            '.pdf': 'application/pdf',
            '.docx': 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
            '.xlsx': 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
            '.pptx': 'application/vnd.openxmlformats-officedocument.presentationml.presentation',
            '.md': 'text/markdown',
            '.markdown': 'text/markdown',
            '.adoc': 'text/asciidoc',
            '.asciidoc': 'text/asciidoc',
            '.html': 'text/html',
            '.htm': 'text/html',
            '.xhtml': 'application/xhtml+xml',
            '.csv': 'text/csv',
            '.png': 'image/png',
            '.jpg': 'image/jpeg',
            '.jpeg': 'image/jpeg',
            '.tif': 'image/tiff',
            '.tiff': 'image/tiff',
            '.bmp': 'image/bmp',
            '.webp': 'image/webp',
            '.reqif': 'application/reqif+xml',
        }
        content_type = content_type_map.get(file_ext, 'application/octet-stream')
        
        # Prepare files and data for multipart/form-data request
        files = {
            'files': (filename, file_bytes, content_type)
        }
        
        # Form data parameters according to official documentation
        # The curl example shows: content_type, strategy, output_format
        # Reference: https://docs.unstructured.io/api-reference/partition/overview
        data = {
            'strategy': strategy,
            'output_format': 'application/json',  # As shown in curl example
        }
        
        try:
            # Make API request - configure SSL verification based on parameter
            # When should_disable_ssl is True, explicitly set verify=False
            verify_ssl = not should_disable_ssl
            
            response = session.post(
                UNSTRUCTURED_API_URL,
                headers=headers,
                files=files,
                data=data,
                timeout=300,
                verify=verify_ssl  # Explicitly set SSL verification
            )
            
            # Check for errors
            if response.status_code != 200:
                error_msg = f"API request failed with status {response.status_code}"
                try:
                    error_data = response.json()
                    if 'detail' in error_data:
                        error_msg += f": {error_data['detail']}"
                    elif 'message' in error_data:
                        error_msg += f": {error_data['message']}"
                except:
                    error_msg += f": {response.text[:200]}"
                raise UnstructuredServiceError(error_msg)
            
            # Parse response
            result = response.json()
            
            # The API returns a list of dictionaries with structured content
            if isinstance(result, list):
                return _prepare_elements(result, filename)
            elif isinstance(result, dict) and 'elements' in result:
                return _prepare_elements(result['elements'], filename)
            else:
                normalized = [result] if isinstance(result, dict) else result
                return _prepare_elements(normalized, filename)
                
        except requests.exceptions.SSLError as e:
            # If SSL error and SSL verification was enabled, retry with it disabled
            # This is a common issue with some networks/proxies - disable SSL verification as fallback
            if not should_disable_ssl:
                if InsecureRequestWarning is not None:
                    warnings.filterwarnings('ignore', category=InsecureRequestWarning)
                print(f"Warning: SSL error encountered. Retrying with SSL verification disabled...")
                try:
                    return process_document(file_bytes, filename, strategy, disable_ssl_verify=True)
                except Exception as retry_error:
                    raise UnstructuredServiceError(
                        f"SSL error while processing '{filename}': {str(e)}. "
                        f"Retry with SSL verification disabled also failed: {str(retry_error)}. "
                        "This may indicate a network or firewall issue."
                    )
            else:
                raise UnstructuredServiceError(
                    f"SSL error while processing '{filename}': {str(e)}. "
                    "Please check your network connection and SSL certificates."
                )
        except requests.exceptions.Timeout:
            raise UnstructuredServiceError(
                f"Request timeout while processing '{filename}'. "
                "The file may be too large or the server is taking too long to respond."
            )
        except requests.exceptions.RequestException as e:
            raise UnstructuredServiceError(
                f"Network error while processing '{filename}': {str(e)}"
            )
        except json.JSONDecodeError as e:
            raise UnstructuredServiceError(
                f"Failed to parse API response for '{filename}': {str(e)}"
            )
        except Exception as e:
            raise UnstructuredServiceError(
                f"Unexpected error while processing '{filename}': {str(e)}"
            )


def create_single_document_json(filename: str, elements: List[Dict[str, Any]], file_size: int) -> Dict[str, Any]:
    """Create a single-document JSON payload compatible with GraphRAG indexing."""
    file_extension = os.path.splitext(filename)[1].lower()
    source_type = _infer_source_type(file_extension)
    return {
        'documents': [
            {
                'filename': filename,
                'elements': elements,
                'element_count': len(elements),
                'file_size': file_size,
                'source_extension': file_extension,
                'source_type': source_type
            }
        ],
        'total_elements': len(elements),
        'total_files': 1,
        'total_size': file_size
    }


def process_multiple_documents(
    files: List[tuple[bytes, str]],
    strategy: str = "fast",
    disable_ssl_verify: bool = False,
    return_individual: bool = False
) -> Union[Dict[str, Any], Tuple[Dict[str, Any], List[Dict[str, Any]]]]:
    """
    Process multiple documents and combine their outputs.
    
    Args:
        files: List of tuples (file_bytes, filename)
        strategy: Processing strategy (default: "fast")
        
    Returns:
        Dict[str, Any]: Combined structured output with metadata
        {
            'documents': [
                {
                    'filename': str,
                    'elements': List[Dict],
                    'element_count': int,
                    'file_size': int
                }
            ],
            'total_elements': int,
            'total_files': int,
            'total_size': int
        }
        
    Raises:
        UnstructuredServiceError: If processing fails
    """
    if not files:
        raise UnstructuredServiceError("No files provided for processing")
    
    # Validate total size
    total_size = sum(len(file_bytes) for file_bytes, _ in files)
    if total_size > MAX_FILE_SIZE:
        total_size_mb = total_size / (1024 * 1024)
        raise UnstructuredServiceError(
            f"Total combined file size ({total_size_mb:.2f}MB) exceeds the maximum limit of 10MB."
        )
    
    results = {
        'documents': [],
        'total_elements': 0,
        'total_files': len(files),
        'total_size': total_size
    }
    
    individual_jsons: List[Dict[str, Any]] = []
    
    # Process each file
    for file_bytes, filename in files:
        try:
            elements = process_document(
                file_bytes, 
                filename, 
                strategy=strategy,
                disable_ssl_verify=disable_ssl_verify
            )
            
            document_result = {
                'filename': filename,
                'elements': elements,
                'element_count': len(elements),
                'file_size': len(file_bytes),
                'source_extension': os.path.splitext(filename)[1].lower(),
                'source_type': _infer_source_type(os.path.splitext(filename)[1].lower())
            }
            
            results['documents'].append(document_result)
            results['total_elements'] += len(elements)
            
            if return_individual:
                individual_jsons.append(create_single_document_json(filename, elements, len(file_bytes)))
            
        except UnstructuredServiceError as e:
            # Re-raise with filename context
            raise UnstructuredServiceError(f"Error processing '{filename}': {str(e)}")
    
    if return_individual:
        return results, individual_jsons
    return results


def format_structured_output(result: Dict[str, Any]) -> str:
    """
    Format the structured output as a readable text string.
    
    Args:
        result: The combined structured output from process_multiple_documents
        
    Returns:
        str: Formatted text representation
    """
    output_lines = []
    output_lines.append(f"Processed {result['total_files']} file(s)")
    output_lines.append(f"Total elements: {result['total_elements']}")
    output_lines.append(f"Total size: {result['total_size'] / 1024:.2f} KB")
    output_lines.append("")
    
    for doc in result['documents']:
        output_lines.append(f"=== {doc['filename']} ===")
        output_lines.append(f"Elements: {doc['element_count']}, Size: {doc['file_size'] / 1024:.2f} KB")
        output_lines.append("")
        
        for i, element in enumerate(doc['elements'], 1):
            element_type = element.get('type', 'unknown')
            text = element.get('text', '')
            
            # Truncate long text for display
            if len(text) > 500:
                text = text[:500] + "..."
            
            output_lines.append(f"Element {i} ({element_type}):")
            output_lines.append(text)
            output_lines.append("")
    
    return "\n".join(output_lines)


