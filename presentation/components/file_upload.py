"""
File Upload Component for ReqVibe

This module provides a secure file upload component for Streamlit that:
- Supports PDF, Word (.docx), Excel (.xlsx), PowerPoint (.pptx), Markdown (.md),
  AsciiDoc (.adoc), HTML/XHTML, CSV, and common image formats (PNG/JPEG/TIFF/BMP/WEBP)
  alongside ReqIF (.reqif) and vectorized JSON documents
- Allows multiple file uploads
- Enforces 10MB total size limit
- Shows clear error messages
- Processes files locally using Docling to preserve complex layouts
- Detects and uses vectorized JSON documents directly without reprocessing
"""

import streamlit as st
from typing import List, Dict, Any, Optional, Tuple
import os
import json
import io
import zipfile
from datetime import datetime

# Domain Services
from domain.documents.unstructured import (
    process_multiple_documents,
    validate_file,
    UnstructuredServiceError,
    MAX_FILE_SIZE,
    create_single_document_json
)


def render_file_upload():
    """
    Render the file upload component in the sidebar.
    
    This component allows users to upload documents for processing
    and automatically processes them locally via Docling.
    """
    st.markdown(
        "<div style='margin-top: 1.5rem; margin-bottom: 1rem;'>"
        "<h3 style='color: #8e8ea0; font-size: 0.9rem; font-weight: 600; "
        "text-transform: uppercase; letter-spacing: 0.5px;'>Document Upload</h3>"
        "</div>",
        unsafe_allow_html=True
    )
    
    # Check authentication
    if not st.session_state.authenticated:
        st.info("Please log in to upload documents")
        return
    
    # File uploader
    supported_types = [
        "pdf", "docx", "xlsx", "pptx", "md", "adoc", "html", "xhtml",
        "csv", "png", "jpg", "jpeg", "tiff", "bmp", "webp", "reqif", "json"
    ]
    uploaded_files = st.file_uploader(
        "Upload Documents",
        type=supported_types,
        accept_multiple_files=True,
        help=(
            "Upload PDF, DOCX, XLSX, PPTX, Markdown, AsciiDoc, HTML/XHTML, CSV, "
            "PNG/JPEG/TIFF/BMP/WEBP, ReqIF, or vectorized JSON files. Maximum 10MB total."
        ),
        key="file_uploader"
    )
    
    # Display file information
    if uploaded_files:
        # Calculate total size
        total_size = sum(file.size for file in uploaded_files)
        total_size_mb = total_size / (1024 * 1024)
        
        # Validate total size
        if total_size > MAX_FILE_SIZE:
            st.error(
                f"Total file size ({total_size_mb:.2f}MB) exceeds the maximum limit of 10MB. "
                "Please upload smaller files or reduce the number of files."
            )
            return
        
        # Display uploaded files info
        st.markdown("**Uploaded Files:**")
        file_info_html = "<div style='margin-bottom: 0.75rem;'>"
        for i, file in enumerate(uploaded_files, 1):
            file_size_kb = file.size / 1024
            file_ext = os.path.splitext(file.name)[1].upper()
            file_info_html += (
                f"<div style='color: #8e8ea0; font-size: 0.75rem; margin-bottom: 0.25rem;'>"
                f"{i}. {file.name} ({file_size_kb:.2f} KB, {file_ext})"
                f"</div>"
            )
        file_info_html += f"<div style='color: #8e8ea0; font-size: 0.75rem; margin-top: 0.25rem;'>"
        file_info_html += f"Total: {total_size_mb:.2f} MB / 10 MB</div>"
        file_info_html += "</div>"
        st.markdown(file_info_html, unsafe_allow_html=True)
        
        # Validate individual files (skip JSON files as they're validated separately)
        invalid_files = []
        for file in uploaded_files:
            file_ext = os.path.splitext(file.name.lower())[1]
            if file_ext == '.json':
                # JSON files are validated separately in process_uploaded_files
                continue
            file_bytes = file.getvalue()
            is_valid, error_msg = validate_file(file_bytes, file.name)
            if not is_valid:
                invalid_files.append((file.name, error_msg))
        
        if invalid_files:
            st.error("**Invalid Files:**")
            for filename, error_msg in invalid_files:
                st.error(f"- {filename}: {error_msg}")
            return
        
        # Process button
        if st.button("Process Documents", use_container_width=True, key="process_documents_button"):
            process_uploaded_files(uploaded_files)
    
    # Display processing results if available
    if st.session_state.get("document_processing_results"):
        display_processing_results()


def is_valid_vectorized_json(file_bytes: bytes) -> Tuple[bool, Optional[Dict[str, Any]], Optional[str]]:
    """
    Check if a JSON file is a valid vectorized document from previous processing.
    
    Args:
        file_bytes: The file content as bytes
        
    Returns:
        Tuple of (is_valid, parsed_data, error_message)
    """
    try:
        # Try to parse JSON
        file_content = file_bytes.decode('utf-8')
        data = json.loads(file_content)
        
        # Check if it has the expected structure from process_multiple_documents
        if not isinstance(data, dict):
            return False, None, "JSON file is not a dictionary"
        
        # Check for required top-level keys
        required_keys = ['documents', 'total_elements', 'total_files', 'total_size']
        missing_keys = [key for key in required_keys if key not in data]
        if missing_keys:
            return False, None, f"Missing required keys: {', '.join(missing_keys)}"
        
        # Validate documents structure
        if not isinstance(data['documents'], list):
            return False, None, "'documents' must be a list"
        
        # Validate each document in the list
        for i, doc in enumerate(data['documents']):
            if not isinstance(doc, dict):
                return False, None, f"Document {i} is not a dictionary"
            
            doc_required_keys = ['filename', 'elements', 'element_count', 'file_size']
            missing_doc_keys = [key for key in doc_required_keys if key not in doc]
            if missing_doc_keys:
                return False, None, f"Document {i} missing required keys: {', '.join(missing_doc_keys)}"
            
            # Check that elements is a list
            if not isinstance(doc['elements'], list):
                return False, None, f"Document {i} 'elements' must be a list"
        
        # Validate numeric fields
        if not isinstance(data['total_elements'], int) or data['total_elements'] < 0:
            return False, None, "'total_elements' must be a non-negative integer"
        if not isinstance(data['total_files'], int) or data['total_files'] < 0:
            return False, None, "'total_files' must be a non-negative integer"
        if not isinstance(data['total_size'], int) or data['total_size'] < 0:
            return False, None, "'total_size' must be a non-negative integer"
        
        return True, data, None
        
    except json.JSONDecodeError as e:
        return False, None, f"Invalid JSON format: {str(e)}"
    except UnicodeDecodeError as e:
        return False, None, f"File is not valid UTF-8: {str(e)}"
    except Exception as e:
        return False, None, f"Error validating JSON: {str(e)}"


def combine_vectorized_results(results_list: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Combine multiple vectorized document results into a single result.
    
    Args:
        results_list: List of result dictionaries from process_multiple_documents or vectorized JSON
        
    Returns:
        Combined result dictionary
    """
    combined = {
        'documents': [],
        'total_elements': 0,
        'total_files': 0,
        'total_size': 0
    }
    
    for result in results_list:
        if 'documents' in result:
            combined['documents'].extend(result['documents'])
            combined['total_elements'] += result.get('total_elements', 0)
            combined['total_files'] += result.get('total_files', 0)
            combined['total_size'] += result.get('total_size', 0)
    
    return combined


def process_uploaded_files(uploaded_files: List):
    """
    Process uploaded files using Docling or reuse vectorized JSON directly.
    
    This function:
    1. Separates JSON files from other files
    2. Validates JSON files to check if they're vectorized documents
    3. Processes non-JSON files with Docling
    4. Combines all vectorized JSON files (from upload + from processing) for Q&A
    
    Args:
        uploaded_files: List of uploaded file objects from Streamlit
    """
    # Separate JSON files from other files
    json_files = []
    non_json_files = []
    
    for file in uploaded_files:
        file_ext = os.path.splitext(file.name.lower())[1]
        if file_ext == '.json':
            json_files.append(file)
        else:
            non_json_files.append(file)
    
    # Validate and collect vectorized JSON files
    vectorized_results = []
    invalid_json_files = []
    
    for json_file in json_files:
        file_bytes = json_file.getvalue()
        is_valid, parsed_data, error_msg = is_valid_vectorized_json(file_bytes)
        
        if is_valid and parsed_data:
            vectorized_results.append(parsed_data)
            st.info(f"✅ Using vectorized JSON: {json_file.name}")
        else:
            invalid_json_files.append((json_file.name, error_msg or "Invalid vectorized JSON format"))
    
    # Show errors for invalid JSON files
    if invalid_json_files:
        st.warning("**Invalid Vectorized JSON Files:**")
        for filename, error_msg in invalid_json_files:
            st.warning(f"- {filename}: {error_msg}")
        st.info("These files will be skipped. Please ensure they are valid vectorized JSON documents from previous processing.")
    
    # Process non-JSON files with Docling if any
    processed_results = None
    individual_jsons = []
    if non_json_files:
        files_to_process = []
        for file in non_json_files:
            file_bytes = file.getvalue()
            files_to_process.append((file_bytes, file.name))
        
        try:
            with st.spinner("Processing documents with Docling... This may take a few moments."):
                processed_results, individual_jsons = process_multiple_documents(
                    files_to_process,
                    strategy="fast",
                    return_individual=True
                )
        except UnstructuredServiceError as e:
            st.session_state.document_processing_error = str(e)
            st.error(f"Error processing documents: {str(e)}")
            if vectorized_results:
                st.info("Note: You can still use the uploaded vectorized JSON files for Q&A.")
            return
        except Exception as e:
            st.session_state.document_processing_error = str(e)
            st.error(f"Unexpected error: {str(e)}")
            if vectorized_results:
                st.info("Note: You can still use the uploaded vectorized JSON files for Q&A.")
            return
    
    # Combine all results (vectorized JSON + processed documents)
    all_results = []
    all_individual_jsons = []
    
    if vectorized_results:
        all_results.extend(vectorized_results)
        all_individual_jsons.extend(vectorized_results)
    
    if processed_results:
        all_results.append(processed_results)
        all_individual_jsons.extend(individual_jsons)
    
    # If we have no results at all, show error
    if not all_results:
        st.error("No valid documents to process. Please upload valid files or vectorized JSON documents.")
        return
    
    # Combine all results
    try:
        combined_results = combine_vectorized_results(all_results)
        
        # Store results in session state
        st.session_state.document_processing_results = combined_results
        st.session_state.document_processing_error = None
        
        # Persist individual JSONs for download
        per_doc_json: Dict[str, Dict[str, Any]] = {}
        for json_blob in all_individual_jsons:
            docs = json_blob.get('documents') or []
            if not docs:
                continue
            filename = docs[0].get('filename', 'document')
            safe_name = os.path.splitext(filename)[0] or 'document'
            per_doc_json[f"{safe_name}_processed.json"] = json_blob
        st.session_state.document_individual_jsons = per_doc_json
        
        # Also store formatted output for easy access
        from domain.documents.unstructured import format_structured_output
        st.session_state.document_processing_formatted = format_structured_output(combined_results)
        
        # Build GraphRAG index
        with st.spinner("Building knowledge graph index..."):
            from infrastructure.graphrag.service import build_graphrag_index
            graphrag_index = build_graphrag_index(combined_results)
            
            # Store GraphRAG index in session state (serialized)
            st.session_state.graphrag_index = graphrag_index.to_dict()
            st.session_state.graphrag_index_built = True
        
        # Show success message with details
        json_count = len(vectorized_results)
        processed_count = len(non_json_files) if non_json_files else 0
        
        if json_count > 0 and processed_count > 0:
            st.success(
                f"✅ Documents processed successfully! "
                f"Used {json_count} vectorized JSON file(s) and processed {processed_count} new file(s). "
                f"You can now ask questions about the documents."
            )
        elif json_count > 0:
            st.success(
                f"✅ Using {json_count} vectorized JSON file(s). "
                f"You can now ask questions about the documents."
            )
        else:
            st.success(f"✅ Documents processed successfully! You can now ask questions about the documents.")
        
        st.rerun()
        
    except Exception as e:
        st.session_state.document_processing_error = str(e)
        st.session_state.document_processing_results = None
        st.error(f"Error combining documents: {str(e)}")


def display_processing_results():
    """Display the results of document processing."""
    results = st.session_state.document_processing_results
    
    st.markdown("<div style='margin-top: 1rem;'>", unsafe_allow_html=True)
    st.markdown("**Processing Results:**")
    
    # Summary
    summary_html = (
        f"<div style='padding: 0.75rem; background-color: #343541; "
        f"border-radius: 6px; border: 1px solid #565869; margin-bottom: 1rem;'>"
        f"<div style='color: #8e8ea0; font-size: 0.75rem; margin-bottom: 0.25rem;'>"
        f"Summary</div>"
        f"<div style='color: #ececf1; font-size: 0.9rem;'>"
        f"Files: {results['total_files']} | "
        f"Elements: {results['total_elements']} | "
        f"Size: {results['total_size'] / 1024:.2f} KB"
        f"</div>"
        f"</div>"
    )
    st.markdown(summary_html, unsafe_allow_html=True)
    
    # Show formatted output in expander
    if st.session_state.get("document_processing_formatted"):
        with st.expander("View Structured Output", expanded=False):
            st.text(st.session_state.document_processing_formatted)
    
    # Allow downloading per-document JSONs as a ZIP
    if st.session_state.get("document_individual_jsons"):
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zf:
            for filename, json_blob in st.session_state.document_individual_jsons.items():
                zf.writestr(filename, json.dumps(json_blob, indent=2, ensure_ascii=False))
        zip_buffer.seek(0)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        st.download_button(
            label=f"Download Individual Results (ZIP - {len(st.session_state.document_individual_jsons)} files)",
            data=zip_buffer.getvalue(),
            file_name=f"docling_results_{timestamp}.zip",
            mime="application/zip",
            use_container_width=True,
            key="download_results_zip"
        )
    
    # Also provide the combined JSON for backward compatibility
    results_json = json.dumps(results, indent=2, ensure_ascii=False)
    st.download_button(
        label="Download Combined Results (JSON)",
        data=results_json,
        file_name="document_processing_results_combined.json",
        mime="application/json",
        use_container_width=True,
        key="download_results_combined"
    )
    
    # Note: GraphRAG is now automatically available for document-related questions
    # Users can ask questions directly without needing to send results to chat
    
    st.markdown("</div>", unsafe_allow_html=True)

