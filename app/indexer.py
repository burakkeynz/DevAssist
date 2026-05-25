# Importing required libraries for AST parsing and TreeRAG chunking
import ast
import os
import re
import hashlib
import json
import logging
from pathlib import Path
from typing import Optional, List, Dict, Any

# Configuring logging for indexer operations
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Defining excluded directories for codebase scanning
EXCLUDED_DIRS = {"venv", "__pycache__", ".git", "node_modules", ".env", ".idea", ".vscode"}
SUPPORTED_EXTENSIONS = {".py", ".txt", ".md"}


# Defining parent-level file summary chunk
def create_parent_chunk(file_path: Path, source_code: str) -> Dict[str, Any]:
    file_id = hashlib.md5(str(file_path).encode()).hexdigest()
    lines = source_code.splitlines()
    return {
        "chunk_id": f"parent_{file_id}",
        "type": "parent",
        "file_path": str(file_path),
        "content": source_code[:500],
        "total_lines": len(lines),
        "children": [],
        "metadata": {
            "file_name": file_path.name,
            "extension": file_path.suffix,
            "size_bytes": len(source_code.encode())
        }
    }


# Extracting child function and class blocks using AST traversal
def extract_python_chunks(file_path: Path, source_code: str, parent_id: str) -> List[Dict[str, Any]]:
    child_chunks: List[Dict[str, Any]] = []
    try:
        tree = ast.parse(source_code)
    except SyntaxError as e:
        logger.warning(f"Skipping unparseable file due to syntax error: {file_path} — {e}")
        return []

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            try:
                node_source = ast.get_source_segment(source_code, node)
                if not node_source:
                    continue

                end_line = node.end_lineno if node.end_lineno is not None else node.lineno
                line_count = end_line - node.lineno + 1
                node_type = type(node).__name__
                chunk_id = hashlib.md5(
                    f"{file_path}_{node.name}_{node.lineno}".encode()
                ).hexdigest()

                child_chunk: Dict[str, Any] = {
                    "chunk_id": f"child_{chunk_id}",
                    "type": "child",
                    "node_type": node_type,
                    "name": node.name,
                    "file_path": str(file_path),
                    "content": node_source,
                    "start_line": node.lineno,
                    "end_line": end_line,
                    "parent_id": parent_id,
                    "metadata": {
                        "file_name": file_path.name,
                        "node_type": node_type,
                        "function_name": node.name,
                        "line_count": line_count
                    }
                }
                child_chunks.append(child_chunk)
                logger.info(f"Extracting Python chunk: {node_type} '{node.name}' from {file_path.name}")

            except Exception as e:
                logger.warning(f"Skipping node due to extraction error: {node.name} — {e}")
                continue

    return child_chunks


# Extracting section-based chunks from repomix .txt output
def extract_txt_chunks(file_path: Path, content: str, parent_id: str) -> List[Dict[str, Any]]:
    child_chunks: List[Dict[str, Any]] = []

    # Splitting repomix output by file separator pattern
    section_pattern = re.compile(
        r'<file path="([^"]+)">\s*(.*?)</file>',
        re.DOTALL
    )
    matches = section_pattern.findall(content)

    if matches:
        # Parsing repomix-formatted sections
        for section_path, section_content in matches:
            section_content = section_content.strip()
            if not section_content:
                continue

            chunk_id = hashlib.md5(
                f"{file_path}_{section_path.strip()}".encode()
            ).hexdigest()

            child_chunk: Dict[str, Any] = {
                "chunk_id": f"child_{chunk_id}",
                "type": "child",
                "node_type": "repomix_section",
                "name": section_path.strip(),
                "file_path": str(file_path),
                "content": section_content[:2000],
                "start_line": 0,
                "end_line": len(section_content.splitlines()),
                "parent_id": parent_id,
                "metadata": {
                    "file_name": file_path.name,
                    "node_type": "repomix_section",
                    "function_name": section_path.strip(),
                    "line_count": len(section_content.splitlines())
                }
            }
            child_chunks.append(child_chunk)
            logger.info(f"Extracting repomix section: '{section_path.strip()}' from {file_path.name}")
    else:
        # Falling back to fixed-size chunking for plain text
        chunk_size = 1500
        lines = content.splitlines()
        for i in range(0, len(lines), chunk_size):
            chunk_lines = lines[i:i + chunk_size]
            chunk_content = "\n".join(chunk_lines).strip()
            if not chunk_content:
                continue

            chunk_id = hashlib.md5(
                f"{file_path}_chunk_{i}".encode()
            ).hexdigest()

            child_chunk = {
                "chunk_id": f"child_{chunk_id}",
                "type": "child",
                "node_type": "text_chunk",
                "name": f"{file_path.stem}_chunk_{i}",
                "file_path": str(file_path),
                "content": chunk_content,
                "start_line": i,
                "end_line": i + len(chunk_lines),
                "parent_id": parent_id,
                "metadata": {
                    "file_name": file_path.name,
                    "node_type": "text_chunk",
                    "function_name": f"{file_path.stem}_chunk_{i}",
                    "line_count": len(chunk_lines)
                }
            }
            child_chunks.append(child_chunk)
            logger.info(f"Extracting text chunk {i} from {file_path.name}")

    return child_chunks


# Extracting header-based chunks from markdown files
def extract_md_chunks(file_path: Path, content: str, parent_id: str) -> List[Dict[str, Any]]:
    child_chunks: List[Dict[str, Any]] = []

    # Splitting markdown by headers...
    header_pattern = re.compile(r'^(#{1,3}\s+.+)$', re.MULTILINE)
    sections = header_pattern.split(content)

    current_header = file_path.stem
    buffer = []

    for part in sections:
        if header_pattern.match(part):
            if buffer:
                section_content = "\n".join(buffer).strip()
                if section_content:
                    chunk_id = hashlib.md5(
                        f"{file_path}_{current_header}".encode()
                    ).hexdigest()
                    child_chunks.append({
                        "chunk_id": f"child_{chunk_id}",
                        "type": "child",
                        "node_type": "md_section",
                        "name": current_header,
                        "file_path": str(file_path),
                        "content": f"{current_header}\n{section_content}"[:2000],
                        "start_line": 0,
                        "end_line": len(section_content.splitlines()),
                        "parent_id": parent_id,
                        "metadata": {
                            "file_name": file_path.name,
                            "node_type": "md_section",
                            "function_name": current_header,
                            "line_count": len(section_content.splitlines())
                        }
                    })
                    logger.info(f"Extracting markdown section: '{current_header}' from {file_path.name}")
            current_header = part.strip()
            buffer = []
        else:
            buffer.append(part)

    # Flushing remaining buffer
    if buffer:
        section_content = "\n".join(buffer).strip()
        if section_content:
            chunk_id = hashlib.md5(
                f"{file_path}_{current_header}_last".encode()
            ).hexdigest()
            child_chunks.append({
                "chunk_id": f"child_{chunk_id}",
                "type": "child",
                "node_type": "md_section",
                "name": current_header,
                "file_path": str(file_path),
                "content": f"{current_header}\n{section_content}"[:2000],
                "start_line": 0,
                "end_line": len(section_content.splitlines()),
                "parent_id": parent_id,
                "metadata": {
                    "file_name": file_path.name,
                    "node_type": "md_section",
                    "function_name": current_header,
                    "line_count": len(section_content.splitlines())
                }
            })

    return child_chunks


# Processing single file based on extension type
def process_file(file_path: Path) -> Optional[Dict[str, Any]]:
    try:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            source_code = f.read()

        if not source_code.strip():
            logger.warning(f"Skipping empty file: {file_path}")
            return None

        logger.info(f"Processing file for TreeRAG indexing: {file_path}")

        parent_chunk = create_parent_chunk(file_path, source_code)

        ext = file_path.suffix.lower()
        if ext == ".py":
            child_chunks = extract_python_chunks(file_path, source_code, parent_chunk["chunk_id"])
        elif ext == ".txt":
            child_chunks = extract_txt_chunks(file_path, source_code, parent_chunk["chunk_id"])
        elif ext == ".md":
            child_chunks = extract_md_chunks(file_path, source_code, parent_chunk["chunk_id"])
        else:
            return None

        parent_chunk["children"] = [c["chunk_id"] for c in child_chunks]

        return {
            "parent": parent_chunk,
            "children": child_chunks
        }

    except Exception as e:
        logger.error(f"Failing to process file: {file_path} — {e}")
        return None


# Scanning codebase directory and building full TreeRAG chunk index
def index_codebase(codebase_path: str) -> Dict[str, Any]:
    base_path = Path(codebase_path)

    if not base_path.exists():
        logger.error(f"Codebase directory not found: {base_path}")
        return {"parents": [], "children": [], "stats": {}}

    all_parents: List[Dict[str, Any]] = []
    all_children: List[Dict[str, Any]] = []
    processed_files = 0
    skipped_files = 0

    # Walking directory tree and collecting supported files
    for root, dirs, files in os.walk(base_path):
        dirs[:] = [d for d in dirs if d not in EXCLUDED_DIRS]

        for file in files:
            file_path = Path(root) / file
            if file_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
                continue

            result = process_file(file_path)
            if result:
                all_parents.append(result["parent"])
                all_children.extend(result["children"])
                processed_files += 1
            else:
                skipped_files += 1

    logger.info(
        f"Completing codebase indexing — "
        f"processed: {processed_files} files, "
        f"skipped: {skipped_files} files, "
        f"total chunks: {len(all_parents) + len(all_children)}"
    )

    return {
        "parents": all_parents,
        "children": all_children,
        "stats": {
            "total_files": processed_files,
            "total_parents": len(all_parents),
            "total_children": len(all_children),
            "total_chunks": len(all_parents) + len(all_children)
        }
    }


# Saving indexed chunks to JSON output for debugging and verification
def save_index(index: Dict[str, Any], output_path: str = "index_debug.json") -> None:
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(index, f, indent=2, ensure_ascii=False)
    logger.info(f"Saving index snapshot to: {output_path}")


if __name__ == "__main__":
    logger.info("Starting TreeRAG indexer in standalone test mode...")
    index = index_codebase("codebase/")
    save_index(index)

    print("\n--- TreeRAG Index Summary ---")
    print(f"Total files processed : {index['stats']['total_files']}")
    print(f"Total parent chunks   : {index['stats']['total_parents']}")
    print(f"Total child chunks    : {index['stats']['total_children']}")
    print(f"Total chunks          : {index['stats']['total_chunks']}")