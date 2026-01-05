import sys
import pickle
import numpy as np
from pathlib import Path
import html

# Set numpy print options to show full arrays without truncation
np.set_printoptions(threshold=np.inf, linewidth=np.inf, edgeitems=np.inf)

# Add project root to Python path if needed (uncomment and adjust if necessary)
# project_root = Path(__file__).parent.parent.parent
# sys.path.insert(0, str(project_root))

def format_value(v, indent="    ", content_indent=None):
    """Format a single value with proper indentation"""
    if content_indent is None:
        content_indent = indent + "    "  # Extra indent for content
    
    if isinstance(v, np.ndarray):
        shape_str = f"shape={v.shape}"
        dtype_str = f"dtype={v.dtype}"
        result = f"ndarray ({shape_str}, {dtype_str})"
        
        # Add min, max, mean for numeric arrays
        if v.size > 0 and np.issubdtype(v.dtype, np.number):
            try:
                result += f" min: {v.min()}, max: {v.max()}, mean: {v.mean():.5f}"
            except:
                pass
        
        # Format array contents with proper indentation on new line
        array_str = np.array2string(v, separator=' ', max_line_width=np.inf, threshold=np.inf)
        # Replace newlines with indented newlines
        lines = array_str.split('\n')
        indented_lines = [content_indent + line.strip() for line in lines if line.strip()]
        result += "\n" + "\n".join(indented_lines)
        return result
    
    elif isinstance(v, (bytes, bytearray)):
        # Display bytes content like print() would, on new line with indentation
        bytes_repr = repr(v)
        result = f"bytes (len={len(v)})\n{content_indent}{bytes_repr}"
        return result
    
    elif isinstance(v, str):
        return f"(len={len(v)})'{v}'"
    
    elif isinstance(v, dict):
        if len(v) == 0:
            return f"dict (len=0) {{}}"
        result = f"dict (len={len(v)}) {{\n"
        for k, val in v.items():
            item_indent = indent + "    "
            formatted_val = format_value(val, item_indent)
            # Wrap dict keys in <strong> tag for bold
            result += f"{item_indent}<strong>'{k}'</strong>: {formatted_val}\n"
        result += indent + "}"
        return result
    
    elif isinstance(v, (list, tuple)):
        if len(v) == 0:
            return f"{type(v).__name__} (len=0) {{}}"
        result = f"{type(v).__name__} (len={len(v)}) {{\n"
        for i, item in enumerate(v):
            item_indent = indent + "    "
            formatted_item = format_value(item, item_indent)
            result += f"{item_indent}[{i}]: {formatted_item}\n"
        result += indent + "}"
        return result
    
    elif hasattr(v, '__reduce__'):
        # Handle special objects like convert_to_tensor
        try:
            attrs_count = len([x for x in dir(v) if not x.startswith('__')])
        except:
            attrs_count = 0
        
        result = f"{type(v).__name__} (attrs={attrs_count}) {{\n"
        item_indent = indent + "    "
        
        # Try to get args and kwargs from __reduce__
        try:
            reduce_result = v.__reduce__()
            # Always try to show args if available
            if len(reduce_result) > 1:
                args = reduce_result[1]
                if args is not None:
                    formatted_args = format_value(args, item_indent)
                    result += f"{item_indent}<strong>args</strong>: {formatted_args}\n"
            # Always show kwargs, even if empty
            kwargs = {}
            if len(reduce_result) > 2:
                kwargs = reduce_result[2] if reduce_result[2] is not None else {}
            formatted_kwargs = format_value(kwargs, item_indent)
            result += f"{item_indent}<strong>kwargs</strong>: {formatted_kwargs}\n"
        except:
            # If __reduce__ fails, still show empty kwargs
            formatted_kwargs = format_value({}, item_indent)
            result += f"{item_indent}<strong>kwargs</strong>: {formatted_kwargs}\n"
        
        result += indent + "}"
        return result
    
    else:
        return f"{type(v).__name__} ({v})"

def summarize(obj, item_num=1, total_items=1):
    # Build output as a single string with bold field names
    output_lines = []
    output_lines.append(f"Item {item_num}/{total_items}:")
    output_lines.append("")
    
    if isinstance(obj, dict):
        output_lines.append(f"dict (len={len(obj)}) {{")
        for k, v in obj.items():
            formatted = format_value(v, "    ")
            # Wrap field name in <strong> tag for bold
            output_lines.append(f"    <strong>'{k}'</strong>: {formatted}")
        output_lines.append("}")
    
    # Join and wrap in <pre> tag with Consolas font to preserve whitespace
    output = "\n".join(output_lines)
    print(f'<pre style="font-family: Consolas, monospace;">{output}</pre>')

def process_pickle_file(file_path):
    """Process and display pickle file contents"""
    with open(file_path, "rb") as f:
        data = pickle.load(f)
    summarize(data)

if __name__ == "__main__":
    if len(sys.argv) >= 3:
        # Plugin mode: receives file_type and file_path as arguments
        file_type = int(sys.argv[1])
        file_path = sys.argv[2]
        
        if file_type == 1:  # Pickle file
            process_pickle_file(file_path)
    else:
        # Standalone mode: use default file for testing
        pkl_file = "/D2/NXT/vlm/miaobinqi/DriveMoE/b2d_dynamic_camera/val/Accident_Town05_Route218_Weather10_step4.pkl"
        process_pickle_file(pkl_file)
