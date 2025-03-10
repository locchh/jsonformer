"""
This module provides formatting utilities for JSON output in the JSONFormer library.
It includes functionality to pretty-print JSON structures with colored highlighting
for better readability in terminal output.
"""

# Import colored text functionality for terminal output
from termcolor import colored


def highlight_values(value):
    """
    Pretty-prints a JSON-like data structure with colored highlighting for values.
    
    This function recursively traverses through dictionaries, lists, and primitive values,
    formatting them with proper indentation and coloring. String values are properly quoted,
    and all primitive values are highlighted in green for better visibility.
    
    Args:
        value: Any JSON-serializable Python object (dict, list, str, int, etc.)
    
    Example:
        >>> data = {"name": "John", "scores": [95, 87, 91]}
        >>> highlight_values(data)
        {
          name: "John",
          scores: [
            95,
            87,
            91
          ]
        }
    """
    def recursive_print(obj, indent=0, is_last_element=True):
        """
        Recursively prints a JSON-like object with proper formatting.
        
        Args:
            obj: The object to print
            indent (int): Current indentation level
            is_last_element (bool): Whether this is the last element in its container
                                  (determines if a comma should be added)
        """
        if isinstance(obj, dict):
            print("{")
            last_key = list(obj.keys())[-1]
            for key, value in obj.items():
                print(f"{' ' * (indent + 2)}{key}: ", end="")
                recursive_print(value, indent + 2, key == last_key)
            print(f"{' ' * indent}}}", end=",\n" if not is_last_element else "\n")
        elif isinstance(obj, list):
            print("[")
            for index, value in enumerate(obj):
                print(f"{' ' * (indent + 2)}", end="")
                recursive_print(value, indent + 2, index == len(obj) - 1)
            print(f"{' ' * indent}]", end=",\n" if not is_last_element else "\n")
        else:
            # Handle primitive values - strings get quoted, all values are colored green
            if isinstance(obj, str):
                obj = f'"{obj}"'
            print(colored(obj, "green"), end=",\n" if not is_last_element else "\n")

    recursive_print(value)
