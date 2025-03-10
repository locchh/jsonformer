"""
JSONFormer: A powerful tool for generating structured JSON data using language models.
This module provides the core functionality for generating JSON objects that conform
to a specified JSON schema, using a pre-trained language model for generation.
"""

from typing import List, Union, Dict, Any

from jsonformer.logits_processors import (
    NumberStoppingCriteria,
    OutputNumbersTokens,
    StringStoppingCriteria,
)
from termcolor import cprint
from transformers import PreTrainedModel, PreTrainedTokenizer
import json

# Marker used to track the current generation position in the JSON structure
GENERATION_MARKER = "|GENERATION|"


class Jsonformer:
    """
    A class that generates JSON data using a language model while conforming to a specified schema.
    
    This class orchestrates the generation of JSON objects by breaking down the schema
    into individual components (strings, numbers, booleans, arrays, objects) and using
    specialized generation strategies for each type.
    
    Attributes:
        value (Dict[str, Any]): The current state of the generated JSON object
    """
    value: Dict[str, Any] = {}

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        json_schema: Dict[str, Any],
        prompt: str,
        *,
        debug: bool = False,
        max_array_length: int = 10,
        max_number_tokens: int = 6,
        temperature: float = 1.0,
        max_string_token_length: int = 10,
    ):
        """
        Initialize the Jsonformer with a model, tokenizer, schema, and generation parameters.
        
        Args:
            model: The pre-trained language model to use for generation
            tokenizer: The tokenizer corresponding to the model
            json_schema: The JSON schema that defines the structure to generate
            prompt: The input prompt to guide the generation
            debug: Whether to print debug information during generation
            max_array_length: Maximum number of elements in generated arrays
            max_number_tokens: Maximum tokens for generating numbers
            temperature: Sampling temperature for generation (higher = more random)
            max_string_token_length: Maximum tokens for generating strings
        """
        self.model = model
        self.tokenizer = tokenizer
        self.json_schema = json_schema
        self.prompt = prompt

        # Initialize the number generation logits processor
        self.number_logit_processor = OutputNumbersTokens(self.tokenizer, self.prompt)

        self.generation_marker = "|GENERATION|"
        self.debug_on = debug
        self.max_array_length = max_array_length

        self.max_number_tokens = max_number_tokens
        self.temperature = temperature
        self.max_string_token_length = max_string_token_length

    def debug(self, caller: str, value: str, is_prompt: bool = False):
        """Print debug information if debug mode is enabled."""
        if self.debug_on:
            if is_prompt:
                cprint(caller, "green", end=" ")
                cprint(value, "yellow")
            else:
                cprint(caller, "green", end=" ")
                cprint(value, "blue")

    def generate_number(self, temperature: Union[float, None] = None, iterations=0):
        """
        Generate a numeric value using the language model.
        
        Uses the NumberStoppingCriteria to ensure valid number generation and
        supports decimal numbers with controlled precision.
        
        Args:
            temperature: Optional override for generation temperature
            iterations: Number of retry attempts for invalid generations
            
        Returns:
            float: The generated number
            
        Raises:
            ValueError: If unable to generate a valid number after max retries
        """
        prompt = self.get_prompt()
        self.debug("[generate_number]", prompt, is_prompt=True)
        input_tokens = self.tokenizer.encode(prompt, return_tensors="pt").to(
            self.model.device
        )
        response = self.model.generate(
            input_tokens,
            max_new_tokens=self.max_number_tokens,
            num_return_sequences=1,
            logits_processor=[self.number_logit_processor],
            stopping_criteria=[
                NumberStoppingCriteria(self.tokenizer, len(input_tokens[0]))
            ],
            temperature=temperature or self.temperature,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        response = self.tokenizer.decode(response[0], skip_special_tokens=True)

        # Extract just the generated number from the full response
        response = response[len(prompt) :]
        response = response.strip().rstrip(".")
        self.debug("[generate_number]", response)
        try:
            return float(response)
        except ValueError:
            if iterations > 3:
                raise ValueError("Failed to generate a valid number")
            # Retry with higher temperature for more variety
            return self.generate_number(temperature=self.temperature * 1.3, iterations=iterations+1)

    def generate_boolean(self) -> bool:
        """
        Generate a boolean value by comparing logits for 'true' and 'false' tokens.
        
        Returns:
            bool: The generated boolean value
        """
        prompt = self.get_prompt()
        self.debug("[generate_boolean]", prompt, is_prompt=True)

        input_tensor = self.tokenizer.encode(prompt, return_tensors="pt")
        output = self.model.forward(input_tensor.to(self.model.device))
        logits = output.logits[0, -1]

        # Compare logits for true/false tokens to make decision
        # TODO: Handle cases where true/false are multi-token
        true_token_id = self.tokenizer.convert_tokens_to_ids("true")
        false_token_id = self.tokenizer.convert_tokens_to_ids("false")

        result = logits[true_token_id] > logits[false_token_id]

        self.debug("[generate_boolean]", result)

        return result.item()

    def generate_string(self) -> str:
        """
        Generate a string value using the language model.
        
        Uses StringStoppingCriteria to ensure the string is properly terminated
        with a quote character.
        
        Returns:
            str: The generated string value
        """
        # Start with opening quote
        prompt = self.get_prompt() + '"'
        self.debug("[generate_string]", prompt, is_prompt=True)
        input_tokens = self.tokenizer.encode(prompt, return_tensors="pt").to(
            self.model.device
        )

        response = self.model.generate(
            input_tokens,
            max_new_tokens=self.max_string_token_length,
            num_return_sequences=1,
            temperature=self.temperature,
            stopping_criteria=[
                StringStoppingCriteria(self.tokenizer, len(input_tokens[0]))
            ],
            pad_token_id=self.tokenizer.eos_token_id,
        )

        # Handle models that may include the prompt in their output
        if (
            len(response[0]) >= len(input_tokens[0])
            and (response[0][: len(input_tokens[0])] == input_tokens).all()
        ):
            response = response[0][len(input_tokens[0]) :]
        if response.shape[0] == 1:
            response = response[0]

        response = self.tokenizer.decode(response, skip_special_tokens=True)

        self.debug("[generate_string]", "|" + response + "|")

        # Extract string content between quotes if present
        if response.count('"') < 1:
            return response

        return response.split('"')[0].strip()

    def generate_object(
        self, properties: Dict[str, Any], obj: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Generate a JSON object by recursively generating values for each property.
        
        Args:
            properties: Schema properties defining the object structure
            obj: The object being populated with generated values
            
        Returns:
            Dict[str, Any]: The generated object with all properties filled
        """
        for key, schema in properties.items():
            self.debug("[generate_object] generating value for", key)
            obj[key] = self.generate_value(schema, obj, key)
        return obj

    def generate_value(
        self,
        schema: Dict[str, Any],
        obj: Union[Dict[str, Any], List[Any]],
        key: Union[str, None] = None,
    ) -> Any:
        """
        Generate a value based on the schema type.
        
        This is the core dispatch method that routes to the appropriate generator
        based on the schema type (number, boolean, string, array, object).
        
        Args:
            schema: The schema defining the value type and constraints
            obj: The parent object/array being populated
            key: The key in the parent object (if applicable)
            
        Returns:
            Any: The generated value matching the schema type
            
        Raises:
            ValueError: If the schema type is not supported
        """
        schema_type = schema["type"]
        if schema_type == "number":
            if key:
                obj[key] = self.generation_marker
            else:
                obj.append(self.generation_marker)
            return self.generate_number()
        elif schema_type == "boolean":
            if key:
                obj[key] = self.generation_marker
            else:
                obj.append(self.generation_marker)
            return self.generate_boolean()
        elif schema_type == "string":
            if key:
                obj[key] = self.generation_marker
            else:
                obj.append(self.generation_marker)
            return self.generate_string()
        elif schema_type == "array":
            new_array = []
            obj[key] = new_array
            return self.generate_array(schema["items"], new_array)
        elif schema_type == "object":
            new_obj = {}
            if key:
                obj[key] = new_obj
            else:
                obj.append(new_obj)
            return self.generate_object(schema["properties"], new_obj)
        else:
            raise ValueError(f"Unsupported schema type: {schema_type}")

    def generate_array(self, item_schema: Dict[str, Any], obj: Dict[str, Any]) -> list:
        """
        Generate an array of values conforming to the item schema.
        
        Uses the language model to determine when to stop generating array elements
        by looking at the likelihood of comma vs closing bracket tokens.
        
        Args:
            item_schema: Schema defining the type of items in the array
            obj: The array being populated
            
        Returns:
            list: The generated array with all elements
        """
        for _ in range(self.max_array_length):
            # Generate at least one element
            element = self.generate_value(item_schema, obj)
            obj[-1] = element

            # Check if we should continue generating elements
            obj.append(self.generation_marker)
            input_prompt = self.get_prompt()
            obj.pop()
            input_tensor = self.tokenizer.encode(input_prompt, return_tensors="pt")
            output = self.model.forward(input_tensor.to(self.model.device))
            logits = output.logits[0, -1]

            # Look at top token probabilities to decide whether to continue
            top_indices = logits.topk(30).indices
            sorted_token_ids = top_indices[logits[top_indices].argsort(descending=True)]

            found_comma = False
            found_close_bracket = False

            for token_id in sorted_token_ids:
                decoded_token = self.tokenizer.decode(token_id)
                if ',' in decoded_token:
                    found_comma = True
                    break
                if ']' in decoded_token:
                    found_close_bracket = True
                    break

            # Stop if we're more likely to close the array than continue
            if found_close_bracket or not found_comma:
                break

        return obj

    def get_prompt(self):
        """
        Build the generation prompt by combining the user prompt, schema, and current progress.
        
        Returns:
            str: The formatted prompt for the next generation step
        
        Raises:
            ValueError: If the generation marker is not found in the current state
        """
        template = """{prompt}\nOutput result in the following JSON schema format:\n{schema}\nResult: {progress}"""
        progress = json.dumps(self.value)
        gen_marker_index = progress.find(f'"{self.generation_marker}"')
        if gen_marker_index != -1:
            progress = progress[:gen_marker_index]
        else:
            raise ValueError("Failed to find generation marker")

        prompt = template.format(
            prompt=self.prompt,
            schema=json.dumps(self.json_schema),
            progress=progress,
        )

        return prompt

    def __call__(self) -> Dict[str, Any]:
        """
        Generate a complete JSON object conforming to the schema.
        
        This is the main entry point for using the Jsonformer.
        
        Returns:
            Dict[str, Any]: The complete generated JSON object
        """
        self.value = {}
        generated_data = self.generate_object(
            self.json_schema["properties"], self.value
        )
        return generated_data
