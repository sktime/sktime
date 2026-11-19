# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""OpenAI LLM interface."""

import os
from sktime.agent.llm.base import BaseLLM

class OpenAI(BaseLLM):
    """OpenAI LLM interface.

    Parameters
    ----------
    api_key : str, optional
        OpenAI API key. If not provided, will look for OPENAI_API_KEY env var.
    model : str, default="gpt-4"
        The model to use.
    """

    def __init__(self, api_key=None, model="gpt-4"):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model = model
        super().__init__()

    def generate_code(self, prompt: str) -> str:
        """Generate code using OpenAI API."""
        try:
            import openai
        except ImportError:
            raise ImportError(
                "openai package is required for OpenAI interface. "
                "Please install it with `pip install openai`."
            )

        client = openai.OpenAI(api_key=self.api_key)
        response = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a specialized sktime code generator."},
                {"role": "user", "content": prompt}
            ],
            temperature=0,
        )
        code = response.choices[0].message.content
        
        # Strip markdown code blocks if present
        if "```python" in code:
            code = code.split("```python")[1].split("```")[0]
        elif "```" in code:
            code = code.split("```")[1].split("```")[0]
            
        return code.strip()
