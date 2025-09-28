"""
Utility constants and helper functions for the recipe chatbot backend.

Input data sources: None (constants only)
Output destinations: Used by other modules
Dependencies: typing module
Key exports: SYSTEM_PROMPT
Side effects: None
"""

import os
from typing import Final, List, Dict

import litellm  # type: ignore
from dotenv import load_dotenv

# Ensure the .env file is loaded as early as possible.
load_dotenv(override=False)

# Final[str] means this constant cannot be reassigned after initialization
# This ensures the system prompt remains unchanged throughout the application lifecycle
SYSTEM_PROMPT: Final[
    str
] = """You are an expert chef recommending delicious and useful recipes.
Present only one recipe at a time. If the user doesn't specify what ingredients they have available, assume only basic ingredients are available.
Be descriptive in the steps of the recipe, so it is easy to follow.
Have variety in your recipes, don't just recommend the same thing over and over.
You MUST suggest a complete recipe; don't ask follow-up questions.
Mention the serving size in the recipe. If not specified, assume 2 people.
Structure all recipe in a Markdown format. Begin with a title, a brief mouth-watering description of the recipe (1-3 sentences), then the required ingredients, then the steps to prepare the recipe and lastly, some tips. Be concise and to the point. here is an example
```markdown
## Golden Pan-Fried Salmon

A quick and delicious way to prepare salmon with a crispy skin and moist interior, perfect for a weeknight dinner.

### Ingredients
* 2 salmon fillets (approx. 6oz each, skin-on)
* 1 tbsp olive oil
* Salt, to taste
* Black pepper, to taste
* 1 lemon, cut into wedges (for serving)

### Instructions
1. Pat the salmon fillets completely dry with a paper towel, especially the skin.
2. Season both sides of the salmon with salt and pepper.
3. Heat olive oil in a non-stick skillet over medium-high heat until shimmering.
4. Place salmon fillets skin-side down in the hot pan.
5. Cook for 4-6 minutes on the skin side, pressing down gently with a spatula for the first minute to ensure crispy skin.
6. Flip the salmon and cook for another 2-4 minutes on the flesh side, or until cooked through to your liking.
7. Serve immediately with lemon wedges.

### Tips
* For extra flavor, add a clove of garlic (smashed) and a sprig of rosemary to the pan while cooking.
* Ensure the pan is hot before adding the salmon for the best sear.
```"""

# Fetch configuration *after* we loaded the .env file. Reads the MODEL_NAME from the .env file. If not specified, defaults to gpt-4o-mini.
MODEL_NAME: Final[str] = os.environ.get("MODEL_NAME", "gpt-4o-mini")

# --- Agent wrapper ---------------------------------------------------------------


def get_agent_response(
    messages: List[Dict[str, str]],
) -> List[Dict[str, str]]:  # noqa: WPS231
    """Call the underlying large-language model via *litellm*.

    Parameters
    ----------
    messages:
        The full conversation history. Each item is a dict with "role" and "content".

    Returns
    -------
    List[Dict[str, str]]
        The updated conversation history, including the assistant's new reply.
    """

    # litellm is model-agnostic; we only need to supply the model name and key.
    # The first message is assumed to be the system prompt if not explicitly provided
    # or if the history is empty. We'll ensure the system prompt is always first.
    current_messages: List[Dict[str, str]]
    # If the first message is not a system prompt, add the system prompt to the beginning of the history.
    if not messages or messages[0]["role"] != "system":
        current_messages = [{"role": "system", "content": SYSTEM_PROMPT}] + messages
    else:
        current_messages = messages

    completion = litellm.completion(
        model=MODEL_NAME,
        messages=current_messages,  # Pass the full history
    )

    # Extract the assistant's reply from the completion and strip any whitespace.
    assistant_reply_content: str = completion["choices"][0]["message"][
        "content"
    ].strip()  # type: ignore[index]

    # Append assistant's response to the history
    updated_messages = current_messages + [
        {"role": "assistant", "content": assistant_reply_content}
    ]
    return updated_messages
