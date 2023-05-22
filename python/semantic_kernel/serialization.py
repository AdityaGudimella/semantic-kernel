"""API to serialize SemanticKernel `Kernel`s."""
import pickle
from semantic_kernel.kernel import Kernel


def from_json(json: str) -> Kernel:
    """Deserialize a `Kernel` from JSON."""
    # TODO: Implement this.
    asdict = {}
    if text_completion_client in asdict:
        if asdict[text_completion_client] == "openai":
            kernel.add_text_completion_service(
                sk.OpenAITextCompletionService(api_key, org_id)
            )
    return pickle.loads(json)


def to_json(kernel: Kernel) -> str:
    """Serialize a `Kernel` to JSON."""
    # TODO: Implement this.
    return pickle.dumps(kernel)
