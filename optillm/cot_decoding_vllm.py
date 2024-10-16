from typing import List, Tuple
from optillm.cot_decoding import aggregate_paths_based_on_scores
import numpy as np
import warnings


def calculate_confidence_logprobs(logprobs: List[List[float]], answer_indexes: List[int]) -> float:
    """
    Calculate the confidence score (Δ) as specified in the paper.

    Args:
        logprobs: List of Lists containing 2 most-likely log-probabilities for each decoding step
        answer_indexes: List of valid token's indexes considered for confidence calculation

    Returns:
        Confidence score (Δ)
    """
    confidence_sum = 0.0
    valid_tokens = 0

    lp_len = len(logprobs)
    for ti in answer_indexes:
        if ti >= lp_len:
            warnings.warn("Provided answer index is out of range!", UserWarning)
            break

        probs = np.exp(logprobs[ti])  # Convert log-probabilities to probabilities
        if len(probs) > 1:
            confidence_sum += np.abs(probs[0] - probs[1])
        else:
            confidence_sum += 1.0  # Max confidence if there's only one token
        valid_tokens += 1

    return confidence_sum / valid_tokens if valid_tokens > 0 else 0.0


def build_qwen_chat_prompt(system_message, user_message):
    # Start with the system message
    prompt = "<|im_start|>system\n"
    prompt += f"{system_message}"
    prompt += "<|im_end|>\n"

    prompt += "<|im_start|>user\n"
    prompt += f"{user_message}"
    prompt += "<|im_end|>\n"

    prompt += "<|im_start|>assistant\n"

    return prompt


def cot_decode_vllm(
    base_prompt: str,
    client,
    model: str,
    k: int = 10,
    max_new_tokens: int = 512,
    temperature: float = 1.0,
    top_p: float = 1.0,
    aggregate_paths: bool = False,
    is_return_all: bool = False,
) -> Tuple[str, float] | List[Tuple[str, float]]:
    """
    Implement CoT-decoding for a given chat input.
    
    Args:
        base_prompt: Full prompt with special symbols and formatting (model specific(!))
        client: OpenAI client object
        model: model name
        k: The number of alternative tokens to consider at the first step.
        max_new_tokens: Maximum number of new tokens to generate.
        temperature: Sampling temperature.
        top_p: Nucleus sampling probability.
        aggregate_paths: Whether to aggregate multiple paths.
        is_return_all: Returns all generated paths and its confidence scores

    Returns:
        A tuple containing the best path (or aggregated result) and its confidence score.
    """

    # Get the top-k tokens for the first decoding step (API call)
    completion = client.completions.create(
        model=model,
        prompt=base_prompt,
        temperature=temperature,
        top_p=top_p,
        max_tokens=1,
        logprobs=k,
    )
    top_tokens_full = completion.choices[0].logprobs.top_logprobs[0]
    top_tokens = [key for key in top_tokens_full]

    # Generating all paths
    paths = []
    for token in top_tokens:
        prompt = base_prompt + token
        completion = client.completions.create(
            model=model,
            prompt=prompt,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_new_tokens - 1,
            logprobs=2,
        )

        answer_text = token + completion.choices[0].text
        tokens_with_log_probs = completion.choices[0].logprobs.top_logprobs
        probs = [[*item.values()] for item in tokens_with_log_probs]

        # Calculate confidence score (Δ)
        answer_indxs = [i for i in range(len(probs))]  # calculating certainty based on all tokens
        # actually we need to calculate confidence only for "Result" tokens, not all tokens
        confidence = calculate_confidence_logprobs(probs, answer_indxs)
        paths.append((answer_text, confidence))

    if is_return_all:
        return paths
    elif aggregate_paths:
        return aggregate_paths_based_on_scores(paths)
    else:
        return max(paths, key=lambda x: x[1])
