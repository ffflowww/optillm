from typing import List, Tuple
from optillm.cot_decoding import aggregate_paths_based_on_scores
import numpy as np
from transformers import AutoTokenizer
import logging

logger = logging.getLogger(__name__)


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
            logger.warning("Provided answer index is out of range!")
            break

        probs = np.exp(logprobs[ti])  # Convert log-probabilities to probabilities
        if len(probs) > 1:
            confidence_sum += np.abs(probs[0] - probs[1])
        else:
            confidence_sum += 1.0  # Max confidence if there's only one token
        valid_tokens += 1

    return confidence_sum / valid_tokens if valid_tokens > 0 else 0.0


def cot_decode_vllm(
        system_prompt: str,
        user_prompt: str,
        client,
        model: str,
        n_paths: int = 10,
        max_new_tokens: int = 512,
        temperature: float = 1.0,
        top_p: float = 1.0,
        answer_keywords: str = "",
        tokenizer_name: str = "",
        aggregate_paths: bool = False,
        return_all: bool = False,
        debug_print: bool = False,
        return_completion_tokens: bool = False,
) -> Tuple[str, float] | List[Tuple[str, float]]:
    """
    Implement CoT-decoding for a given chat input.
    
    Args:
        system_prompt: string.
        user_prompt: string.
        client: OpenAI client object
        model: model name
        n_paths: The number of alternative tokens to consider at the first step.
        max_new_tokens: Maximum number of new tokens to generate.
        temperature: Sampling temperature.
        top_p: Nucleus sampling probability.
        answer_keywords: Everything behind that substring is treated as answer to the question
        tokenizer_name: specific tokenizer name (if different from the model name)
        aggregate_paths: Whether to aggregate multiple paths.
        return_all: Returns all generated paths and its confidence scores
        debug_print: print extra info
        return_completion_tokens: if it's used internally (returns with completion tokens)

    Returns:
        A tuple containing the best path (or aggregated result) and its confidence score.
    """
    cot_completion_tokens = 0

    # Building base prompt with model's tokenizer
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name) if tokenizer_name else AutoTokenizer.from_pretrained(model)
    base_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    # Get the top-k tokens for the first decoding step (API call)
    completion_init = client.completions.create(
        model=model,
        prompt=base_prompt,
        temperature=temperature,
        top_p=top_p,
        max_tokens=1,
        logprobs=n_paths,
    )
    cot_completion_tokens += completion_init.usage.completion_tokens
    top_tokens_full = completion_init.choices[0].logprobs.top_logprobs[0]
    # it's needed because vllm sometimes is bugged and returns more tokens than needed
    top_tokens = sorted(top_tokens_full, key=top_tokens_full.get, reverse=True)[:n_paths]
    if len(top_tokens) != n_paths:
        logger.warning(f"The number of paths explored is less than requested! {len(top_tokens)} vs {n_paths}")
        if debug_print:
            print(f"The number of paths explored is less than requested! {len(top_tokens)} vs {n_paths}")

    # Generating all paths
    paths = []
    prompts = [base_prompt + token for token in top_tokens]
    completions = client.completions.create(
        model=model,
        prompt=prompts,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_new_tokens - 1,
        logprobs=2,
    )
    cot_completion_tokens += completions.usage.completion_tokens

    for i, completion in enumerate(completions.choices):
        answer_text = top_tokens[i] + completion.text
        tokens_with_log_probs = completion.logprobs.top_logprobs
        probs = [[*item.values()] for item in tokens_with_log_probs]

        # Calculate confidence scores
        answer_indexes = [i for i in range(len(probs))]  # calculating certainty based on all tokens
        res_text = "||all_text||"  # just a placeholder for pretty print in logger
        if answer_keywords:
            ida = answer_text.rfind(answer_keywords)
            if ida == -1:
                logger.warning("Provided answer keywords are not found in the answer! "
                               "Falling back to calculating confidence for the whole answer.")
                if debug_print:
                    print("Provided answer keywords are not found in the answer! "
                          "Falling back to calculating confidence for the whole answer.")
            else:
                res_text = answer_text[ida + len(answer_keywords):]  # real answer text here
                n_last_tokens = len(tokenizer.encode(res_text))  # count how many tokens there
                answer_indexes = answer_indexes[-n_last_tokens:]  # using only those last tokens for confidence
        confidence = calculate_confidence_logprobs(probs, answer_indexes)
        paths.append((answer_text, confidence))
        logger.info(f"[CoT decode {i+1}/{n_paths}]\n[Confidence:] {confidence}\n[Real answer for confidence scoring:] {res_text}\n[Answer:] {answer_text}")
        if debug_print:
            print(f"[CoT decode {i+1}/{n_paths}]\n[Confidence:] {confidence}\n[Real answer for confidence scoring:] {res_text}\n[Answer:] {answer_text}")

    if return_completion_tokens:
        return max(paths, key=lambda x: x[1])[0], cot_completion_tokens
    elif return_all:
        return paths
    elif aggregate_paths:
        return aggregate_paths_based_on_scores(paths)
    else:
        return max(paths, key=lambda x: x[1])
