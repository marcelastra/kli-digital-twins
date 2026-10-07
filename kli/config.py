"""Read the nested study-config schema, including this project's additions.

ORIGINAL WORK (Kravis Leadership Institute). See ATTRIBUTION.md.

The key layout follows the Twin-2K-500 Mega-Study's published configs (so their
479 released config files remain readable here), with four additions marked
NEW below.

    text_simulation:
      questions_to_texts:
        input_path:  data/<study>/response_json
        output_dir:  text_simulation/<study>/text_questions
        include_reasoning: false

      create_text_simulation_input:
        persona_text_dir: data/full_persona_text
        question_prompts_dir: ...
        output_combined_prompts_dir: ...
        persona_prompt_header: |
          ## Persona Profile ...
        persona_question_prompt_separator: |
          ## New Survey Question ...
        empty_persona_prompt: false        # the no-persona control arm

      LLM_simulation:
        input_dir: ...
        output_dir: ...
        question_json_base_dir: data/<study>/response_json
        output_updated_questions_dir: .../response_json_llm_imputed
        system_instruction: |
          You are ...
        LLM_config:
          provider: local                  # NEW: "local" uses kli.providers.local
          model_name: mistral
          base_url: http://localhost:11434 # NEW: local server endpoint
          temperature: 0.7
          max_tokens: 16384
          max_retries: 5
        num_workers: 4
        questions_per_call: 10             # NEW: chunk size, null = whole survey
        allow_remote_endpoint: false       # NEW: privacy guard override
        force_regenerate: false
        max_personas: -1

      convert_to_csv:
        input_json_dir: .../response_json_llm_imputed
        reference_csv_path: data/<study>/raw_data/response.csv
        output_dir: .../csv_output

Note that the vendored ``text_simulation/*.py`` scripts do NOT read this
schema -- they take individual CLI flags and flat keys. The ``kli`` pipeline
reads it properly. That difference is why the ``kli`` stages exist.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

__all__ = ["load_config", "section", "get", "ConfigError"]

#: Sentinel distinguishing "no default supplied" from "default is None".
_MISSING = object()


class ConfigError(ValueError):
    """Raised when a required config key is missing or malformed."""


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a study YAML config.

    Args:
        path: Path to the config file.

    Returns:
        The parsed config.

    Raises:
        ConfigError: If the file is missing, unparseable, or not a mapping.
    """
    path = Path(path)
    if not path.is_file():
        raise ConfigError(f"Config file not found: {path}")
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ConfigError(f"Could not parse {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ConfigError(f"{path} does not contain a YAML mapping.")
    return data


def get(config: dict[str, Any], dotted: str, default: Any = _MISSING) -> Any:
    """Fetch a nested key by dotted path.

    Args:
        config: The parsed config.
        dotted: e.g. ``"text_simulation.LLM_simulation.LLM_config.model_name"``.
        default: Returned when the key is absent. If omitted, a missing key is
            an error -- which is deliberate: a silently-defaulted model name or
            output directory is how an experiment ends up mislabelled.

    Returns:
        The value at that path.

    Raises:
        ConfigError: If the path is absent and no default was given.
    """
    node: Any = config
    for part in dotted.split("."):
        if not isinstance(node, dict) or part not in node:
            if default is _MISSING:
                raise ConfigError(
                    f"Required config key '{dotted}' is missing. "
                    "See kli/config.py for the expected layout."
                )
            return default
        node = node[part]
    return node


def section(config: dict[str, Any], dotted: str) -> dict[str, Any]:
    """Fetch a nested mapping by dotted path, defaulting to empty.

    Args:
        config: The parsed config.
        dotted: Dotted path to a mapping.

    Returns:
        The mapping, or ``{}`` if absent or null.
    """
    value = get(config, dotted, None)
    return value if isinstance(value, dict) else {}
