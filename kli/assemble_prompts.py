"""Assemble persona + question into a final prompt, including the no-persona control.

ORIGINAL WORK (Kravis Leadership Institute). See ATTRIBUTION.md.

-------------------------------------------------------------------------------
THE NO-PERSONA CONTROL ARM
-------------------------------------------------------------------------------

This is the reason the module exists. Setting ``empty_persona_prompt: true``
produces prompts with the persona removed, everything else held fixed.

Why it matters: the Mega-Study's central negative finding is that twins are
*"only modestly more accurate than those of a homogeneous base LLM."* You
cannot make -- or refute -- that comparison without running the identical
questions with the persona withheld. A twin accuracy figure on its own does not
distinguish "the model captured this individual" from "the model captured the
population average," and the published evidence says the gap is small.

The vendored upstream code cannot do this at all. The key appears in every
released config, but no implementation reads it:
``text_simulation/create_text_simulation_input.py:57`` requires a *matching
persona file per participant* (``set(persona_files) & set(question_files)``), so
a persona-free prompt is unreachable by construction.

What this implementation holds constant between the two arms:

* the participant set -- the control iterates the same participants, so N and
  the per-participant question set are identical;
* the questions, including each participant's realized randomization and
  stimuli, because those live in the per-participant response JSON rather than
  in the persona;
* the system instruction, model, temperature, and chunking.

Only the persona text is withheld. The persona header is withheld with it,
since a header introducing a profile that is not there would itself be a cue.
The system instruction is left untouched on purpose -- it is part of the
treatment being held fixed -- so if yours names a "Persona Profile" that the
control prompt lacks, override ``system_instruction`` for that arm and say so
in your appendix. :func:`check_arm_consistency` warns about exactly this.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

__all__ = [
    "DEFAULT_PERSONA_HEADER",
    "DEFAULT_SEPARATOR",
    "DEFAULT_SEPARATOR_NO_PERSONA",
    "DEFAULT_SYSTEM_INSTRUCTION",
    "DEFAULT_SYSTEM_INSTRUCTION_NO_PERSONA",
    "find_persona_files",
    "load_persona",
    "assemble_prompt",
    "check_arm_consistency",
    "MissingPersonaError",
]

DEFAULT_PERSONA_HEADER = (
    "## Persona Profile (this individual's past survey responses):"
)

DEFAULT_SEPARATOR = (
    "## New Survey Question & Instructions "
    "(respond as the person described above):"
)

#: Control-arm separator. See :data:`kli.render_questions._OPENING_NO_PERSONA`
#: for why the control does not reference a persona it was never given.
DEFAULT_SEPARATOR_NO_PERSONA = "## Survey Questions & Instructions:"

DEFAULT_SYSTEM_INSTRUCTION = (
    "You are simulating a specific survey respondent. Answer the new survey "
    "questions as that person would answer them, staying consistent with their "
    "earlier answers and stated characteristics. Real people are inconsistent, "
    "uncertain, and not always rational -- answer as the person would, not as an "
    "ideal reasoner would. Follow the response format exactly."
)

#: Control-arm system instruction. Matched in structure and in its
#: "answer like a person, not an ideal reasoner" framing, with every reference
#: to a specific respondent removed -- so the comparison isolates the persona
#: rather than the instruction to imitate someone.
DEFAULT_SYSTEM_INSTRUCTION_NO_PERSONA = (
    "You are answering a survey as a member of the general public. Real people "
    "are inconsistent, uncertain, and not always rational -- answer as a person "
    "would, not as an ideal reasoner would. Follow the response format exactly."
)

#: Matches ``pid_<token>`` at the START of a filename, so non-numeric
#: participant IDs work. Upstream hardcodes ``pid_\\d+`` in three places, which
#: silently excludes any non-numeric ID.
#:
#: Anchored deliberately: an unanchored search matches the ``pid_here`` inside
#: ``no_pid_here.json`` and would invent a participant called "here".
_PID_RE = re.compile(r"^pid[_-]([A-Za-z0-9][A-Za-z0-9.\-]*?)(?:_|\.|$)")


class MissingPersonaError(FileNotFoundError):
    """Raised when a participant has no persona file in a persona-using arm."""


def extract_pid(path: str | Path) -> str | None:
    """Pull the participant ID out of a filename.

    Args:
        path: A file path such as ``pid_1030_response.json``.

    Returns:
        The ID (``"1030"``), or None if the filename has no ``pid_`` segment.
    """
    match = _PID_RE.search(Path(path).name)
    return match.group(1) if match else None


def find_persona_files(persona_dir: str | Path) -> dict[str, Path]:
    """Index persona text files by participant ID.

    Args:
        persona_dir: Directory of ``.txt`` persona files.

    Returns:
        Mapping of participant ID to file path. Empty if the directory does not
        exist, which is a valid state for the no-persona arm.
    """
    persona_dir = Path(persona_dir)
    if not persona_dir.is_dir():
        return {}
    found: dict[str, Path] = {}
    for path in sorted(persona_dir.glob("*.txt")):
        pid = extract_pid(path)
        if pid is not None:
            found[pid] = path
    return found


def load_persona(path: str | Path) -> str:
    """Read a persona text file.

    Args:
        path: Path to the persona ``.txt``.

    Returns:
        The persona text, stripped of surrounding whitespace.
    """
    return Path(path).read_text(encoding="utf-8").strip()


def assemble_prompt(
    *,
    question_prompt: str,
    persona_text: str | None,
    empty_persona: bool = False,
    persona_header: str = DEFAULT_PERSONA_HEADER,
    separator: str = DEFAULT_SEPARATOR,
    prompt_end: str | None = None,
) -> str:
    """Build the final user prompt for one participant and one question chunk.

    Args:
        question_prompt: Rendered questions plus response skeleton, from
            :func:`kli.render_questions.build_question_prompt`.
        persona_text: The participant's persona. Ignored when
            ``empty_persona`` is True.
        empty_persona: **The control arm.** When True, the persona and its
            header are omitted entirely; everything else is unchanged.
        persona_header: Text introducing the persona.
        separator: Text between persona and questions. Present in both arms, so
            the question framing is identical.
        prompt_end: Optional trailing text. (Upstream's ``combine_prompt_end``
            is inert -- it resolves to ``null`` in all 114 configs that set it,
            because the scaffolder never passes it to the template. Here it
            works.)

    Returns:
        The assembled prompt.

    Raises:
        ValueError: If a persona-using arm is given no persona text. Silently
            emitting a persona-free prompt in the treatment arm would quietly
            turn it into the control.
    """
    parts: list[str] = []

    if not empty_persona:
        if persona_text is None or not persona_text.strip():
            raise ValueError(
                "assemble_prompt() received no persona text but "
                "empty_persona=False. Refusing to build a persona-free prompt "
                "in a persona arm: that would silently convert this "
                "participant into a control observation. Either supply the "
                "persona or set empty_persona=True for the whole arm."
            )
        if persona_header.strip():
            parts.append(persona_header.strip())
        parts.append(persona_text.strip())

    if separator.strip():
        parts.append(separator.strip())

    parts.append(question_prompt.strip())

    if prompt_end and prompt_end.strip():
        parts.append(prompt_end.strip())

    return "\n\n".join(parts)


#: Phrases that presuppose a persona the control arm was never given. Kept
#: deliberately broad: a false warning costs one line of output, while a missed
#: one means the control arm was quietly asked to imitate a person it never saw,
#: which would undercut every comparison drawn from it.
_PERSONA_PHRASES = (
    "persona",
    "profile",
    "person described",
    "that person",
    "this person",
    "the individual",
    "this individual",
    "respondent",
    "as they would",
    "their past",
    "their previous",
    "their earlier",
    "stated characteristics",
    "consistent with their",
)


def check_arm_consistency(
    *,
    empty_persona: bool,
    system_instruction: str,
    separator: str = "",
) -> list[str]:
    """Warn about wording that would confound the no-persona comparison.

    Called by :mod:`kli.run_simulation` on every run. It fires when the control
    arm has been given text -- usually an explicit ``system_instruction`` in the
    config, which overrides the arm-aware default -- that refers to a person the
    prompt does not contain.

    Args:
        empty_persona: Whether this is the control arm.
        system_instruction: The system message for this arm.
        separator: The text between persona and questions, also checked.

    Returns:
        Human-readable warnings; empty if nothing looks wrong.
    """
    if not empty_persona:
        return []

    warnings: list[str] = []
    for field_name, text in (
        ("system_instruction", system_instruction),
        ("persona_question_prompt_separator", separator),
    ):
        if not text:
            continue
        lowered = text.lower()
        hits = sorted({p for p in _PERSONA_PHRASES if p in lowered})
        if hits:
            warnings.append(
                f"No-persona control arm: {field_name} refers to "
                f"{', '.join(repr(h) for h in hits)}, but no persona is "
                "supplied. The model is being told to imitate someone it was "
                "never shown, which is a confound rather than a clean baseline. "
                "Either remove that value from the config so the arm-aware "
                "default applies, or keep it deliberately and state the choice "
                "in your methods section. The exact strings used are recorded "
                "in run_manifest.json."
            )
    return warnings


def summarize_arm(*, empty_persona: bool, persona_dir: str | Path | None) -> str:
    """One-line description of the arm, for logs and run manifests.

    Args:
        empty_persona: Whether this is the control arm.
        persona_dir: Persona directory for persona arms.

    Returns:
        A short label.
    """
    if empty_persona:
        return "empty_persona (control: no persona supplied)"
    return f"persona from {persona_dir}"
