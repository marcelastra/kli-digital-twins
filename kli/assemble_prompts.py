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
* the model, temperature, chunking, and (by default) every word of the prompt.

-------------------------------------------------------------------------------
CONTROL-ARM WORDING: A CHOICE WORTH UNDERSTANDING
-------------------------------------------------------------------------------

``control_wording`` selects between two defensible designs. The default,
``"upstream"``, reproduces what Peng et al. actually did.

**What they did.** Verified across all 76 of their released
``empty_persona`` configs: the *only* substantive difference from the
corresponding persona config is ``empty_persona_prompt: false -> true``. Every
prompt string is left in place. Their control arm therefore ran with this
system instruction, verbatim::

    You are an AI assistant. Your task is to answer the 'New Survey Question'
    as if you are the person described in the 'Persona Profile' (which consists
    of their past survey responses).
    Remain consistent with the persona's previous answers and stated traits. ...

and this separator::

    ## New Survey Question & Instructions (Please respond as the persona
    described above):

with ``persona_prompt_header`` ("## Persona Profile (This individual's past
survey responses):") still supplied. In other words, their control was told to
imitate a person described in a profile that was not there.

(One detail is not recoverable from the public release: whether that header line
was actually *printed* when the persona body was empty. The config supplies the
string, but the code that decided -- ``create_text_simulation_input.py`` -- is
the unpublished file. Worth asking the authors.)

**Why that is the stricter design.** Exactly one thing changes between arms.
Any difference in outcome is attributable to the persona and nothing else, and
the numbers are directly comparable to the published ones.

**Why it is also arguably a confound.** A model instructed to imitate someone
it was never shown is being handicapped, not merely deprived of information. If
that depresses control performance, the twin-minus-control gap is inflated.

**The interesting consequence.** Peng et al. report that twins are *"only
modestly more accurate than those of a homogeneous base LLM."* If their control
was in fact handicapped by incoherent instructions, then their headline is
*conservative* -- the true twin advantage over a fairly-prompted baseline would
be smaller still. Their negative finding survives the criticism, and is
strengthened by it.

**So the difference between the two wordings is itself measurable**, and worth
measuring: running both tells you how much of the "base LLM" benchmark is
prompt artifact rather than model capability. Nobody appears to have reported
this. It is a cheap, self-contained result -- one extra arm on data you already
have.

Settings:

``control_wording: upstream`` (default)
    Keep every persona-arm string in the control. Matches Peng et al.; results
    comparable to the paper. The run prints a one-line note recording the
    choice.

``control_wording: neutral``
    Replace the persona header, the separator and the opening sentence with
    wording that does not presuppose a persona. Cleaner baseline, but two
    things differ between arms rather than one.

Either way the exact strings used are written into ``run_manifest.json``, so
whichever you choose is on the record and reportable.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

__all__ = [
    "UPSTREAM_WORDING_NOTE",
    "resolve_arm_wording",
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


#: Note printed once per control-arm run under ``control_wording: upstream``.
#: Not a warning: this is the default and it is deliberate. It exists so that
#: nobody discovers the design choice six months later while writing it up.
UPSTREAM_WORDING_NOTE = (
    "Control arm is using UPSTREAM wording (control_wording: upstream).\n"
    "  The prompt still refers to a 'Persona Profile' that this arm does not\n"
    "  contain. That is deliberate and matches Peng et al. exactly -- verified\n"
    "  across all 76 of their released empty_persona configs, where the only\n"
    "  change from the persona arm is the empty_persona_prompt flag.\n"
    "  Upside: exactly one variable differs, so results are comparable to the\n"
    "  published numbers. Caveat: a model told to imitate someone it was never\n"
    "  shown may be handicapped rather than merely uninformed, which would\n"
    "  inflate the twin-minus-control gap.\n"
    "  Set control_wording: neutral for a baseline whose prompt does not\n"
    "  presuppose a persona. Running BOTH measures how much of the 'base LLM'\n"
    "  benchmark is prompt artifact -- a result nobody appears to have\n"
    "  reported. See kli/assemble_prompts.py for the full argument.\n"
    "  Exact strings used are recorded in run_manifest.json."
)


def resolve_arm_wording(
    *,
    empty_persona: bool,
    control_wording: str = "upstream",
    persona_header: str | None = None,
    separator: str | None = None,
    system_instruction: str | None = None,
) -> dict[str, Any]:
    """Resolve the prompt strings for one arm.

    An explicit value from the config always wins. Otherwise the defaults are
    chosen by arm and by ``control_wording`` -- see the module docstring for why
    the default reproduces Peng et al. rather than the cleaner baseline.

    Args:
        empty_persona: Whether this is the no-persona control arm.
        control_wording: ``"upstream"`` (default) or ``"neutral"``.
        persona_header: Explicit override, or None for the default.
        separator: Explicit override, or None for the default.
        system_instruction: Explicit override, or None for the default.

    Returns:
        ``{"persona_header", "separator", "system_instruction",
        "persona_wording", "control_wording", "note"}``. ``persona_wording``
        is passed to :func:`kli.render_questions.build_question_prompt` and
        controls the opening sentence only. ``note`` is a one-line string to
        print, or None.

    Raises:
        ValueError: If ``control_wording`` is not one of the two known values.
    """
    control_wording = (control_wording or "upstream").strip().lower()
    if control_wording not in {"upstream", "neutral"}:
        raise ValueError(
            f"control_wording must be 'upstream' or 'neutral', got "
            f"{control_wording!r}. See kli/assemble_prompts.py."
        )

    use_neutral = empty_persona and control_wording == "neutral"

    resolved = {
        "persona_header": persona_header or DEFAULT_PERSONA_HEADER,
        "separator": separator
        or (DEFAULT_SEPARATOR_NO_PERSONA if use_neutral else DEFAULT_SEPARATOR),
        "system_instruction": system_instruction
        or (
            DEFAULT_SYSTEM_INSTRUCTION_NO_PERSONA
            if use_neutral
            else DEFAULT_SYSTEM_INSTRUCTION
        ),
        "persona_wording": not use_neutral,
        "control_wording": control_wording if empty_persona else None,
        "note": None,
    }

    if empty_persona and control_wording == "upstream":
        resolved["note"] = UPSTREAM_WORDING_NOTE

    return resolved


def check_arm_consistency(
    *,
    empty_persona: bool,
    system_instruction: str,
    separator: str = "",
    control_wording: str = "upstream",
) -> list[str]:
    """Flag wording that presupposes a persona the control arm lacks.

    Under ``control_wording: upstream`` this returns nothing: the mismatch is
    the intended design, and :data:`UPSTREAM_WORDING_NOTE` covers it instead.
    Under ``control_wording: neutral`` it fires, because a persona-referencing
    string there means an explicit config override has silently defeated the
    neutral wording the researcher asked for.

    Args:
        empty_persona: Whether this is the control arm.
        system_instruction: The system message for this arm.
        separator: The text between persona and questions, also checked.
        control_wording: ``"upstream"`` or ``"neutral"``.

    Returns:
        Human-readable warnings; empty if nothing looks wrong.
    """
    if not empty_persona or (control_wording or "upstream").lower() != "neutral":
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
                f"control_wording is 'neutral', but {field_name} is explicitly "
                f"set in the config and refers to "
                f"{', '.join(repr(h) for h in hits)} -- a persona this arm does "
                "not contain. The explicit value wins, so this arm is running "
                "upstream-style wording under a 'neutral' label. Remove that "
                "key from the config, or set control_wording: upstream so the "
                "label matches what is actually happening."
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
