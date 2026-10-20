"""Original pipeline components developed at the Kravis Leadership Institute.

ORIGINAL WORK (Kravis Leadership Institute).

Everything in this package is original work. Nothing here is copied or derived
from either upstream repository -- see ATTRIBUTION.md for the full provenance
map and the README for why each piece was needed.

The four components, and the gap each one closes:

survey_tree
    The schema adapter, and the fix for upstream's silent answer-misassignment
    hazard. One canonical traversal of the nested per-participant response
    document, used by both prompt construction and response parsing, with a
    QuestionID round-trip check so a mismatch fails loudly instead of shifting
    every subsequent answer.

render_questions
    Renders questions into an answerable prompt with an explicit JSON skeleton.
    Covers constant-sum questions, which the upstream renderer cannot handle at
    all.

assemble_prompts
    Persona and question assembly, including the no-persona control arm that
    upstream specifies in every config but never implements.

providers.local
    A local model provider (Mistral via Ollama) with a hard no-egress guard, so
    participant data stays on the machine.

parse_responses
    Model output back into the response-JSON schema, with the round-trip check
    and explicit per-question imputation flags -- so a question the model failed
    can never be mistaken for one it answered.

responses_to_csv
    The analysis CSV. This pipeline stage exists in neither upstream
    repository.

correspondence
    Human-twin correspondence metrics following Stojanov (2026): group-level
    mean and dispersion agreement, individual-level profile similarity, and a
    decomposition of error into systematic and random components. Includes the
    published benchmark values and the paired arm comparison.

reconcile
    Post-ingestion validation. Upstream has no test coverage for Qualtrics
    parsing; this reports what was dropped and why.
"""

__version__ = "0.1.0"
