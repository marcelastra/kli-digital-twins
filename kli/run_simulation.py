"""Run the twin simulation: prompts -> local model -> imputed response JSON.

ORIGINAL WORK (Kravis Leadership Institute). See ATTRIBUTION.md.

Replaces the vendored ``text_simulation/run_LLM_simulations.py``, which reads
flat top-level config keys, prefixes every path with ``./text_simulation``,
hardcodes its output directory, defaults its provider to a broken Gemini
client, and cannot express the no-persona arm.

Usage::

    # check the local model is reachable before committing to a long run
    python -m kli.run_simulation --config configs/my_study.yaml --health-check

    # dry run: build and inspect prompts, call no model
    python -m kli.run_simulation --config configs/my_study.yaml --dry-run --limit 3

    # offline self-test: a stub model fills the skeleton, exercising the whole
    # parse/validate/write path with no model and no network
    python -m kli.run_simulation --config configs/my_study.yaml --stub-model

    # real run
    python -m kli.run_simulation --config configs/my_study.yaml

Every run writes ``run_manifest.json`` beside its outputs, recording the model,
temperature, arm, chunk size, whether the endpoint was loopback, and the
imputation rate. Cite it in your appendix rather than reconstructing settings
from memory.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from kli import __version__
from kli.assemble_prompts import (
    DEFAULT_PERSONA_HEADER,
    assemble_prompt,
    check_arm_consistency,
    extract_pid,
    find_persona_files,
    load_persona,
    resolve_arm_wording,
    scramble_persona_assignment,
    summarize_arm,
)
from kli.config import ConfigError, get, load_config, section
from kli.parse_responses import aggregate_reports, apply_responses
from kli.providers.local import LocalModelClient, LocalModelConfig
from kli.render_questions import build_question_prompt, chunk_refs
from kli.survey_tree import RoundtripError, walk_questions

__all__ = ["RunSettings", "run", "main"]


@dataclass
class RunSettings:
    """Everything one simulation run needs, resolved from the config."""

    question_json_dir: Path
    output_json_dir: Path
    prompt_dir: Path | None
    persona_dir: Path | None
    empty_persona: bool
    scramble_personas: bool
    scramble_seed: int
    persona_header: str
    separator: str
    prompt_end: str | None
    system_instruction: str
    include_reasoning: bool
    control_wording: str | None
    persona_wording: bool
    wording_note: str | None
    questions_per_call: int | None
    model: LocalModelConfig
    num_workers: int
    max_participants: int | None
    force_regenerate: bool
    on_failure: str

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> RunSettings:
        """Resolve settings from a study config.

        Args:
            config: Parsed study YAML.

        Returns:
            Resolved settings.

        Raises:
            ConfigError: If a required key is missing, or the provider is not
                ``local``.
        """
        sim = section(config, "text_simulation.LLM_simulation")
        combine = section(config, "text_simulation.create_text_simulation_input")
        questions = section(config, "text_simulation.questions_to_texts")
        llm = section(config, "text_simulation.LLM_simulation.LLM_config")

        provider = str(llm.get("provider", "local")).lower()
        if provider != "local":
            raise ConfigError(
                f"LLM_config.provider is {provider!r}, but this pipeline only "
                "supports 'local'. Participant data is kept on this machine by "
                "design -- see kli/providers/local.py. Set provider: local."
            )

        empty_persona = bool(combine.get("empty_persona_prompt", False))
        persona_dir = combine.get("persona_text_dir")

        model = LocalModelConfig(
            model_name=str(llm.get("model_name", "mistral")),
            base_url=str(llm.get("base_url", "http://localhost:11434")),
            api_style=str(llm.get("api_style", "ollama")),
            temperature=float(llm.get("temperature", 0.0)),
            max_tokens=int(llm.get("max_tokens", 8192)),
            max_retries=int(llm.get("max_retries", 5)),
            request_timeout=float(llm.get("request_timeout", 900.0)),
            allow_remote_endpoint=bool(sim.get("allow_remote_endpoint", False)),
        )

        max_personas = sim.get("max_personas", -1)
        try:
            max_personas = int(max_personas)
        except (TypeError, ValueError):
            max_personas = -1

        # Resolve the arm's prompt strings. The DEFAULT for the control arm is
        # 'upstream' -- it keeps the persona-arm wording, exactly as Peng et al.
        # ran it, so results stay comparable to the published numbers. See the
        # "control-arm wording" section of kli/assemble_prompts.py for why that
        # is both the stricter design and arguably a confound, and why the
        # difference between the two wordings is itself worth measuring.
        wording = resolve_arm_wording(
            empty_persona=empty_persona,
            control_wording=str(sim.get("control_wording", "upstream")),
            persona_header=combine.get("persona_prompt_header"),
            separator=combine.get("persona_question_prompt_separator"),
            system_instruction=sim.get("system_instruction"),
        )

        return cls(
            question_json_dir=Path(
                get(config, "text_simulation.LLM_simulation.question_json_base_dir")
            ),
            output_json_dir=Path(
                get(
                    config,
                    "text_simulation.LLM_simulation.output_updated_questions_dir",
                )
            ),
            prompt_dir=(
                Path(combine["output_combined_prompts_dir"])
                if combine.get("output_combined_prompts_dir")
                else None
            ),
            persona_dir=Path(persona_dir) if persona_dir and not empty_persona else None,
            empty_persona=empty_persona,
            scramble_personas=bool(combine.get("scramble_personas", False)),
            scramble_seed=int(combine.get("scramble_seed", 42)),
            persona_header=str(wording["persona_header"]).strip(),
            separator=str(wording["separator"]).strip(),
            prompt_end=combine.get("combine_prompt_end") or None,
            system_instruction=str(wording["system_instruction"]).strip(),
            include_reasoning=bool(questions.get("include_reasoning", False)),
            control_wording=wording["control_wording"],
            persona_wording=bool(wording["persona_wording"]),
            wording_note=wording["note"],
            questions_per_call=(
                int(sim["questions_per_call"])
                if sim.get("questions_per_call")
                else None
            ),
            model=model,
            num_workers=int(sim.get("num_workers", 4) or 4),
            max_participants=None if max_personas < 0 else max_personas,
            force_regenerate=bool(sim.get("force_regenerate", False)),
            on_failure=str(sim.get("on_failure", "blank")),
        )


def _stub_response(
    chunk: list[Any], skeleton: dict[str, Any]
) -> dict[str, Any]:
    """Fill a response skeleton with valid values, without calling a model.

    Used by ``--stub-model`` to exercise prompt assembly, parsing, validation,
    write-back and CSV conversion with no model and no network.

    Values are read off the *instrument* -- option counts, scale bounds,
    constant-sum totals -- so a stub run should reach 100% imputation. That is
    the point: any failure in a stub run is a bug in this pipeline, not a
    limitation of the model. It is a test of plumbing, never of accuracy.
    """
    by_label = {ref.label: ref for ref in chunk if not ref.is_descriptive}
    filled: dict[str, Any] = {}

    for label, entry in skeleton.items():
        ref = by_label.get(label)
        answers = dict(entry.get("Answers", {}))
        question = ref.question if ref is not None else {}
        selector = str(question.get("Settings", {}).get("Selector", "") or "").upper()
        qtype = ref.question_type if ref is not None else ""
        new: dict[str, Any] = {}

        if qtype == "MC":
            options = question.get("Options") or []
            label_text = options[0] if options else "stub"
            if isinstance(answers.get("SelectedByPosition"), list):
                new = {"SelectedByPosition": [1], "SelectedText": [label_text]}
            else:
                new = {"SelectedByPosition": 1, "SelectedText": label_text}

        elif qtype == "Matrix":
            count = len(answers.get("SelectedByPosition") or [None])
            columns = question.get("Columns") or []
            first = columns[0] if columns else "stub"
            new = {
                "SelectedByPosition": [1] * count,
                "SelectedText": [first] * count,
            }

        elif qtype == "Slider":
            count = len(answers.get("Values") or [None])
            rng = question.get("Range") or {}
            try:
                value = str(int(float(rng.get("Min", 0))))
            except (TypeError, ValueError):
                value = "0"
            new = {"Values": [value] * count}

        elif qtype == "CS":
            count = len(answers.get("Values") or [None]) or 1
            try:
                total = float(question.get("Settings", {}).get("ChoiceTotal", 0) or 0)
            except (TypeError, ValueError):
                total = 0.0
            # Put the whole total on the first item so the sum constraint holds
            # exactly, regardless of how many items there are.
            values = ["0"] * count
            values[0] = str(int(total))
            new = {"Values": values}

        elif qtype == "TE":
            if selector == "FORM":
                count = len(answers.get("Text") or [None])
                new = {"Text": ["stub"] * count}
            elif selector == "ESTB":
                new = {"Text": "1"}
            else:
                new = {"Text": "stub"}

        else:
            new = answers

        out: dict[str, Any] = {"QuestionID": entry["QuestionID"], "Answers": new}
        if "Reasoning" in entry:
            out["Reasoning"] = "stub reasoning"
        filled[label] = out

    return filled


def _simulate_one(
    path: Path,
    settings: RunSettings,
    personas: dict[str, Path],
    client: LocalModelClient | None,
    *,
    dry_run: bool,
    stub: bool,
) -> tuple[str, Any]:
    """Simulate one participant. Returns ``(pid, report | Exception)``."""
    pid = extract_pid(path)
    if pid is None:
        return path.name, ValueError("no participant id in filename")

    try:
        document = json.loads(path.read_text(encoding="utf-8"))
        refs = list(walk_questions(document))

        persona_text: str | None = None
        if not settings.empty_persona:
            persona_path = personas.get(pid)
            if persona_path is None:
                return pid, FileNotFoundError(
                    f"no persona file for pid {pid} in {settings.persona_dir}"
                )
            persona_text = load_persona(persona_path)

        chunks = chunk_refs(refs, settings.questions_per_call)
        merged: dict[str, Any] = {}

        for index, chunk in enumerate(chunks, 1):
            question_prompt, skeleton = build_question_prompt(
                chunk,
                include_reasoning=settings.include_reasoning,
                persona_wording=settings.persona_wording,
            )
            if not skeleton:
                continue

            prompt = assemble_prompt(
                question_prompt=question_prompt,
                persona_text=persona_text,
                empty_persona=settings.empty_persona,
                persona_header=settings.persona_header,
                separator=settings.separator,
                prompt_end=settings.prompt_end,
            )

            if settings.prompt_dir is not None:
                settings.prompt_dir.mkdir(parents=True, exist_ok=True)
                suffix = f"_chunk{index}" if len(chunks) > 1 else ""
                (settings.prompt_dir / f"pid_{pid}{suffix}_prompt.txt").write_text(
                    prompt, encoding="utf-8"
                )

            if dry_run:
                continue
            if stub:
                merged.update(_stub_response(chunk, skeleton))
                continue

            assert client is not None
            response, _meta = client.complete_json(
                system=settings.system_instruction, user=prompt
            )
            merged.update(response)

        if dry_run:
            return pid, None

        report = apply_responses(
            [r for r in refs if not r.is_descriptive],
            merged,
            participant_id=pid,
            on_failure=settings.on_failure,  # type: ignore[arg-type]
        )

        settings.output_json_dir.mkdir(parents=True, exist_ok=True)
        (settings.output_json_dir / f"pid_{pid}_response.json").write_text(
            json.dumps(document, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        return pid, report

    except RoundtripError:
        raise
    except Exception as exc:  # noqa: BLE001 - collected into the run report
        return pid, exc


def run(
    config_path: str | Path,
    *,
    dry_run: bool = False,
    stub: bool = False,
    limit: int | None = None,
) -> dict[str, Any]:
    """Execute a simulation run.

    Args:
        config_path: Study config path.
        dry_run: Build prompts, call no model, write no imputed JSON.
        stub: Fill skeletons locally instead of calling a model. Exercises the
            full parse/validate/write path offline.
        limit: Cap participants, overriding ``max_personas``.

    Returns:
        The run manifest, also written to ``run_manifest.json``.

    Raises:
        ConfigError: On a bad config.
        FileNotFoundError: If no input response JSONs are found.
    """
    config = load_config(config_path)
    settings = RunSettings.from_config(config)

    paths = sorted(settings.question_json_dir.glob("*.json"))
    if not paths:
        raise FileNotFoundError(
            f"No response JSONs in {settings.question_json_dir}. Run the "
            "Qualtrics stages first:\n"
            f"  python processing_qualtrics_qsf/parse_qsf.py --config {config_path}\n"
            f"  python processing_qualtrics_csv/get_responses.py --config {config_path}"
        )

    cap = limit if limit is not None else settings.max_participants
    if cap is not None:
        paths = paths[:cap]

    if not settings.force_regenerate and not dry_run:
        done = {
            extract_pid(p) for p in settings.output_json_dir.glob("*.json")
        } - {None}
        before = len(paths)
        paths = [p for p in paths if extract_pid(p) not in done]
        if before != len(paths):
            print(
                f"skipping {before - len(paths)} participant(s) already done "
                "(force_regenerate: false)"
            )

    personas = (
        find_persona_files(settings.persona_dir)
        if settings.persona_dir is not None
        else {}
    )
    if settings.scramble_personas and personas:
        personas = scramble_persona_assignment(
            personas, seed=settings.scramble_seed
        )
        print(
            "\nNOTE: SCRAMBLED-PERSONA CONTROL is active "
            f"(scramble_seed: {settings.scramble_seed}).\n"
            "  Every participant receives a real persona belonging to someone\n"
            "  else. Comparing this arm against the matched-persona arm\n"
            "  isolates person-specific signal from the generic benefit of\n"
            "  having a persona at all. This is NOT a twin run -- do not report\n"
            "  its output as twin accuracy.\n"
        )

    if not settings.empty_persona and not personas:
        raise FileNotFoundError(
            f"No persona .txt files in {settings.persona_dir}. For the "
            "no-persona control arm set 'empty_persona_prompt: true' instead."
        )

    if settings.wording_note:
        print(f"\nNOTE: {settings.wording_note}\n")

    for warning in check_arm_consistency(
        empty_persona=settings.empty_persona,
        system_instruction=settings.system_instruction,
        separator=settings.separator,
        control_wording=settings.control_wording or "upstream",
    ):
        print(f"WARNING: {warning}", file=sys.stderr)

    client: LocalModelClient | None = None
    health: dict[str, Any] | None = None
    if not dry_run and not stub:
        client = LocalModelClient(settings.model)
        health = client.health_check()
        if not health["reachable"]:
            raise RuntimeError(
                f"Local model server unreachable at {settings.model.base_url}: "
                f"{health.get('error')}\nStart it with: ollama serve"
            )
        if not health["model_available"]:
            raise RuntimeError(
                f"Model {settings.model.model_name!r} is not installed on the "
                f"local server. Available: {health['models']}\n"
                f"Install it with: ollama pull {settings.model.model_name}"
            )

    arm = summarize_arm(
        empty_persona=settings.empty_persona,
        persona_dir=settings.persona_dir,
        scrambled=settings.scramble_personas,
    )
    mode = "dry-run" if dry_run else ("stub-model" if stub else "live")
    print(
        f"running {len(paths)} participant(s) | arm: {arm} | mode: {mode} | "
        f"model: {settings.model.model_name} @ T={settings.model.temperature}"
    )

    started = time.monotonic()
    reports: list[Any] = []
    errors: dict[str, str] = {}

    workers = 1 if (dry_run or stub) else max(1, settings.num_workers)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [
            pool.submit(
                _simulate_one,
                path,
                settings,
                personas,
                client,
                dry_run=dry_run,
                stub=stub,
            )
            for path in paths
        ]
        for done_count, future in enumerate(futures, 1):
            pid, result = future.result()
            if isinstance(result, Exception):
                errors[str(pid)] = f"{type(result).__name__}: {result}"
            elif result is not None:
                reports.append(result)
            if done_count % 50 == 0 or done_count == len(futures):
                print(f"  {done_count}/{len(futures)}")

    manifest: dict[str, Any] = {
        "kli_version": __version__,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "config_path": str(config_path),
        "mode": mode,
        "arm": arm,
        "empty_persona_control": settings.empty_persona,
        "scrambled_persona_control": settings.scramble_personas,
        "scramble_seed": settings.scramble_seed if settings.scramble_personas else None,
        "control_wording": settings.control_wording,
        "include_reasoning": settings.include_reasoning,
        "questions_per_call": settings.questions_per_call,
        "on_failure": settings.on_failure,
        "system_instruction": settings.system_instruction,
        "separator": settings.separator,
        "persona_header": (
            None
            if (settings.empty_persona and not settings.persona_wording)
            else settings.persona_header
        ),
        "persona_wording_refers_to_persona": settings.persona_wording,
        "model": settings.model.manifest(),
        "model_health": health,
        "participants_attempted": len(paths),
        "participants_failed": len(errors),
        "errors": errors,
        "elapsed_seconds": round(time.monotonic() - started, 1),
    }
    if reports:
        manifest["imputation"] = aggregate_reports(reports)

    settings.output_json_dir.mkdir(parents=True, exist_ok=True)
    (settings.output_json_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )

    if "imputation" in manifest:
        imputation = manifest["imputation"]
        print(
            f"imputed {imputation['questions_imputed']}/"
            f"{imputation['questions_total']} questions "
            f"({imputation['imputation_rate']:.1%}); "
            f"{imputation['participants_complete']}/{imputation['participants']} "
            "participants complete"
        )
        if imputation["failure_reasons"]:
            print(f"failure reasons: {imputation['failure_reasons']}")
    if errors:
        print(f"{len(errors)} participant(s) errored; see run_manifest.json")

    return manifest


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Run digital twin simulation against a local model."
    )
    parser.add_argument("--config", required=True, help="Study YAML config.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Build and save prompts without calling a model.",
    )
    parser.add_argument(
        "--stub-model",
        action="store_true",
        help="Fill skeletons locally; exercises parsing and write-back offline.",
    )
    parser.add_argument(
        "--health-check",
        action="store_true",
        help="Check the local model server and exit.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Cap participants.")
    args = parser.parse_args(argv)

    if args.health_check:
        config = load_config(args.config)
        settings = RunSettings.from_config(config)
        health = LocalModelClient(settings.model).health_check()
        print(json.dumps({"endpoint": settings.model.base_url, **health}, indent=2))
        return 0 if health.get("model_available") else 1

    run(
        args.config,
        dry_run=args.dry_run,
        stub=args.stub_model,
        limit=args.limit,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
