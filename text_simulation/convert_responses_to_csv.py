"""Compatibility shim: this stage is now implemented in kli/responses_to_csv.py.

ORIGINAL WORK (Kravis Leadership Institute). See ATTRIBUTION.md.

This path exists because the Mega-Study ``Snakefile`` invokes
``text_simulation/convert_responses_to_csv.py``. The upstream repositories never
published that script; we implemented the stage as
:mod:`kli.responses_to_csv`, and this file forwards to it so the documented
Snakemake path keeps working.

Prefer calling it directly::

    python -m kli.responses_to_csv --config configs/my_study.yaml
"""

from __future__ import annotations

import sys

from kli.responses_to_csv import main

if __name__ == "__main__":
    print(
        "note: forwarding to kli.responses_to_csv "
        "(the real implementation lives there)",
        file=sys.stderr,
    )
    raise SystemExit(main())
