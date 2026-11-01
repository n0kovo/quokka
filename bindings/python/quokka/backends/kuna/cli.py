"""Thin wrapper around the ``kuna`` command line interface.

Kuna (https://github.com/Noelo-Lab/kuna) is a SLEIGH based decompiler shipped
as a single CLI binary. It keeps no state between invocations, so everything the
exporter needs is pulled out of a handful of JSON documents:

* ``kuna decompile-graph``: the function inventory (names, extents, kinds, entry
  point, forwarding thunks), the call graph and, optionally, the decompiled C
  and recovered prototypes;
* ``kuna disassemble --json``: the instruction stream (address, size, bytes and
  text) of every executable section;
* ``kuna strings --json``: the string literals.
"""

#  Copyright 2022-2026 Quarkslab
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      https://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from quokka.exc import QuokkaError

logger = logging.getLogger(__name__)

#: Environment variable that can point to the kuna executable.
KUNA_PATH_ENV = "KUNA_PATH"


class KunaError(QuokkaError):
    """Raised when kuna is missing or one of its commands fails."""


def find_kuna(kuna_path: Path | str | None = None) -> Path | None:
    """Locate the kuna executable.

    Resolution order: explicit argument, ``$KUNA_PATH``, then ``PATH``.
    ``$KUNA_PATH`` may point either to the binary itself or to a directory
    containing it.

    Returns:
        The path to the executable or None if it cannot be found.
    """
    candidates: list[Path] = []
    if kuna_path is not None:
        candidates.append(Path(kuna_path))
    if env := os.environ.get(KUNA_PATH_ENV):
        candidates.append(Path(env))
    for candidate in candidates:
        if candidate.is_dir():
            candidate = candidate / "kuna"
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return candidate
    if found := shutil.which("kuna"):
        return Path(found)
    return None


@dataclass
class KunaInstruction:
    """One row of a ``kuna disassemble --json`` listing."""

    address: int
    size: int
    raw: bytes
    mnemonic: str
    operands: str

    @property
    def end(self) -> int:
        return self.address + self.size

    @property
    def is_data(self) -> bool:
        """``.byte``/``.word``... rows: bytes kuna refused to decode or folded
        as a literal pool constant."""
        return self.mnemonic.startswith(".")


@dataclass
class KunaFunction:
    """One node of the ``kuna decompile-graph`` document."""

    address: int
    name: str
    size: int
    kind: str  # normal | thunk | import | data | external
    signature: str | None = None
    code: str | None = None
    error: str | None = None
    forwards_to: int | None = None
    is_entry_point: bool = False
    has_indirect_calls: bool = False

    @property
    def end(self) -> int:
        return self.address + self.size


@dataclass
class KunaEdge:
    caller: int
    callee: int
    kind: str  # call | jump | data


@dataclass
class KunaGraph:
    functions: list[KunaFunction]
    edges: list[KunaEdge]
    image_base: int | None
    raw: dict[str, Any] = field(repr=False, default_factory=dict)


@dataclass
class KunaString:
    address: int
    text: str
    byte_length: int
    encoding: str
    nul_terminated: bool


class KunaCli:
    """Run kuna sub-commands and decode their JSON output."""

    def __init__(
        self,
        exec_path: Path | str,
        kuna_path: Path | str | None = None,
        timeout: int | None = None,
        extra_args: list[str] | None = None,
    ):
        found = find_kuna(kuna_path)
        if found is None:
            raise KunaError(
                "kuna executable not found. Install it from "
                "https://github.com/Noelo-Lab/kuna and put it on PATH, "
                f"or set ${KUNA_PATH_ENV}."
            )
        self.kuna: Path = found
        self.exec_path = Path(exec_path)
        self.timeout = timeout if timeout else None
        # Arguments shared by every analysis command (e.g. --mode, --option).
        self.extra_args: list[str] = list(extra_args or [])

    # ------------------------------------------------------------------ #
    # Low level helpers
    # ------------------------------------------------------------------ #
    def _run(self, args: list[str]) -> str:
        cmd = [str(self.kuna), *args]
        logger.debug("Running %s", " ".join(cmd))
        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.timeout,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise KunaError(f"kuna timed out after {self.timeout}s: {' '.join(args[:2])}") from exc
        for line in proc.stderr.splitlines():
            if line.strip():
                logger.debug("kuna: %s", line)
        if proc.returncode != 0:
            msg = proc.stderr.strip().splitlines()
            raise KunaError(
                f"kuna {args[0]} failed (exit {proc.returncode}): "
                + (msg[-1] if msg else "no error message")
            )
        return proc.stdout

    def _run_json(self, args: list[str]) -> Any:
        out = self._run(args)
        try:
            return json.loads(out)
        except json.JSONDecodeError as exc:
            raise KunaError(f"kuna {args[0]} returned invalid JSON") from exc

    # ------------------------------------------------------------------ #
    # Commands
    # ------------------------------------------------------------------ #
    def version(self) -> str:
        """Return the kuna version string (e.g. ``"1.984"``)."""
        out = self._run(["--version"]).strip()
        if match := re.search(r"(\d+(?:\.\d+)+)", out):
            return match.group(1)
        return out

    def graph(self, decompile: bool = False, jobs: str | int | None = None) -> KunaGraph:
        """Run ``decompile-graph``.

        Without ``decompile`` no body is decompiled (``--functions ''`` keeps
        every node and edge but selects no body), which makes the command
        cost a single load + discovery pass.
        """
        args = ["decompile-graph", str(self.exec_path), *self.extra_args]
        if not decompile:
            args += ["--functions", ""]
        elif jobs is not None:
            args += ["--jobs", str(jobs)]
        doc = self._run_json(args)

        functions = [
            KunaFunction(
                address=int(row["address"]),
                name=row.get("name") or f"sub_{int(row['address']):x}",
                size=int(row.get("size") or 0),
                kind=row.get("kind") or "normal",
                signature=row.get("signature"),
                code=row.get("codeC"),
                error=row.get("error"),
                forwards_to=row.get("forwardsTo"),
                is_entry_point=bool(row.get("isEntryPoint")),
                has_indirect_calls=bool(row.get("hasIndirectCalls")),
            )
            for row in doc.get("functions", [])
        ]
        edges = [
            KunaEdge(int(e["callerAddress"]), int(e["calleeAddress"]), e.get("kind", "call"))
            for e in doc.get("edges", [])
        ]
        base = doc.get("binary", {}).get("analysisImageBase")
        return KunaGraph(functions, edges, base, raw=doc)

    def disassemble(self, start: int, end: int, isa: str | None = None) -> list[KunaInstruction]:
        """Linear disassembly of ``[start, end)`` as code."""
        if end <= start:
            return []
        args = [
            "disassemble",
            str(self.exec_path),
            f"{start:#x}-{end:#x}",
            "--as",
            "code",
            "--json",
            *self.extra_args,
        ]
        if isa is not None:
            args += ["--isa", isa]
        doc = self._run_json(args)
        return [
            KunaInstruction(
                address=int(row["address"]),
                size=int(row["size"]),
                raw=bytes.fromhex(row.get("bytes") or ""),
                mnemonic=row.get("mnemonic") or "",
                operands=row.get("operands") or "",
            )
            for row in doc.get("instructions", [])
            if int(row.get("size") or 0) > 0
        ]

    def strings(self, min_length: int = 4) -> list[KunaString]:
        """String literals (``char[N]`` / UTF-16 NUL terminated runs)."""
        try:
            doc = self._run_json(
                [
                    "strings",
                    str(self.exec_path),
                    "--json",
                    "--no-xrefs",
                    "--termination",
                    "nul",
                    "--encoding",
                    "all",
                    "--min-length",
                    str(min_length),
                ]
            )
        except KunaError as exc:
            logger.warning("String extraction failed: %s", exc)
            return []
        return [
            KunaString(
                address=int(row["address"]),
                text=row.get("text") or "",
                byte_length=int(row.get("byte_length") or 0),
                encoding=row.get("encoding") or "ascii",
                nul_terminated=bool(row.get("nul_terminated")),
            )
            for row in doc.get("strings", [])
        ]


__all__ = [
    "KUNA_PATH_ENV",
    "KunaCli",
    "KunaEdge",
    "KunaError",
    "KunaFunction",
    "KunaGraph",
    "KunaInstruction",
    "KunaString",
    "find_kuna",
]
