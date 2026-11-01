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

"""Tests for the kuna exporter (quokka.backends.kuna).

These tests run the real ``kuna`` executable and skip when it (or LIEF) is
not installed. The produced files are loaded back through the regular Python
bindings and, where an IDA export of the same binary is shipped in the
dataset, compared against it.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

import quokka
from quokka.types import Disassembler, ExporterMode, FunctionType, OperandType

pytest.importorskip("lief")
from quokka.backends.kuna import KunaError, export_binary, find_kuna  # noqa: E402

if find_kuna() is None:
    pytest.skip("kuna executable not found", allow_module_level=True)

Pb = quokka.pb.Quokka


@pytest.fixture(scope="module")
def kuna_export(root_directory: Path, tmp_path_factory: pytest.TempPathFactory):
    """Export (and cache) a binary of the repository with kuna."""
    out_dir = tmp_path_factory.mktemp("kuna")
    cache: dict[tuple[str, int], quokka.Program] = {}

    def _export(relative: str, mode: int = Pb.ExporterMeta.MODE_LIGHT) -> quokka.Program:
        if (relative, mode) not in cache:
            binary = root_directory / relative
            if not binary.exists():
                pytest.skip(f"{relative} not found")
            output = out_dir / f"{binary.name}.{mode}.quokka"
            export_binary(binary, output, mode=mode)
            cache[relative, mode] = quokka.Program(output, binary)
        return cache[relative, mode]

    return _export


def _reference(root_directory: Path, relative: str) -> quokka.Program:
    binary = root_directory / relative
    reference = binary.with_suffix(".quokka")
    if not reference.exists():
        pytest.skip(f"{reference.name} not found")
    return quokka.Program(reference, binary)


def _normal_functions(program: quokka.Program) -> dict[int, quokka.Function]:
    return {f.start: f for f in program.values() if f.type == FunctionType.NORMAL}


@pytest.mark.parametrize(
    "relative",
    ["docs/samples/qb-crackme", "tests/dataset/many_types_cpp", "tests/dataset/puraUpdate"],
)
def test_export_structure(kuna_export, relative: str):
    prog = kuna_export(relative)
    assert prog.disassembler == Disassembler.KUNA
    assert prog.mode == ExporterMode.LIGHT
    digest = hashlib.sha256(Path(prog.executable.exec_file).read_bytes()).hexdigest()
    assert prog.proto.meta.hash.hash_value == digest

    normal = _normal_functions(prog)
    assert normal, "no function exported"
    for func in normal.values():
        assert len(func) > 0, f"{func.name} has no block"
        blocks = func.keys()  # Function is lazily populated: use keys()
        assert func.start in blocks, f"{func.name} has no entry block"
        for src, dst in func.graph.edges:
            assert src in blocks and dst in blocks
        for block in func.values():
            assert block.instructions, f"empty block at 0x{block.start:x}"

    # Imports are exported as functions without body
    imported = [f for f in prog.values() if f.type == FunctionType.IMPORTED]
    assert imported
    assert all(len(f) == 0 for f in imported)


def test_crackme_names_and_calls(kuna_export):
    prog = kuna_export("docs/samples/qb-crackme")
    main = prog.get_function("main", approximative=False)
    callees = {f.name for f in main.callees}
    assert callees == {"cprintf"}  # same as IDA
    # Callers are the reverse relation
    for callee in main.callees:
        assert main.start in {f.start for f in callee.callers}


def test_crackme_strings(kuna_export):
    prog = kuna_export("docs/samples/qb-crackme")
    strings = {d.value for d in prog.data if d.size > 0 and isinstance(d.value, bytes)}
    assert b"What's the flag?\x00" in {s for s in strings if s.startswith(b"What")}


@pytest.mark.parametrize("relative", ["tests/dataset/many_types_cpp", "tests/dataset/puraUpdate"])
def test_cfg_matches_ida(kuna_export, root_directory: Path, relative: str):
    """Basic blocks and CFG edges agree with the IDA export of the dataset."""
    ours = _normal_functions(kuna_export(relative))
    ref = _normal_functions(_reference(root_directory, relative))
    common = ours.keys() & ref.keys()
    # IDA additionally lists a few PLT/init stubs as normal functions
    assert len(common) >= 0.85 * len(ref)

    identical = 0
    for addr in common:
        fo, fr = ours[addr], ref[addr]
        identical += set(fo.keys()) == set(fr.keys()) and set(fo.graph.edges) == set(fr.graph.edges)
    assert identical >= 0.95 * len(common), f"{identical}/{len(common)} identical CFGs"


@pytest.mark.parametrize("relative", ["tests/dataset/many_types_cpp", "tests/dataset/puraUpdate"])
def test_call_graph_matches_ida(kuna_export, root_directory: Path, relative: str):
    def call_edges(program: quokka.Program) -> set[tuple[int, int]]:
        return {(f.start, c.start) for f in _normal_functions(program).values() for c in f.callees}

    ours = call_edges(kuna_export(relative))
    ref = call_edges(_reference(root_directory, relative))
    assert len(ours & ref) >= 0.7 * len(ref)


def test_arm32_thumb_flags(kuna_export, root_directory: Path):
    """The Thumb state of every block matches IDA on the ARM32 sample."""
    ours = _normal_functions(kuna_export("tests/dataset/puraUpdate"))
    ref = _normal_functions(_reference(root_directory, "tests/dataset/puraUpdate"))
    for addr in ours.keys() & ref.keys():
        for start in ours[addr].keys() & ref[addr].keys():
            assert ours[addr][start].proto.is_thumb == ref[addr][start].proto.is_thumb


def test_full_mode(kuna_export):
    prog = kuna_export("tests/dataset/many_types_cpp", Pb.ExporterMeta.MODE_SELF_CONTAINED)
    assert prog.mode == ExporterMode.FULL
    main = prog.get_function("main", approximative=False)
    seen_register = False
    for block in main.values():
        for inst in block.instructions:
            assert inst.mnemonic
            assert str(inst).startswith(inst.mnemonic)
            for op in inst.operands:
                if op.type == OperandType.REGISTER:
                    assert op.register and op.register == op.value
                    seen_register = True
    assert seen_register


def test_relocatable_object_rejected(root_directory: Path, tmp_path: Path):
    binary = root_directory / "tests/dataset/nf_nat_pt.ko"
    if not binary.exists():
        pytest.skip("nf_nat_pt.ko not found")
    with pytest.raises(KunaError):
        export_binary(binary, tmp_path / "out.quokka")
