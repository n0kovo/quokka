"""Build a Quokka export from kuna's analysis.

Pipeline:

1. read the image (sections, permissions, file offsets) with LIEF;
2. ``kuna decompile-graph``: function inventory, kinds and call graph
   (optionally the decompiled C and prototypes);
3. ``kuna disassemble``: instruction boundaries of every executable section;
4. per function, rebuild the basic blocks by recursive descent over kuna's
   instructions, classified with capstone (see :mod:`.flow`);
5. derive code and data references, data items (strings, symbols, referenced
   addresses) and write the protobuf.
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

import bisect
import logging
import lzma
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import quokka
from quokka.quokka_pb2 import Quokka as Pb  # pyright: ignore[reportMissingImports]

from .cli import KunaCli, KunaError, KunaFunction, KunaInstruction, KunaString
from .flow import ACCESS_READ, ACCESS_WRITE, Flow, FlowAnalyzer, InsnInfo
from .image import Image, load_image

logger = logging.getLogger(__name__)

#: Functions that never return, matched after stripping decorations
#: (leading underscores, ``@plt``, ``__imp_``, symbol versions).
NORETURN_NAMES = frozenset(
    {
        "abort",
        "exit",
        "_exit",
        "exit_group",
        "quick_exit",
        "_Exit",
        "__assert_fail",
        "__assert_rtn",
        "__assert",
        "assert_fail",
        "__stack_chk_fail",
        "__chk_fail",
        "__fortify_fail",
        "__cxa_throw",
        "__cxa_rethrow",
        "__cxa_bad_cast",
        "__cxa_bad_typeid",
        "__cxa_pure_virtual",
        "__cxa_call_unexpected",
        "_Unwind_Resume",
        "longjmp",
        "_longjmp",
        "siglongjmp",
        "__longjmp_chk",
        "err",
        "errx",
        "verr",
        "verrx",
        "pthread_exit",
        "thrd_exit",
        "ExitProcess",
        "ExitThread",
        "FatalExit",
        "RaiseFailFastException",
        "__fastfail",
        "_invalid_parameter_noinfo_noreturn",
        "__report_gsfailure",
        "_CxxThrowException",
        "std::terminate",
        "terminate",
        "__std_terminate",
        "panic",
        "__libc_fatal",
        "__ubsan_handle_builtin_unreachable",
        "objc_terminate",
        "swift_unexpectedError",
        "__clang_call_terminate",
    }
)
_NORETURN_PREFIXES = ("_ZSt20__throw_", "_ZSt19__throw_", "_ZSt24__throw_", "_ZSt16__throw_",
                      "_ZSt17__throw_", "_ZSt18__throw_", "_ZSt21__throw_", "_ZSt25__throw_",
                      "_ZSt26__throw_", "_ZSt15__throw_", "_ZSt22__throw_", "_ZSt23__throw_",
                      "core::panicking::", "std::__throw_")

#: Bytes of alignment padding: never seeded as switch-case islands.
_PADDING_MNEMONICS = frozenset({"nop", "int3", "hlt", "ud2", "udf", "brk", ".byte", ".word",
                                ".short", ".long", ".quad", ".dword", ".hword", ".2byte",
                                ".4byte", ".8byte"})

_MAX_FALLBACK_DISASSEMBLY = 16  # per function

#: Placeholder names kuna (or Ghidra style tools) give to unnamed items.
_GENERIC_NAME = re.compile(
    r"^(sub|fun|data|dat|off|unk|loc|lab|byte|word|dword|qword|ptr|thunk)_[0-9a-f]+$", re.IGNORECASE
)

_PRIMITIVES = (
    Pb.TYPE_UNK,
    Pb.TYPE_B,
    Pb.TYPE_W,
    Pb.TYPE_DW,
    Pb.TYPE_QW,
    Pb.TYPE_OW,
    Pb.TYPE_FLOAT,
    Pb.TYPE_DOUBLE,
    Pb.TYPE_VOID,
)
_SIZE_TO_PRIMITIVE = {1: Pb.TYPE_B, 2: Pb.TYPE_W, 4: Pb.TYPE_DW, 8: Pb.TYPE_QW, 16: Pb.TYPE_OW}


def _normalize_name(name: str) -> str:
    """Strip decorations to match a symbol against NORETURN_NAMES."""
    name = name.split("@", 1)[0]
    for prefix in ("__imp_", "_imp_", "j_", "."):
        name = name.removeprefix(prefix)
    return name


def is_noreturn_name(name: str) -> bool:
    name = _normalize_name(name)
    if name.startswith(_NORETURN_PREFIXES) or name.lstrip("_").startswith(
        tuple(p.lstrip("_") for p in _NORETURN_PREFIXES)
    ):
        return True
    candidates = {name, name.lstrip("_"), name.removeprefix("_")}
    return any(c in NORETURN_NAMES for c in candidates)


def demangle(name: str) -> str:
    """Demangle a C++/Rust symbol (Mach-O adds one leading underscore)."""
    raw = name.split("@", 1)[0]
    candidates = [raw]
    if raw.startswith("__Z"):
        candidates.insert(0, raw[1:])
    try:
        import lief  # type: ignore[import-not-found]
    except ImportError:  # pragma: no cover
        return name
    for candidate in candidates:
        if not candidate.startswith(("_Z", "_R", "?")):
            continue
        try:
            out = lief.demangle(candidate)
        except Exception:  # pragma: no cover
            out = None
        if out:
            return out
    return name


def split_operands(text: str) -> list[str]:
    """Split an operand string at top-level commas (not inside [] {} ())."""
    out: list[str] = []
    depth = 0
    current: list[str] = []
    for char in text:
        if char in "[{(":
            depth += 1
        elif char in "]})":
            depth = max(0, depth - 1)
        if char == "," and depth == 0:
            if piece := "".join(current).strip():
                out.append(piece)
            current = []
            continue
        current.append(char)
    if piece := "".join(current).strip():
        out.append(piece)
    return out


# ---------------------------------------------------------------------- #
# Intermediate model
# ---------------------------------------------------------------------- #
@dataclass
class BlockModel:
    start: int
    insns: list[KunaInstruction] = field(default_factory=list)
    infos: list[InsnInfo] = field(default_factory=list)
    block_type: int = Pb.Block.BLOCK_TYPE_NORMAL
    successors: list[tuple[int, int]] = field(default_factory=list)  # (addr, edge type)

    @property
    def end(self) -> int:
        return self.insns[-1].end if self.insns else self.start

    @property
    def size(self) -> int:
        return self.end - self.start


@dataclass
class FunctionModel:
    address: int
    name: str
    mangled_name: str
    function_type: int
    kuna: KunaFunction | None = None
    thumb: bool = False
    blocks: list[BlockModel] = field(default_factory=list)
    is_exported: bool = False


@dataclass
class DataModel:
    address: int
    type_index: int
    size: int
    name: str = ""
    priority: int = 0  # higher wins when two candidates share an address


class _Instructions:
    """Address -> kuna instruction lookup with per-function fallbacks."""

    def __init__(self, cli: KunaCli, image: Image):
        self.cli = cli
        self.image = image
        self.by_addr: dict[int, KunaInstruction] = {}

    def load_sections(self) -> None:
        for section in self.image.code_sections:
            start, end = section.start, section.start + section.file_size
            try:
                rows = self.cli.disassemble(start, end)
            except KunaError as exc:
                logger.warning("Could not disassemble %s: %s", section.name, exc)
                continue
            for row in rows:
                self.by_addr[row.address] = row
            logger.debug("%s: %d instructions", section.name, len(rows))

    def fetch(self, addr: int, local: dict[int, KunaInstruction], limit: int,
              budget: list[int]) -> KunaInstruction | None:
        """Instruction at addr: function local listing first, then the global
        one; disassemble ``[addr, limit)`` on a miss (desynchronized linear
        sweep, code kuna did not mark as code...)."""
        if (insn := local.get(addr)) is not None:
            return insn
        if (insn := self.by_addr.get(addr)) is not None:
            return insn
        if budget[0] <= 0 or not self.image.is_code(addr) or limit <= addr:
            return None
        budget[0] -= 1
        try:
            rows = self.cli.disassemble(addr, min(limit, addr + 0x4000))
        except KunaError as exc:
            logger.debug("Fallback disassembly at %#x failed: %s", addr, exc)
            return None
        for row in rows:
            local.setdefault(row.address, row)
        return local.get(addr)


# ---------------------------------------------------------------------- #
# Exporter
# ---------------------------------------------------------------------- #
class KunaExporter:
    """Turn a kuna analysis into a ``Pb`` (Quokka protobuf) message."""

    def __init__(
        self,
        exec_path: Path,
        mode: int = Pb.ExporterMeta.MODE_LIGHT,
        decompiled: bool = False,
        kuna_path: Path | str | None = None,
        timeout: int | None = None,
        kuna_args: list[str] | None = None,
        jobs: str | int | None = None,
        min_string_length: int = 4,
    ):
        self.exec_path = Path(exec_path)
        self.mode = mode
        self.decompiled = decompiled
        self.jobs = jobs
        self.min_string_length = min_string_length
        self.image = load_image(self.exec_path)
        if self.image.is_relocatable:
            raise KunaError(
                "Relocatable objects (.o/.ko) are not supported by the kuna backend: "
                "their sections are not laid out at distinct addresses. Link the "
                "object or use another backend."
            )
        self.cli = KunaCli(self.exec_path, kuna_path, timeout, kuna_args)
        self.flow = FlowAnalyzer(self.image)
        self.insns = _Instructions(self.cli, self.image)

        self.functions: dict[int, FunctionModel] = {}
        self.import_rows: dict[int, KunaFunction] = {}
        self.data_rows: dict[int, KunaFunction] = {}
        self.noreturn: set[int] = set()
        # (src, dst, type)
        self.references: set[tuple[int, int, int]] = set()
        self.data: dict[int, DataModel] = {}
        self.types: list[Pb.Type] = []
        self._array_types: dict[tuple[int, int], int] = {}
        self.strings: list[KunaString] = []

    # ------------------------------------------------------------------ #
    def run(self) -> Pb:
        version = self.cli.version()
        graph = self.cli.graph(decompile=self.decompiled, jobs=self.jobs)
        logger.info("kuna %s: %d functions, %d edges", version, len(graph.functions),
                    len(graph.edges))

        unmapped = [f.address for f in graph.functions
                    if f.kind in ("external", "import") and not self.image.is_mapped(f.address)]
        self.image.add_extern_section(unmapped)

        self.insns.load_sections()
        self._classify_functions(graph)
        self._find_noreturn()

        for func in sorted(self.functions.values(), key=lambda f: f.address):
            if func.function_type in (Pb.Function.TYPE_NORMAL, Pb.Function.TYPE_THUNK):
                self._build_cfg(func)
                self._function_references(func)
                self.flow.clear_cache()

        self._promote_imports()
        self._build_types_and_data()
        return self._serialize(version)

    # ------------------------------------------------------------------ #
    # Functions
    # ------------------------------------------------------------------ #
    def _merge_import_slots(self, graph: Any) -> None:
        """Turn the loader's import slots into kuna ``import`` rows.

        kuna does not always name imports (e.g. stripped ELF files, where GOT
        slots come back as ``data`` rows named after the PLT stub); LIEF reads
        the relocations / bindings / IAT to recover them.
        """
        slots = self.image.import_slots
        rows = {f.address: f for f in graph.functions}
        for addr, name in slots.items():
            row = rows.get(addr)
            if row is None:
                graph.functions.append(KunaFunction(addr, name, 0, "import"))
            elif row.kind in ("data", "import"):
                if row.kind == "data" or _GENERIC_NAME.match(row.name):
                    row.name = name
                row.kind = "import"

        # Thunks kuna left unnamed: take the name of the import they jump to.
        imports = {f.address: f for f in graph.functions if f.kind in ("import", "external")}
        jumps: dict[int, set[int]] = defaultdict(set)
        for edge in graph.edges:
            if edge.kind in ("jump", "call") and edge.callee in imports:
                jumps[edge.caller].add(edge.callee)
        for row in graph.functions:
            if row.kind != "thunk" or not _GENERIC_NAME.match(row.name):
                continue
            targets = {row.forwards_to} if row.forwards_to in imports else jumps.get(row.address, set())
            if len(targets) == 1:
                row.name = imports[next(iter(targets))].name

    def _classify_functions(self, graph: Any) -> None:
        self._merge_import_slots(graph)
        code_targets = {e.callee for e in graph.edges if e.kind in ("call", "jump")}
        thunk_targets = {f.forwards_to for f in graph.functions
                         if f.kind == "thunk" and f.forwards_to is not None}
        thunk_names = {_normalize_name(f.name).lstrip("_") for f in graph.functions
                       if f.kind == "thunk"}

        for row in graph.functions:
            if row.kind == "data":
                self.data_rows[row.address] = row
                continue
            if row.kind == "import":
                self.import_rows[row.address] = row
                as_function = (
                    row.address in thunk_targets
                    or row.address in code_targets
                    or _normalize_name(row.name).lstrip("_") in thunk_names
                    or (section := self.image.section_at(row.address)) is None
                    or section.kind == Pb.Segment.SEGMENT_EXTERN
                )
                if not as_function:
                    continue
                ftype = Pb.Function.TYPE_IMPORTED
            elif row.kind == "external":
                ftype = Pb.Function.TYPE_IMPORTED
            elif row.kind == "thunk":
                ftype = Pb.Function.TYPE_THUNK
            else:
                ftype = Pb.Function.TYPE_NORMAL
            if not self.image.is_mapped(row.address):
                logger.debug("Skipping unmapped function %s at %#x", row.name, row.address)
                continue
            if ftype in (Pb.Function.TYPE_NORMAL, Pb.Function.TYPE_THUNK) and not self.image.is_code(
                row.address
            ):
                # e.g. functions kuna found in a writable section: keep them,
                # quokka does not care about the permission bits.
                logger.debug("Function %s at %#x is not in a code section", row.name, row.address)
            self.functions[row.address] = FunctionModel(
                address=row.address,
                name=demangle(row.name),
                mangled_name=row.name,
                function_type=ftype,
                kuna=row,
                is_exported=row.address in self.image.exported or row.is_entry_point,
            )

    def _find_noreturn(self) -> None:
        for func in self.functions.values():
            if is_noreturn_name(func.mangled_name) or is_noreturn_name(func.name):
                self.noreturn.add(func.address)
        # Import slots, called through as ``call [rip+slot]`` / ``br x16``.
        for addr, row in self.import_rows.items():
            if is_noreturn_name(row.name):
                self.noreturn.add(addr)
        # A thunk to a noreturn import is noreturn too.
        for func in self.functions.values():
            if func.kuna is not None and func.kuna.forwards_to in self.noreturn:
                self.noreturn.add(func.address)

    def _function_limit(self, func: FunctionModel, starts: list[int]) -> int:
        """End of the address range recursive descent may explore."""
        section = self.image.section_at(func.address)
        section_end = section.start + section.file_size if section else func.address
        if func.kuna is not None and func.kuna.size > 0:
            return min(func.address + func.kuna.size, section_end)
        idx = bisect.bisect_right(starts, func.address)
        nxt = starts[idx] if idx < len(starts) else section_end
        return min(nxt, section_end)

    def _is_thumb(self, func: FunctionModel, first: KunaInstruction | None) -> bool:
        if self.flow.cs_thumb is None:
            return False
        state = self.image.is_thumb(func.address)
        if state is not None:
            return state
        return first is not None and first.size == 2

    # ------------------------------------------------------------------ #
    # CFG
    # ------------------------------------------------------------------ #
    def _build_cfg(self, func: FunctionModel) -> None:
        if not hasattr(self, "_starts"):
            self._starts = sorted(self.functions)
        limit = self._function_limit(func, self._starts)
        local: dict[int, KunaInstruction] = {}
        budget = [_MAX_FALLBACK_DISASSEMBLY]

        def fetch(addr: int) -> KunaInstruction | None:
            return self.insns.fetch(addr, local, max(limit, addr + 16), budget)

        first = fetch(func.address)
        func.thumb = self._is_thumb(func, first)
        if first is None:
            logger.debug("No instruction at function %s (%#x)", func.name, func.address)
            return

        lo, hi = func.address, max(limit, first.end)
        inside = lambda a: lo <= a < hi

        def is_other_function(addr: int) -> bool:
            return addr != func.address and addr in self.functions

        visited: dict[int, InsnInfo] = {}
        insn_at: dict[int, KunaInstruction] = {}
        leaders: set[int] = {func.address}
        ends: set[int] = set()  # addresses of block terminating instructions
        succ: dict[int, list[tuple[int, int]]] = {}
        btype: dict[int, int] = {}

        def explore(root: int) -> None:
            work = [root]
            while work:
                addr = work.pop()
                while True:
                    if addr in visited:
                        leaders.add(addr)
                        break
                    insn = fetch(addr)
                    if insn is None or insn.is_data or not inside(addr):
                        break
                    info = self.flow.analyze(insn, func.thumb)
                    visited[addr] = info
                    insn_at[addr] = insn
                    nxt = insn.end
                    # MIPS / SPARC style delay slots belong to the branch block.
                    last = addr
                    for _ in range(info.delay_slots):
                        slot_insn = fetch(nxt)
                        if slot_insn is None or slot_insn.is_data or not inside(nxt):
                            break
                        visited[nxt] = InsnInfo()
                        insn_at[nxt] = slot_insn
                        last = nxt
                        nxt = slot_insn.end

                    flow = info.flow
                    if flow == Flow.NORMAL and last == addr:
                        addr = nxt
                        continue
                    if flow == Flow.CALL:
                        noret = (info.target is not None and info.target in self.noreturn) or (
                            info.slot is not None and info.slot in self.noreturn
                        )
                        after = fetch(nxt) if nxt < hi else None
                        if not noret and after is not None and not after.is_data:
                            if last != addr:
                                ends.add(last)
                                succ[last] = [(nxt, Pb.EDGE_JUMP_UNCOND)]
                                leaders.add(nxt)
                                work.append(nxt)
                                break
                            addr = nxt
                            continue
                        ends.add(last)
                        succ[last] = []
                        btype[last] = Pb.Block.BLOCK_TYPE_NORET
                        break
                    ends.add(last)
                    edges: list[tuple[int, int]] = []
                    if flow == Flow.NORMAL:  # delay slot after a non branch (should not happen)
                        edges.append((nxt, Pb.EDGE_JUMP_UNCOND))
                    elif flow in (Flow.JUMP, Flow.CJUMP):
                        cond = flow == Flow.CJUMP
                        etype = Pb.EDGE_JUMP_COND if cond else Pb.EDGE_JUMP_UNCOND
                        tgt = info.target
                        if tgt is not None and inside(tgt) and not is_other_function(tgt):
                            edges.append((tgt, etype))
                        elif not cond and (tgt in self.noreturn or info.slot in self.noreturn):
                            btype[last] = Pb.Block.BLOCK_TYPE_NORET
                        elif info.indirect and not cond:
                            btype[last] = Pb.Block.BLOCK_TYPE_INDJUMP
                        if cond:
                            edges.append((nxt, Pb.EDGE_JUMP_COND))
                    elif flow == Flow.RET:
                        btype[last] = Pb.Block.BLOCK_TYPE_RET
                    elif flow == Flow.CRET:
                        btype[last] = Pb.Block.BLOCK_TYPE_CNDRET
                        edges.append((nxt, Pb.EDGE_JUMP_COND))
                    elif flow == Flow.TRAP:
                        btype[last] = Pb.Block.BLOCK_TYPE_NORET
                    edges = [(a, t) for a, t in edges if inside(a)]
                    succ[last] = edges
                    for target, _ in edges:
                        leaders.add(target)
                        work.append(target)
                    break

        explore(func.address)

        # Switch cases and other code only reachable through computed jumps:
        # seed every unreached, non padding instruction run of the function.
        has_indjump = Pb.Block.BLOCK_TYPE_INDJUMP in btype.values()
        if (
            has_indjump
            and func.function_type == Pb.Function.TYPE_NORMAL
            and func.kuna is not None
            and func.kuna.size > 0
            and len(visited) < 4096
        ):
            for _ in range(64):
                seed = self._find_island(lo, hi, visited, fetch)
                if seed is None:
                    break
                leaders.add(seed)
                explore(seed)

        func.blocks = self._split_blocks(visited, insn_at, leaders, ends, succ, btype)

    def _find_island(
        self, lo: int, hi: int, visited: dict[int, InsnInfo], fetch: Any
    ) -> int | None:
        addr = lo
        covered = sorted(visited)
        idx = 0
        while addr < hi:
            while idx < len(covered) and covered[idx] < addr:
                idx += 1
            if idx < len(covered) and covered[idx] == addr:
                insn = fetch(addr)
                addr = insn.end if insn is not None else addr + 1
                continue
            insn = fetch(addr)
            if insn is None:
                # Not on an instruction boundary: move to the next known one.
                nxt = self._next_known(addr, hi)
                if nxt is None:
                    return None
                addr = nxt
                continue
            if not self._is_padding(insn):
                return addr
            addr = insn.end
        return None

    def _next_known(self, addr: int, hi: int) -> int | None:
        for probe in range(addr + 1, min(hi, addr + 16)):
            if probe in self.insns.by_addr:
                return probe
        return None

    @staticmethod
    def _is_padding(insn: KunaInstruction) -> bool:
        if insn.is_data:
            return True
        mnem = insn.mnemonic.lower().split(".")[0]
        if mnem in _PADDING_MNEMONICS:
            return True
        if insn.raw and (set(insn.raw) <= {0} or set(insn.raw) <= {0xCC} or set(insn.raw) <= {0x90}):
            return True
        # x86 multi-byte nops (``nop word cs:[rax+rax]``) and ``data16`` prefixes
        return mnem in ("data16", "cs") and "nop" in insn.operands.lower()

    @staticmethod
    def _split_blocks(
        visited: dict[int, InsnInfo],
        insn_at: dict[int, KunaInstruction],
        leaders: set[int],
        ends: set[int],
        succ: dict[int, list[tuple[int, int]]],
        btype: dict[int, int],
    ) -> list[BlockModel]:
        blocks: list[BlockModel] = []
        current: BlockModel | None = None
        for addr in sorted(visited):
            insn = insn_at[addr]
            if current is None or addr in leaders or addr != current.end:
                if current is not None and current.insns and not current.successors \
                        and current.insns[-1].address not in ends:
                    # Split by a leader (or a gap): fall-through edge.
                    if addr == current.end and addr in leaders:
                        current.successors.append((addr, Pb.EDGE_JUMP_UNCOND))
                    elif current.block_type == Pb.Block.BLOCK_TYPE_NORMAL:
                        current.block_type = Pb.Block.BLOCK_TYPE_ERROR
                current = BlockModel(addr)
                blocks.append(current)
            current.insns.append(insn)
            current.infos.append(visited[addr])
            if addr in ends:
                current.successors = list(dict.fromkeys(succ.get(addr, [])))
                current.block_type = btype.get(addr, Pb.Block.BLOCK_TYPE_NORMAL)
                current = None
        if current is not None and current.insns and current.insns[-1].address not in ends:
            # Ran into missing bytes / the function limit while falling through.
            current.block_type = Pb.Block.BLOCK_TYPE_ERROR
        return blocks

    # ------------------------------------------------------------------ #
    # References
    # ------------------------------------------------------------------ #
    def _function_references(self, func: FunctionModel) -> None:
        """Code and data references of one function (second pass: the
        constant tracker restarts at every block boundary)."""
        refs: set[tuple[int, int, int]] = set()
        for block in func.blocks:
            self.flow.reset()
            block.infos = [self.flow.analyze(insn, func.thumb) for insn in block.insns]
            for insn, info in zip(block.insns, block.infos):
                src = insn.address
                if info.target is not None and self.image.is_mapped(info.target):
                    if info.flow == Flow.CALL:
                        refs.add((src, info.target, Pb.EDGE_CALL))
                    elif info.flow == Flow.CJUMP:
                        refs.add((src, info.target, Pb.EDGE_JUMP_COND))
                    elif info.flow == Flow.JUMP:
                        refs.add((src, info.target, Pb.EDGE_JUMP_UNCOND))
                if info.slot is not None and (
                    info.slot in self.import_rows or info.slot in self.functions
                ):
                    rtype = Pb.EDGE_CALL_INDIR if info.flow == Flow.CALL else Pb.EDGE_JUMP_INDIR
                    refs.add((src, info.slot, rtype))
                for addr, access, _size in info.mem_refs:
                    if access & ACCESS_WRITE:
                        refs.add((src, addr, Pb.EDGE_DATA_WRITE))
                    if access & ACCESS_READ or not access:
                        refs.add((src, addr, Pb.EDGE_DATA_READ))
                for addr in info.addr_refs:
                    if info.target is None or addr != info.target:
                        refs.add((src, addr, Pb.EDGE_DATA_INDIR))

        # Thunks kuna resolved but we could not (e.g. register indirect jump).
        kuna = func.kuna
        if kuna is not None and kuna.forwards_to is not None and func.blocks:
            last = func.blocks[0].insns[-1].address
            code = (Pb.EDGE_JUMP_INDIR, Pb.EDGE_JUMP_UNCOND, Pb.EDGE_CALL, Pb.EDGE_CALL_INDIR)
            resolved = any(s == last and t in code for s, _, t in refs)
            if not resolved and self.image.is_mapped(kuna.forwards_to):
                refs.add((last, kuna.forwards_to, Pb.EDGE_JUMP_INDIR))
        self.references |= refs

    def _promote_imports(self) -> None:
        """Import rows targeted by code references become imported functions."""
        code = {Pb.EDGE_CALL, Pb.EDGE_CALL_INDIR, Pb.EDGE_JUMP_INDIR, Pb.EDGE_JUMP_UNCOND}
        targets = {dst for _, dst, t in self.references if t in code}
        for addr, row in self.import_rows.items():
            if addr in self.functions or addr not in targets or not self.image.is_mapped(addr):
                continue
            self.functions[addr] = FunctionModel(
                address=addr,
                name=demangle(row.name),
                mangled_name=row.name,
                function_type=Pb.Function.TYPE_IMPORTED,
                kuna=row,
            )

    # ------------------------------------------------------------------ #
    # Types and data
    # ------------------------------------------------------------------ #
    def _array_type(self, element: int, count: int, c_name: str) -> int:
        key = (element, count)
        if (idx := self._array_types.get(key)) is not None:
            return idx
        idx = len(self.types)
        t = Pb.Type()
        comp = t.composite_type
        comp.name = f"{c_name}[{count}]"
        comp.type = Pb.CompositeType.TYPE_ARRAY
        comp.element_type_idx = element
        comp.size = count * {Pb.TYPE_B: 1, Pb.TYPE_W: 2, Pb.TYPE_DW: 4}.get(element, 1)
        comp.c_str = f"{c_name}[{count}];\n"
        self.types.append(t)
        self._array_types[key] = idx
        return idx

    def _add_data(self, item: DataModel) -> None:
        if not self.image.is_mapped(item.address):
            return
        current = self.data.get(item.address)
        if current is None or item.priority > current.priority:
            if current is not None and not item.name:
                item.name = current.name
            self.data[item.address] = item
        elif item.name and not current.name:
            current.name = item.name

    def _build_types_and_data(self) -> None:
        for prim in _PRIMITIVES:
            self.types.append(Pb.Type(primitive_type=prim))

        # Data kinds kuna knows about
        for addr, row in self.data_rows.items():
            size = max(0, row.size)
            prim = _SIZE_TO_PRIMITIVE.get(size, Pb.TYPE_UNK)
            self._add_data(DataModel(addr, int(prim), size, demangle(row.name), priority=2))
        for addr, row in self.import_rows.items():
            if addr in self.functions:
                continue
            size = self.image.bits // 8
            self._add_data(DataModel(addr, int(_SIZE_TO_PRIMITIVE[size]), size,
                                     demangle(row.name), priority=2))
        for sym in self.image.data_symbols:
            prim = _SIZE_TO_PRIMITIVE.get(sym.size, Pb.TYPE_UNK)
            self._add_data(DataModel(sym.address, int(prim), sym.size, demangle(sym.name),
                                     priority=1))

        # Strings
        self.strings = self.cli.strings(self.min_string_length)
        for string in self.strings:
            if string.byte_length <= 0:
                continue
            wide = "16" in string.encoding or "32" in string.encoding
            unit = 4 if "32" in string.encoding else (2 if wide else 1)
            count = string.byte_length // unit  # byte_length includes the terminator
            if wide:
                element = Pb.TYPE_DW if unit == 4 else Pb.TYPE_W
                tidx = self._array_type(element, count, "wchar_t")
            else:
                tidx = self._array_type(Pb.TYPE_B, count, "char")
            self._add_data(DataModel(string.address, tidx, count * unit, priority=3))

        # Anything else a data reference points at
        sizes: dict[int, int] = {}
        for func in self.functions.values():
            for block in func.blocks:
                for info in block.infos:
                    for addr, _access, size in info.mem_refs:
                        if size:
                            sizes.setdefault(addr, size)
        for _, dst, rtype in self.references:
            if rtype not in (Pb.EDGE_DATA_READ, Pb.EDGE_DATA_WRITE, Pb.EDGE_DATA_INDIR):
                continue
            if dst in self.data or dst in self.functions:
                continue
            section = self.image.section_at(dst)
            if section is None:
                continue
            size = sizes.get(dst, 0)
            if not size and rtype != Pb.EDGE_DATA_INDIR:
                size = 1
            prim = _SIZE_TO_PRIMITIVE.get(size, Pb.TYPE_UNK)
            self._add_data(DataModel(dst, int(prim), size if prim != Pb.TYPE_UNK else 0))

    # ------------------------------------------------------------------ #
    # Serialization
    # ------------------------------------------------------------------ #
    def _serialize(self, kuna_version: str) -> Pb:
        image = self.image
        out = Pb()
        out.exporter_meta.mode = self.mode
        out.exporter_meta.version = quokka.__version__

        meta = out.meta
        meta.executable_name = image.path.name
        meta.isa = image.isa
        meta.endianess = Pb.Meta.END_BE if image.big_endian else Pb.Meta.END_LE
        meta.address_size = image.address_size
        meta.hash.hash_type = Pb.Meta.Hash.HASH_SHA256
        meta.hash.hash_value = image.sha256
        meta.backend.name = Pb.Meta.Backend.DISASS_KUNA
        meta.backend.version = kuna_version
        meta.calling_convention = Pb.CC_UNK
        meta.decompilation_activated = self.decompiled

        # Segments
        seg_index: dict[int, int] = {}
        for i, section in enumerate(image.sections):
            seg = out.segments.add()
            seg.name = section.name
            seg.virtual_addr = section.start
            seg.size = section.size
            seg.permissions = section.perms
            seg.type = section.kind
            seg.address_size = image.address_size
            seg.file_offset = section.file_offset
            seg_index[id(section)] = i

        def locate(addr: int) -> tuple[int, int]:
            idx = image.section_index(addr)
            if idx < 0:
                raise KunaError(f"Address {addr:#x} is outside every segment")
            return idx, addr - image.sections[idx].start

        # References
        ordered_refs = sorted(self.references)
        ref_index: dict[tuple[int, int, int], int] = {}
        refs_from: dict[int, list[int]] = defaultdict(list)
        refs_to: dict[int, list[int]] = defaultdict(list)
        for i, (src, dst, rtype) in enumerate(ordered_refs):
            ref = out.references.add()
            ref.source.address = src
            ref.destination.address = dst
            ref.reference_type = rtype
            ref_index[(src, dst, rtype)] = i
            refs_from[src].append(i)
            refs_to[dst].append(i)

        # Types
        out.types.extend(self.types)

        # Functions
        full = self.mode == Pb.ExporterMeta.MODE_SELF_CONTAINED
        tables = _FullTables(out) if full else None
        for func in sorted(self.functions.values(), key=lambda f: f.address):
            pf = out.functions.add()
            pf.segment_index, pf.segment_offset = locate(func.address)
            pf.file_offset = image.file_offset(func.address)
            pf.function_type = func.function_type
            pf.name = func.name
            if func.mangled_name != func.name:
                pf.mangled_name = func.mangled_name
            pf.is_exported = func.is_exported
            if func.kuna is not None:
                if func.kuna.signature:
                    pf.prototype = func.kuna.signature
                if self.decompiled and func.kuna.code:
                    pf.decompiled_code = func.kuna.code
            block_ids: dict[int, int] = {}
            for block in func.blocks:
                block_ids[block.start] = len(pf.blocks)
                pb = pf.blocks.add()
                pb.segment_index, pb.segment_offset = locate(block.start)
                pb.file_offset = image.file_offset(block.start)
                pb.block_type = block.block_type
                pb.size = block.size
                pb.is_thumb = func.thumb
                pb.n_instr = len(block.insns)
                for idx, insn in enumerate(block.insns):
                    for ref in refs_from.get(insn.address, ()):
                        pb.instructions_xref_from.add(instr_bb_idx=idx, xref_index=ref)
                    for ref in refs_to.get(insn.address, ()):
                        pb.instructions_xref_to.add(instr_bb_idx=idx, xref_index=ref)
                    if tables is not None:
                        pb.instruction_index.append(
                            tables.instruction(insn, func.thumb, self.flow)
                        )
            for block in func.blocks:
                for target, etype in block.successors:
                    if target not in block_ids:
                        continue
                    edge = pf.edges.add()
                    edge.edge_type = etype
                    edge.source = block_ids[block.start]
                    edge.destination = block_ids[target]

        # Data
        for addr in sorted(self.data):
            item = self.data[addr]
            pd = out.data.add()
            pd.segment_index, pd.segment_offset = locate(addr)
            section = image.sections[pd.segment_index]
            pd.file_offset = section.file_offset_of(addr)
            pd.not_initialized = pd.file_offset < 0
            pd.type_index = item.type_index
            pd.size = item.size
            pd.name = item.name
            pd.xref_to.extend(refs_to.get(addr, ()))

        self._layout(out)
        return out

    def _layout(self, out: Pb) -> None:
        ranges: list[tuple[int, int, int]] = []
        for func in self.functions.values():
            for block in func.blocks:
                ranges.append((block.start, block.end, Pb.Layout.LAYOUT_CODE))
        for addr, item in self.data.items():
            ranges.append((addr, addr + max(item.size, 1), Pb.Layout.LAYOUT_DATA))
        ranges.sort()
        merged: list[list[int]] = []
        for start, end, kind in ranges:
            if merged and merged[-1][2] == kind and start <= merged[-1][1]:
                merged[-1][1] = max(merged[-1][1], end)
            elif merged and start < merged[-1][1]:
                start = merged[-1][1]
                if start < end:
                    merged.append([start, end, kind])
            else:
                merged.append([start, end, kind])
        for start, end, kind in merged:
            layout = out.layout.add()
            layout.address_range.start_address = start
            layout.address_range.size = end - start
            layout.layout_type = kind


class _FullTables:
    """Instruction / mnemonic / operand / register tables of MODE_SELF_CONTAINED."""

    def __init__(self, out: Pb):
        self.out = out
        self.mnemonics: dict[str, int] = {}
        self.operand_strings: dict[str, int] = {}
        self.registers: dict[str, int] = {}
        self.operands: dict[tuple, int] = {}
        self.instructions: dict[tuple, int] = {}

    @staticmethod
    def _intern(table: dict[str, int], values: Any, value: str) -> int:
        if (idx := table.get(value)) is not None:
            return idx
        idx = len(values)
        values.append(value)
        table[value] = idx
        return idx

    def _operand(self, text: str, cs_op: Any, cs_insn: Any) -> int:
        op = Pb.Operand()
        op.operand_string_index = self._intern(self.operand_strings, self.out.operand_strings, text)
        access = getattr(cs_op, "access", 0) if cs_op is not None else 0
        op.access = (access & 3) or 1
        kind = getattr(cs_op, "type", None)
        # Capstone operand type values are shared across architectures:
        # 1 = REG, 2 = IMM, 3 = MEM (x86, ARM, ARM64, MIPS, PPC).
        if kind == 1:
            op.type = Pb.Operand.OPERAND_REGISTER
            name = cs_insn.reg_name(cs_op.reg) or text
            self._intern(self.registers, self.out.register_table, name)
            op.register_index = name  # register name, as the Ghidra exporter does
        elif kind == 2:
            op.type = Pb.Operand.OPERAND_IMMEDIATE
            value = int(cs_op.imm) & 0xFFFFFFFFFFFFFFFF
            op.value = value - (1 << 64) if value >= 1 << 63 else value
        elif kind == 3:
            op.type = Pb.Operand.OPERAND_MEMORY
            if (match := re.search(r"0x[0-9a-fA-F]+", text)) is not None:
                value = int(match.group(0), 16)
                op.address = value if value < 1 << 63 else value - (1 << 64)
        else:
            op.type = Pb.Operand.OPERAND_OTHER
            op.other = text
        key = op.SerializeToString(deterministic=True)
        if (idx := self.operands.get(key)) is not None:  # type: ignore[arg-type]
            return idx
        idx = len(self.out.operands)
        self.out.operands.append(op)
        self.operands[key] = idx  # type: ignore[index]
        return idx

    def instruction(self, insn: KunaInstruction, thumb: bool, flow: FlowAnalyzer) -> int:
        cs_insn = flow.decode(insn, thumb)
        texts = split_operands(insn.operands)
        cs_ops = list(cs_insn.operands) if cs_insn is not None else []
        if len(cs_ops) != len(texts):
            cs_ops = [None] * len(texts)
        operand_ids = tuple(self._operand(t, o, cs_insn) for t, o in zip(texts, cs_ops))
        mnem = self._intern(self.mnemonics, self.out.mnemonics, insn.mnemonic.lower())
        key = (insn.size, mnem, operand_ids, thumb)
        if (idx := self.instructions.get(key)) is not None:
            return idx
        idx = len(self.out.instructions)
        pi = self.out.instructions.add()
        pi.size = insn.size
        pi.mnemonic_index = mnem
        pi.operand_index.extend(operand_ids)
        pi.is_thumb = thumb
        self.instructions[key] = idx
        return idx


# ---------------------------------------------------------------------- #
# Public API
# ---------------------------------------------------------------------- #
def export_binary(
    exec_path: Path | str,
    output_file: Path | str | None = None,
    mode: int = Pb.ExporterMeta.MODE_LIGHT,
    decompiled: bool = False,
    kuna_path: Path | str | None = None,
    timeout: int | None = None,
    kuna_args: list[str] | None = None,
    jobs: str | int | None = None,
    compressed: bool = True,
) -> Path:
    """Export a binary to a ``.quokka`` file using kuna.

    Arguments:
        exec_path: Binary to analyze.
        output_file: Destination (default: ``<exec_path>.quokka``).
        mode: ``MODE_LIGHT`` (instructions re-decoded by capstone at load time)
            or ``MODE_SELF_CONTAINED`` (instructions stored in the file).
        decompiled: Decompile every function and store the C and prototypes.
        kuna_path: Explicit path to the kuna executable.
        timeout: Timeout, in seconds, of every kuna invocation.
        kuna_args: Extra arguments for kuna's analysis commands (e.g.
            ``["--mode", "fast"]``).
        jobs: kuna worker count when decompiling (``"auto"`` or an integer).
        compressed: XZ compress the output (what the other exporters do).

    Returns:
        The path of the written file.
    """
    exec_path = Path(exec_path)
    output = Path(output_file) if output_file else exec_path.with_name(exec_path.name + ".quokka")
    exporter = KunaExporter(exec_path, mode, decompiled, kuna_path, timeout, kuna_args, jobs)
    proto = exporter.run()
    data = proto.SerializeToString()
    if compressed:
        data = lzma.compress(data, format=lzma.FORMAT_XZ)
    output.write_bytes(data)
    return output


__all__ = ["KunaExporter", "export_binary", "is_noreturn_name", "split_operands"]
