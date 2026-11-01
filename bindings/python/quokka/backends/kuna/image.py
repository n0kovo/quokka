"""Executable image model for the kuna exporter.

kuna does not report the section table, so the program image (sections,
permissions, file offsets, architecture and data symbols) is read with LIEF.
Every address is a static virtual address, which is the address space kuna
works in (PE ImageBase + RVA, the ELF/Mach-O link-time VMA).
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
import hashlib
import logging
from dataclasses import dataclass, field
from pathlib import Path

from quokka.quokka_pb2 import Quokka as Pb  # pyright: ignore[reportMissingImports]

from .cli import KunaError

logger = logging.getLogger(__name__)

PERM_R, PERM_W, PERM_X = 4, 2, 1


@dataclass
class Section:
    """A mapped region of the image, exported as a quokka Segment."""

    name: str
    start: int
    size: int
    perms: int
    file_offset: int = -1  # -1 when the region has no bytes in the file
    file_size: int = 0
    kind: int = Pb.Segment.SEGMENT_UNK

    @property
    def end(self) -> int:
        return self.start + self.size

    @property
    def is_code(self) -> bool:
        return bool(self.perms & PERM_X)

    def contains(self, addr: int) -> bool:
        return self.start <= addr < self.end

    def file_offset_of(self, addr: int) -> int:
        delta = addr - self.start
        if self.file_offset < 0 or not 0 <= delta < self.file_size:
            return -1
        return self.file_offset + delta


@dataclass
class DataSymbol:
    address: int
    name: str
    size: int


@dataclass
class Image:
    """What the exporter needs to know about the executable file."""

    path: Path
    content: bytes
    format: str  # ELF | PE | MACHO
    isa: int  # Pb.Meta.ISA
    bits: int  # 32 | 64
    big_endian: bool
    sections: list[Section]
    entrypoint: int | None
    exported: set[int] = field(default_factory=set)
    data_symbols: list[DataSymbol] = field(default_factory=list)
    is_relocatable: bool = False
    #: ARM32 only: sorted (address, is_thumb) transitions from the ELF
    #: mapping symbols ($a/$t) and odd function symbols.
    arm_modes: list[tuple[int, bool]] = field(default_factory=list)
    #: Pointer slots the loader binds to imported symbols (ELF GOT entries
    #: with JUMP_SLOT/GLOB_DAT relocations, Mach-O bindings, PE IAT).
    import_slots: dict[int, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.sections.sort(key=lambda s: (s.start, s.size, s.name))
        self._starts = [s.start for s in self.sections]
        self.arm_modes.sort()
        self._mode_addrs = [a for a, _ in self.arm_modes]

    def is_thumb(self, addr: int) -> bool | None:
        """Thumb state at addr according to the image metadata (None: unknown)."""
        idx = bisect.bisect_right(self._mode_addrs, addr) - 1
        return self.arm_modes[idx][1] if idx >= 0 else None

    # ------------------------------------------------------------------ #
    def section_index(self, addr: int) -> int:
        """Index of the section containing addr, -1 if none."""
        idx = bisect.bisect_right(self._starts, addr) - 1
        # Sections may overlap (rare, e.g. TLS templates); look back a little.
        for i in range(idx, max(idx - 4, -1), -1):
            if self.sections[i].contains(addr):
                return i
        return -1

    def section_at(self, addr: int) -> Section | None:
        idx = self.section_index(addr)
        return self.sections[idx] if idx >= 0 else None

    def file_offset(self, addr: int) -> int:
        section = self.section_at(addr)
        return section.file_offset_of(addr) if section else -1

    def read(self, addr: int, size: int) -> bytes:
        """Read initialized bytes at a virtual address (b"" when unmapped)."""
        offset = self.file_offset(addr)
        if offset < 0:
            return b""
        section = self.section_at(addr)
        assert section is not None
        size = min(size, section.file_offset + section.file_size - offset)
        return self.content[offset : offset + size]

    def is_mapped(self, addr: int) -> bool:
        return self.section_index(addr) >= 0

    def is_code(self, addr: int) -> bool:
        section = self.section_at(addr)
        return section is not None and section.is_code

    @property
    def code_sections(self) -> list[Section]:
        return [s for s in self.sections if s.is_code and s.file_offset >= 0 and s.file_size > 0]

    def add_extern_section(self, addresses: list[int], name: str = "extern") -> Section | None:
        """Synthesize a SEGMENT_EXTERN region covering unmapped addresses.

        kuna reports undefined symbols (``kind: external``) at addresses the
        image does not map; quokka can only encode an address relative to a
        segment, so they get a pseudo segment of their own.
        """
        addresses = sorted(a for a in addresses if not self.is_mapped(a))
        if not addresses:
            return None
        start, end = addresses[0], addresses[-1] + (self.bits // 8)
        if any(s.start < end and start < s.end for s in self.sections):
            logger.warning("Cannot place an extern segment over mapped sections; skipping")
            return None
        section = Section(name, start, end - start, PERM_R, kind=Pb.Segment.SEGMENT_EXTERN)
        self.sections.append(section)
        self.__post_init__()
        return section

    # ------------------------------------------------------------------ #
    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.content).hexdigest()

    @property
    def address_size(self) -> int:
        return Pb.ADDR_64 if self.bits == 64 else Pb.ADDR_32


# ---------------------------------------------------------------------- #
# Loading
# ---------------------------------------------------------------------- #
def _lief():
    try:
        import lief  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise KunaError(
            "The kuna backend needs LIEF to read the section table: "
            "pip install 'quokka-project[kuna]' (or pip install lief)"
        ) from exc
    try:
        lief.logging.disable()
    except Exception:  # pragma: no cover
        pass
    return lief


def _segment_kind(perms: int, has_file_data: bool) -> int:
    if perms & PERM_X:
        return Pb.Segment.SEGMENT_CODE
    if not has_file_data:
        return Pb.Segment.SEGMENT_BSS
    return Pb.Segment.SEGMENT_DATA


def _elf_sections(lief, binary) -> list[Section]:
    elf = lief.ELF
    sections: list[Section] = []
    for sec in binary.sections:
        if not sec.has(elf.Section.FLAGS.ALLOC) or sec.size == 0:
            continue
        if sec.virtual_address == 0 and binary.header.file_type != elf.Header.FILE_TYPE.REL:
            continue
        perms = PERM_R
        if sec.has(elf.Section.FLAGS.WRITE):
            perms |= PERM_W
        if sec.has(elf.Section.FLAGS.EXECINSTR):
            perms |= PERM_X
        nobits = sec.type == elf.Section.TYPE.NOBITS
        sections.append(
            Section(
                sec.name,
                sec.virtual_address,
                sec.size,
                perms,
                -1 if nobits else sec.offset,
                0 if nobits else sec.size,
                _segment_kind(perms, not nobits),
            )
        )
    if sections:
        return sections

    # No section header table (stripped/packed): fall back to PT_LOAD.
    for seg in binary.segments:
        if seg.type != elf.Segment.TYPE.LOAD or seg.virtual_size == 0:
            continue
        perms = 0
        if seg.has(elf.Segment.FLAGS.R):
            perms |= PERM_R
        if seg.has(elf.Segment.FLAGS.W):
            perms |= PERM_W
        if seg.has(elf.Segment.FLAGS.X):
            perms |= PERM_X
        sections.append(
            Section(
                f"LOAD_{seg.virtual_address:x}",
                seg.virtual_address,
                seg.virtual_size,
                perms or PERM_R,
                seg.file_offset if seg.physical_size else -1,
                seg.physical_size,
                _segment_kind(perms, seg.physical_size > 0),
            )
        )
    return sections


def _macho_sections(lief, binary) -> list[Section]:
    macho = lief.MachO
    zerofill = {
        macho.Section.TYPE.ZEROFILL,
        macho.Section.TYPE.GB_ZEROFILL,
        macho.Section.TYPE.THREAD_LOCAL_ZEROFILL,
    }
    sections: list[Section] = []
    for sec in binary.sections:
        if sec.size == 0:
            continue
        segment = sec.segment
        prot = segment.init_protection if segment is not None else 1
        perms = 0
        if prot & 1:
            perms |= PERM_R
        if prot & 2:
            perms |= PERM_W
        if prot & 4:
            perms |= PERM_X
        if sec.has(macho.Section.FLAGS.PURE_INSTRUCTIONS) or sec.has(
            macho.Section.FLAGS.SOME_INSTRUCTIONS
        ):
            perms |= PERM_X | PERM_R
        elif perms & PERM_X and segment is not None and segment.name == "__TEXT":
            # __TEXT is r-x as a whole but only instruction sections hold code.
            perms &= ~PERM_X
        nobits = sec.type in zerofill
        seg_name = segment.name if segment is not None else ""
        sections.append(
            Section(
                f"{seg_name},{sec.name}" if seg_name else sec.name,
                sec.virtual_address,
                sec.size,
                perms or PERM_R,
                -1 if nobits else sec.offset,
                0 if nobits else sec.size,
                _segment_kind(perms, not nobits),
            )
        )
    return sections


def _pe_sections(lief, binary) -> list[Section]:
    pe = lief.PE
    chars = pe.Section.CHARACTERISTICS
    base = binary.optional_header.imagebase
    sections: list[Section] = []
    for sec in binary.sections:
        vsize = sec.virtual_size or sec.sizeof_raw_data
        if vsize == 0:
            continue
        perms = 0
        if sec.has_characteristic(chars.MEM_READ):
            perms |= PERM_R
        if sec.has_characteristic(chars.MEM_WRITE):
            perms |= PERM_W
        if sec.has_characteristic(chars.MEM_EXECUTE) or sec.has_characteristic(chars.CNT_CODE):
            perms |= PERM_X
        raw = min(sec.sizeof_raw_data, vsize)
        sections.append(
            Section(
                sec.name,
                base + sec.virtual_address,
                vsize,
                perms or PERM_R,
                sec.pointerto_raw_data if raw else -1,
                raw,
                _segment_kind(perms, raw > 0),
            )
        )
    return sections


def _isa(binary) -> int:
    arch = binary.header.architecture if hasattr(binary.header, "architecture") else None
    name = str(arch).upper() if arch is not None else ""
    if not name:
        name = str(getattr(binary.abstract.header, "architecture", "")).upper()
    if any(x in name for x in ("X86", "I386", "AMD64", "X64")):
        return Pb.Meta.PROC_INTEL
    if any(x in name for x in ("ARM", "AARCH64", "ARM64", "THUMB")):
        return Pb.Meta.PROC_ARM
    if "MIPS" in name:
        return Pb.Meta.PROC_MIPS
    if "PPC" in name or "POWERPC" in name:
        return Pb.Meta.PROC_PPC
    return Pb.Meta.PROC_UNK


def _import_slots(lief, binary, fmt: str, image: Image) -> dict[int, str]:
    slots: dict[int, str] = {}
    if fmt == "ELF":
        for reloc in binary.relocations:
            if not reloc.has_symbol:
                continue
            sym = reloc.symbol
            if not sym.name or not sym.imported:
                continue
            if image.is_mapped(reloc.address) and not image.is_code(reloc.address):
                slots.setdefault(reloc.address, sym.name)
    elif fmt == "MACHO":
        for binding in binary.bindings:
            sym = binding.symbol
            if sym is None or not sym.name:
                continue
            if image.is_mapped(binding.address):
                slots.setdefault(binding.address, sym.name)
    elif fmt == "PE":
        base = binary.optional_header.imagebase
        for imp in binary.imports:
            for entry in imp.entries:
                addr = entry.iat_address
                if addr < base:
                    addr += base
                name = entry.name if not entry.is_ordinal else f"{imp.name}:{entry.ordinal}"
                if name and image.is_mapped(addr):
                    slots.setdefault(addr, name)
    return slots


def load_image(path: Path | str) -> Image:
    """Parse the executable with LIEF and build the image model."""
    lief = _lief()
    path = Path(path)
    content = path.read_bytes()
    binary = lief.parse(str(path))
    if binary is None:
        raise KunaError(f"LIEF could not parse {path}")
    if isinstance(binary, lief.MachO.FatBinary):  # pragma: no cover - lief.parse thins
        binary = binary.at(0)

    abstract = binary.abstract if hasattr(binary, "abstract") else binary
    header = abstract.header
    bits = 64 if header.is_64 else 32
    big_endian = str(header.endianness).upper().endswith("BIG")
    isa = _isa(abstract)
    is_relocatable = False

    if isinstance(binary, lief.ELF.Binary):
        fmt = "ELF"
        sections = _elf_sections(lief, binary)
        is_relocatable = binary.header.file_type == lief.ELF.Header.FILE_TYPE.REL
        entry = binary.entrypoint or None
    elif isinstance(binary, lief.MachO.Binary):
        fmt = "MACHO"
        sections = _macho_sections(lief, binary)
        is_relocatable = binary.header.file_type == lief.MachO.Header.FILE_TYPE.OBJECT
        entry = binary.entrypoint or None
    elif isinstance(binary, lief.PE.Binary):
        fmt = "PE"
        sections = _pe_sections(lief, binary)
        entry = binary.entrypoint or None
    else:
        raise KunaError(f"Unsupported executable format for {path}")

    image = Image(
        path=path,
        content=content,
        format=fmt,
        isa=isa,
        bits=bits,
        big_endian=big_endian,
        sections=sections,
        entrypoint=entry,
        is_relocatable=is_relocatable,
    )

    # Exported functions (Function.is_exported)
    try:
        imagebase = binary.optional_header.imagebase if fmt == "PE" else 0
        for func in abstract.exported_functions:
            addr = func.address
            if fmt == "PE" and addr < imagebase:
                addr += imagebase
            image.exported.add(addr)
    except Exception as exc:  # pragma: no cover - LIEF quirks
        logger.debug("Could not list exported functions: %s", exc)

    try:
        symbols = list(abstract.symbols)
    except Exception:  # pragma: no cover
        symbols = []

    try:
        image.import_slots = _import_slots(lief, binary, fmt, image)
    except Exception as exc:  # pragma: no cover - LIEF quirks
        logger.debug("Could not list import slots: %s", exc)

    # ARM/Thumb interworking state
    if fmt == "ELF" and isa == Pb.Meta.PROC_ARM and bits == 32:
        modes: dict[int, bool] = {}
        for sym in symbols:
            name, value = sym.name, int(sym.value or 0)
            if name in ("$t", "$a") or name.startswith(("$t.", "$a.")):
                modes[value] = name.startswith("$t")
            elif name in ("$d",) or name.startswith("$d."):
                continue
            elif value & 1 and image.is_code(value & ~1):
                modes.setdefault(value & ~1, True)
        image.arm_modes = sorted(modes.items())
        image.__post_init__()

    # Named data (symbols that land in a mapped, non executable section)
    seen: set[int] = set()
    for sym in symbols:
        addr = int(getattr(sym, "value", 0) or 0)
        name = sym.name
        if not addr or not sym.name or addr in seen:
            continue
        section = image.section_at(addr)
        if section is None or section.is_code or addr in image.import_slots:
            continue
        seen.add(addr)
        image.data_symbols.append(DataSymbol(addr, name, int(getattr(sym, "size", 0) or 0)))

    return image


__all__ = ["DataSymbol", "Image", "Section", "load_image"]
