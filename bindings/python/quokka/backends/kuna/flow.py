"""Control flow and reference semantics of kuna instructions.

kuna's ``disassemble`` command gives instruction boundaries, bytes and text,
but no flow information (it does not export basic blocks). This module
classifies each kuna instruction (jump, conditional jump, call, return, ...)
and resolves the addresses it references, so that the exporter can rebuild
basic blocks, CFG edges and cross references.

The classification decodes the bytes kuna reported, at the address kuna
reported, with capstone: capstone is already the decoder quokka uses to
materialize instructions, so the CFG stays consistent with what quokka users
see when they iterate over instructions. Instruction boundaries always come
from kuna.

A tiny per-block constant tracker resolves the multi-instruction address
materializations used by RISC targets (``adrp``/``add``/``ldr`` on AArch64,
PC-relative literal loads on ARM, ``lui``/``addiu`` on MIPS, ``lis``/``addi``
on PowerPC) so their data references are not lost.
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

import enum
import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import capstone

from quokka.quokka_pb2 import Quokka as Pb  # pyright: ignore[reportMissingImports]

if TYPE_CHECKING:
    from .cli import KunaInstruction
    from .image import Image

logger = logging.getLogger(__name__)

ACCESS_READ = 1
ACCESS_WRITE = 2


class Flow(enum.IntEnum):
    """How an instruction transfers control."""

    NORMAL = 0  # falls through
    JUMP = 1  # unconditional direct or indirect jump
    CJUMP = 2  # conditional jump: target + fall-through
    CALL = 3  # call, returns to the fall-through
    RET = 4  # return
    CRET = 5  # conditional return (falls through otherwise)
    TRAP = 6  # never falls through (hlt, ud2, brk, ...)


@dataclass
class InsnInfo:
    """Semantics of one instruction."""

    flow: Flow = Flow.NORMAL
    #: Direct target of a jump / call, if statically known.
    target: int | None = None
    #: True for computed jumps/calls (no static target).
    indirect: bool = False
    #: Fixed memory slot read by an indirect jump/call (``jmp [rip+x]``,
    #: an AArch64 stub ``ldr x16,[x16,#off]; br x16``...).
    slot: int | None = None
    #: Number of delay slot instructions following this branch.
    delay_slots: int = 0
    #: Resolved memory accesses: (address, access mask, size in bytes or 0).
    mem_refs: list[tuple[int, int, int]] = field(default_factory=list)
    #: Address-valued operands that are not dereferenced (lea, adr, imm).
    addr_refs: list[int] = field(default_factory=list)

    @property
    def ends_block(self) -> bool:
        return self.flow in (Flow.JUMP, Flow.CJUMP, Flow.RET, Flow.CRET, Flow.TRAP)

    @property
    def falls_through(self) -> bool:
        return self.flow in (Flow.NORMAL, Flow.CJUMP, Flow.CALL, Flow.CRET)


# capstone 5 calls it ARM64, capstone 6 AARCH64.
try:  # pragma: no cover - import shape depends on capstone version
    from capstone import arm64 as _a64  # type: ignore[attr-defined]

    _A64_PREFIX = "ARM64_"
except ImportError:  # pragma: no cover
    from capstone import aarch64 as _a64  # type: ignore[attr-defined,no-redef]

    _A64_PREFIX = "AARCH64_"
from capstone import arm as _arm
from capstone import mips as _mips
from capstone import ppc as _ppc
from capstone import x86 as _x86


def _a64c(name: str) -> int:
    return getattr(_a64, _A64_PREFIX + name, -1)


A64_OP_REG, A64_OP_IMM, A64_OP_MEM = _a64c("OP_REG"), _a64c("OP_IMM"), _a64c("OP_MEM")
A64_CC_INVALID, A64_CC_AL, A64_CC_NV = _a64c("CC_INVALID"), _a64c("CC_AL"), _a64c("CC_NV")

GRP_JUMP = capstone.CS_GRP_JUMP
GRP_CALL = capstone.CS_GRP_CALL
GRP_RET = capstone.CS_GRP_RET
GRP_IRET = capstone.CS_GRP_IRET
GRP_INT = capstone.CS_GRP_INT


class FlowAnalyzer:
    """Classify instructions of one program.

    Call :meth:`reset` at every basic block boundary candidate (function
    entry, branch target) to drop the constant tracker state; the exporter
    analyzes instructions in address order inside each function.
    """

    def __init__(self, image: Image):
        from quokka.analysis import ArchARMThumb
        from quokka.backends.capstone import get_capstone_context
        from quokka.types import Endianness
        from quokka.utils import get_arch, get_isa

        self.image = image
        self.isa = image.isa
        self.bits = image.bits
        endianness = Endianness.BIG_ENDIAN if image.big_endian else Endianness.LITTLE_ENDIAN
        self.cs: capstone.Cs | None = None
        self.cs_thumb: capstone.Cs | None = None
        try:
            arch = get_arch(get_isa(image.isa), image.bits)
            self.cs = get_capstone_context(arch, endianness)
            if self.isa == Pb.Meta.PROC_ARM and self.bits == 32:
                self.cs_thumb = get_capstone_context(ArchARMThumb, endianness)
        except Exception as exc:  # unsupported architecture
            logger.warning("No capstone context for this architecture (%s); "
                           "falling back to mnemonic heuristics", exc)
        self.regs: dict[int, tuple[str, int]] = {}  # reg -> ("const"|"load", value)
        #: Decoded instructions, kept between the CFG and the reference passes
        #: of a function (see :meth:`clear_cache`).
        self._decoded: dict[tuple[int, bool], Any] = {}

    # ------------------------------------------------------------------ #
    def reset(self) -> None:
        self.regs.clear()

    def clear_cache(self) -> None:
        self._decoded.clear()

    def decode(self, insn: KunaInstruction, thumb: bool = False) -> Any | None:
        key = (insn.address, thumb)
        if key in self._decoded:
            return self._decoded[key]
        ctx = self.cs_thumb if (thumb and self.cs_thumb is not None) else self.cs
        if ctx is None or not insn.raw:
            return None
        try:
            cs_insn = next(ctx.disasm(insn.raw, insn.address, 1), None)
        except capstone.CsError:
            cs_insn = None
        self._decoded[key] = cs_insn
        return cs_insn

    def analyze(self, insn: KunaInstruction, thumb: bool = False) -> InsnInfo:
        """Return the semantics of ``insn`` and update the tracker."""
        if insn.is_data:
            self.reset()
            return InsnInfo()
        cs_insn = self.decode(insn, thumb)
        if cs_insn is None:
            self.reset()
            return self._fallback(insn)
        if cs_insn.size != insn.size:
            logger.debug(
                "capstone/kuna size mismatch at %#x (%d vs %d)",
                insn.address, cs_insn.size, insn.size,
            )
        try:
            if self.isa == Pb.Meta.PROC_INTEL:
                return self._x86(cs_insn)
            if self.isa == Pb.Meta.PROC_ARM and self.bits == 64:
                return self._arm64(cs_insn)
            if self.isa == Pb.Meta.PROC_ARM:
                return self._arm(cs_insn, thumb)
            if self.isa == Pb.Meta.PROC_MIPS:
                return self._mips(cs_insn)
            if self.isa == Pb.Meta.PROC_PPC:
                return self._ppc(cs_insn)
        except capstone.CsError as exc:  # detail not available, etc.
            logger.debug("capstone detail error at %#x: %s", insn.address, exc)
        self.reset()
        return self._generic(cs_insn)

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    def _plausible_address(self, value: int) -> bool:
        """Is an immediate likely to be an address in this image?"""
        if value < 0x1000:
            return False
        if self.bits == 64 and value >= 1 << 63:
            return False
        return self.image.is_mapped(value)

    def _read_pointer(self, addr: int, size: int | None = None) -> int | None:
        size = size or self.bits // 8
        raw = self.image.read(addr, size)
        if len(raw) != size:
            return None
        return int.from_bytes(raw, "big" if self.image.big_endian else "little")

    def _kill_first_reg(self, cs_insn: Any, reg_type: int) -> None:
        ops = cs_insn.operands
        if ops and ops[0].type == reg_type:
            self.regs.pop(ops[0].reg, None)

    @staticmethod
    def _generic(cs_insn: Any) -> InsnInfo:
        groups = set(cs_insn.groups)
        if groups & {GRP_RET, GRP_IRET}:
            return InsnInfo(Flow.RET)
        if GRP_CALL in groups:
            return InsnInfo(Flow.CALL, indirect=True)
        if GRP_JUMP in groups:
            return InsnInfo(Flow.JUMP, indirect=True)
        return InsnInfo()

    def _fallback(self, insn: KunaInstruction) -> InsnInfo:
        """Last resort: classify from kuna's mnemonic text."""
        m = insn.mnemonic.lower()
        target = None
        op = insn.operands.strip()
        if op.startswith(("0x", "#0x")):
            try:
                target = int(op.lstrip("#").split()[0].rstrip(","), 16)
            except ValueError:
                target = None
        if m in ("ret", "retn", "retf", "iret", "iretd", "iretq", "eret") or m.startswith("ret"):
            return InsnInfo(Flow.RET)
        if m in ("hlt", "ud2", "brk", "udf", "bkpt"):
            return InsnInfo(Flow.TRAP)
        if m in ("call", "bl", "blx", "jal", "bal", "jalr", "blr", "bctrl"):
            return InsnInfo(Flow.CALL, target=target, indirect=target is None)
        if m in ("jmp", "b", "j", "br", "bx", "jr", "bctr"):
            return InsnInfo(Flow.JUMP, target=target, indirect=target is None)
        if target is not None and (m.startswith("j") or m.startswith("b.") or m.startswith("cb")
                                   or m.startswith("tb")):
            return InsnInfo(Flow.CJUMP, target=target)
        return InsnInfo()

    # ------------------------------------------------------------------ #
    # x86 / x86-64
    # ------------------------------------------------------------------ #
    def _x86_mem_addr(self, cs_insn: Any, op: Any) -> int | None:
        mem = op.mem
        if mem.segment not in (0, _x86.X86_REG_DS, _x86.X86_REG_CS, _x86.X86_REG_SS,
                               _x86.X86_REG_ES):
            return None  # fs:/gs: (TLS)
        if mem.index != 0:
            return None
        if mem.base in (_x86.X86_REG_RIP, _x86.X86_REG_EIP):
            return (cs_insn.address + cs_insn.size + mem.disp) & ((1 << self.bits) - 1)
        if mem.base == 0:
            return mem.disp & ((1 << self.bits) - 1)
        return None

    def _x86(self, cs_insn: Any) -> InsnInfo:
        groups = set(cs_insn.groups)
        mnem = cs_insn.mnemonic.split()[-1]  # drop prefixes (bnd jmp, notrack jmp)
        ops = cs_insn.operands

        if groups & {GRP_RET, GRP_IRET}:
            return InsnInfo(Flow.RET)
        if mnem in ("hlt", "ud2", "ud1", "ud0", "int3"):
            return InsnInfo(Flow.TRAP)

        if GRP_CALL in groups or GRP_JUMP in groups:
            is_call = GRP_CALL in groups
            info = InsnInfo(Flow.CALL if is_call else (Flow.JUMP if mnem in ("jmp", "ljmp")
                                                        else Flow.CJUMP))
            if ops and ops[0].type == _x86.X86_OP_IMM:
                info.target = ops[0].imm & ((1 << self.bits) - 1)
            elif ops and ops[0].type == _x86.X86_OP_MEM:
                info.indirect = True
                slot = self._x86_mem_addr(cs_insn, ops[0])
                if slot is not None:
                    info.slot = slot
                    info.mem_refs.append((slot, ACCESS_READ, ops[0].size))
            else:
                info.indirect = True
            return info

        info = InsnInfo()
        is_lea = mnem == "lea"
        for op in ops:
            if op.type == _x86.X86_OP_MEM:
                addr = self._x86_mem_addr(cs_insn, op)
                if addr is None or not self.image.is_mapped(addr):
                    continue
                if is_lea:
                    info.addr_refs.append(addr)
                else:
                    access = op.access & (ACCESS_READ | ACCESS_WRITE) or ACCESS_READ
                    info.mem_refs.append((addr, access, op.size))
            elif op.type == _x86.X86_OP_IMM and op.size >= 4:
                value = op.imm & ((1 << self.bits) - 1)
                if self._plausible_address(value):
                    info.addr_refs.append(value)
        return info

    # ------------------------------------------------------------------ #
    # AArch64
    # ------------------------------------------------------------------ #
    def _arm64(self, cs_insn: Any) -> InsnInfo:
        groups = set(cs_insn.groups)
        mnem = cs_insn.mnemonic
        ops = cs_insn.operands

        if GRP_RET in groups or mnem in ("ret", "retaa", "retab", "eret", "eretaa", "eretab"):
            self.reset()
            return InsnInfo(Flow.RET)
        if mnem in ("brk", "udf", "hlt"):
            self.reset()
            return InsnInfo(Flow.TRAP)

        if GRP_CALL in groups or GRP_JUMP in groups:
            is_call = GRP_CALL in groups
            imm_ops = [op.imm for op in ops if op.type == A64_OP_IMM]
            reg_ops = [op.reg for op in ops if op.type == A64_OP_REG]
            conditional = (
                mnem.startswith(("cbz", "cbnz", "tbz", "tbnz"))
                or cs_insn.cc not in (A64_CC_INVALID, A64_CC_AL, A64_CC_NV)
            )
            if is_call:
                info = InsnInfo(Flow.CALL)
            else:
                info = InsnInfo(Flow.CJUMP if conditional else Flow.JUMP)
            if mnem in ("b", "bl") or mnem.startswith(("b.", "cb", "tb", "bc.")):
                info.target = imm_ops[-1] if imm_ops else None
                info.indirect = info.target is None
            else:  # br, blr, braa, blraa, ...
                info.indirect = True
                if reg_ops and (state := self.regs.get(reg_ops[0])) and state[0] == "load":
                    info.slot = state[1]
            if is_call:
                # x0-x18 and x30 are clobbered by a call.
                self.reset()
            return info

        info = InsnInfo()
        if mnem in ("adrp", "adr") and len(ops) == 2 and ops[1].type == A64_OP_IMM:
            self.regs[ops[0].reg] = ("const", ops[1].imm)
            if mnem == "adr" and self.image.is_mapped(ops[1].imm):
                info.addr_refs.append(ops[1].imm)
            return info
        if (
            mnem == "add"
            and len(ops) == 3
            and ops[1].type == A64_OP_REG
            and ops[2].type == A64_OP_IMM
            and (state := self.regs.get(ops[1].reg))
            and state[0] == "const"
        ):
            addr = state[1] + ops[2].imm
            self.regs[ops[0].reg] = ("const", addr)
            if self.image.is_mapped(addr):
                info.addr_refs.append(addr)
            return info
        if mnem.startswith(("ldr", "ldur", "str", "stur", "ldp", "stp", "ldrs")):
            is_store = mnem.startswith(("st",))
            for op in ops:
                if op.type == A64_OP_MEM and op.mem.index == 0:
                    if (state := self.regs.get(op.mem.base)) and state[0] == "const":
                        addr = state[1] + op.mem.disp
                        if self.image.is_mapped(addr):
                            info.mem_refs.append(
                                (addr, ACCESS_WRITE if is_store else ACCESS_READ, 0)
                            )
                            if not is_store and ops[0].type == A64_OP_REG:
                                self.regs[ops[0].reg] = ("load", addr)
                                if cs_insn.writeback:
                                    self.regs.pop(op.mem.base, None)
                                return info
                elif op.type == A64_OP_IMM and mnem.startswith("ldr") and not is_store:
                    # ldr xN, <literal>
                    if self.image.is_mapped(op.imm):
                        info.mem_refs.append((op.imm, ACCESS_READ, 0))
                        if ops[0].type == A64_OP_REG:
                            self.regs[ops[0].reg] = ("load", op.imm)
                            return info
            if not is_store:
                self._kill_first_reg(cs_insn, A64_OP_REG)
            return info
        if mnem == "mov" and len(ops) == 2 and ops[1].type == A64_OP_REG:
            if state := self.regs.get(ops[1].reg):
                self.regs[ops[0].reg] = state
                return info
        self._kill_first_reg(cs_insn, A64_OP_REG)
        return info

    # ------------------------------------------------------------------ #
    # ARM (A32 / T32)
    # ------------------------------------------------------------------ #
    def _arm(self, cs_insn: Any, thumb: bool) -> InsnInfo:
        groups = set(cs_insn.groups)
        # insn_name() is the bare instruction ("b" for "bne.w", "pop" for
        # "popne"); the condition is in cs_insn.cc.
        mnem = cs_insn.insn_name() or cs_insn.mnemonic.split(".")[0]
        ops = cs_insn.operands
        conditional = cs_insn.cc not in (_arm.ARM_CC_AL, _arm.ARM_CC_INVALID)
        pc, lr = _arm.ARM_REG_PC, _arm.ARM_REG_LR
        try:
            _, regs_write = cs_insn.regs_access()
        except capstone.CsError:
            regs_write = []
        writes_pc = pc in regs_write or (ops and ops[0].type == _arm.ARM_OP_REG
                                        and ops[0].reg == pc and not mnem.startswith(("cmp", "tst", "str")))
        reg_list_pc = mnem.startswith(("pop", "ldm")) and any(
            op.type == _arm.ARM_OP_REG and op.reg == pc for op in ops
        )

        if mnem in ("udf", "bkpt") or (mnem == "trap"):
            self.reset()
            return InsnInfo(Flow.TRAP)

        # Returns: bx lr, pop {...,pc}, ldm sp!, {...,pc}, mov pc, lr
        if (
            GRP_RET in groups
            or (mnem.startswith("bx") and ops and ops[0].type == _arm.ARM_OP_REG and ops[0].reg == lr)
            or reg_list_pc
            or (mnem == "mov" and writes_pc and len(ops) == 2 and ops[1].type == _arm.ARM_OP_REG
                and ops[1].reg == lr)
        ):
            self.reset()
            return InsnInfo(Flow.CRET if conditional else Flow.RET)

        if GRP_CALL in groups or mnem in ("bl", "blx"):
            info = InsnInfo(Flow.CALL)
            if ops and ops[0].type == _arm.ARM_OP_IMM:
                info.target = ops[0].imm
            else:
                info.indirect = True
                if ops and ops[0].type == _arm.ARM_OP_REG and (st := self.regs.get(ops[0].reg)) \
                        and st[0] == "load":
                    info.slot = st[1]
            self.reset()
            return info

        if GRP_JUMP in groups or writes_pc or mnem in ("cbz", "cbnz"):
            cond = conditional or mnem in ("cbz", "cbnz")
            info = InsnInfo(Flow.CJUMP if cond else Flow.JUMP)
            imm = [op.imm for op in ops if op.type == _arm.ARM_OP_IMM]
            if mnem in ("b", "cbz", "cbnz") and imm:
                info.target = imm[-1]
            else:
                info.indirect = True
                for op in ops:
                    if op.type == _arm.ARM_OP_MEM and op.mem.base == pc and op.mem.index == 0:
                        info.slot = self._arm_pc_rel(cs_insn, op, thumb)
                    elif op.type == _arm.ARM_OP_REG and (st := self.regs.get(op.reg)) \
                            and st[0] == "load":
                        info.slot = st[1]
            return info

        info = InsnInfo()
        if mnem.startswith(("ldr", "str", "vldr", "vstr")):
            is_store = mnem.startswith(("str", "vstr"))
            for op in ops:
                if op.type != _arm.ARM_OP_MEM or op.mem.index != 0:
                    continue
                addr: int | None = None
                if op.mem.base == pc:
                    addr = self._arm_pc_rel(cs_insn, op, thumb)
                elif (st := self.regs.get(op.mem.base)) and st[0] == "const":
                    addr = st[1] + op.mem.disp
                if addr is None or not self.image.is_mapped(addr):
                    continue
                info.mem_refs.append((addr, ACCESS_WRITE if is_store else ACCESS_READ, 0))
                if not is_store and ops[0].type == _arm.ARM_OP_REG:
                    value = self._read_pointer(addr, 4) if op.mem.base == pc else None
                    if value is not None:
                        # Literal pool word: an address, or a PIC offset that
                        # a following ``add rX, pc`` turns into one.
                        if self._plausible_address(value):
                            info.addr_refs.append(value)
                        self.regs[ops[0].reg] = ("const", value)
                    else:
                        self.regs[ops[0].reg] = ("load", addr)
                    return info
            if not is_store:
                self._kill_first_reg(cs_insn, _arm.ARM_OP_REG)
            return info
        if mnem in ("adr",) and len(ops) == 2 and ops[1].type == _arm.ARM_OP_IMM:
            if self.image.is_mapped(ops[1].imm):
                info.addr_refs.append(ops[1].imm)
                self.regs[ops[0].reg] = ("const", ops[1].imm)
            return info
        if (
            mnem == "add" and len(ops) in (2, 3)
            and all(op.type == _arm.ARM_OP_REG for op in ops)
            and pc in [op.reg for op in ops[1:]] + ([ops[0].reg] if len(ops) == 2 else [])
        ):
            # add rX, pc, rX / add rX, pc (PIC access): rX holds an offset.
            srcs = [op.reg for op in ops] if len(ops) == 2 else [ops[1].reg, ops[2].reg]
            other = srcs[1] if srcs[0] == pc else srcs[0]
            st = self.regs.get(other)
            if st and st[0] == "const":
                base = cs_insn.address + (4 if thumb else 8)
                addr = (base + st[1]) & 0xFFFFFFFF
                if self.image.is_mapped(addr):
                    info.addr_refs.append(addr)
                    self.regs[ops[0].reg] = ("const", addr)
                    return info
        if mnem in ("movw", "movt") and len(ops) == 2 and ops[1].type == _arm.ARM_OP_IMM:
            if mnem == "movw":
                self.regs[ops[0].reg] = ("const", ops[1].imm & 0xFFFF)
            elif (st := self.regs.get(ops[0].reg)) and st[0] == "const":
                addr = (st[1] & 0xFFFF) | ((ops[1].imm & 0xFFFF) << 16)
                self.regs[ops[0].reg] = ("const", addr)
                if self._plausible_address(addr):
                    info.addr_refs.append(addr)
            return info
        self._kill_first_reg(cs_insn, _arm.ARM_OP_REG)
        return info

    @staticmethod
    def _arm_pc_rel(cs_insn: Any, op: Any, thumb: bool) -> int:
        base = (cs_insn.address + (4 if thumb else 8)) & ~3
        disp = op.mem.disp
        if getattr(op, "subtracted", False) and disp > 0:
            disp = -disp
        return (base + disp) & 0xFFFFFFFF

    # ------------------------------------------------------------------ #
    # MIPS
    # ------------------------------------------------------------------ #
    def _mips(self, cs_insn: Any) -> InsnInfo:
        groups = set(cs_insn.groups)
        mnem = cs_insn.mnemonic
        ops = cs_insn.operands
        # Compact (R6) branches have no delay slot.
        delay = 0 if mnem.endswith("c") and mnem not in ("bc1f", "bc1t") else 1
        if mnem.startswith(("bc", "balc")) and mnem not in ("bc1f", "bc1t", "bc1fl", "bc1tl"):
            delay = 0

        if mnem in ("break", "sdbbp"):
            self.reset()
            return InsnInfo(Flow.TRAP)
        if mnem in ("jr", "jr.hb", "jrc") and ops and ops[0].reg == _mips.MIPS_REG_RA:
            self.reset()
            return InsnInfo(Flow.RET, delay_slots=0 if mnem == "jrc" else 1)
        if GRP_RET in groups:
            self.reset()
            return InsnInfo(Flow.RET, delay_slots=delay)
        if GRP_CALL in groups or mnem in ("jal", "jalr", "bal", "bgezal", "bltzal", "jalx",
                                           "jalrc", "balc"):
            info = InsnInfo(Flow.CALL, delay_slots=delay)
            imm = [op.imm for op in ops if op.type == _mips.MIPS_OP_IMM]
            if mnem in ("jal", "bal", "balc", "jalx") and imm:
                info.target = imm[-1]
            else:
                info.indirect = True
            self.reset()
            return info
        if GRP_JUMP in groups:
            imm = [op.imm for op in ops if op.type == _mips.MIPS_OP_IMM]
            uncond = mnem in ("j", "b", "jr", "jr.hb", "jrc", "bc")
            info = InsnInfo(Flow.JUMP if uncond else Flow.CJUMP, delay_slots=delay)
            if mnem in ("jr", "jr.hb", "jrc"):
                info.indirect = True
            elif imm:
                info.target = imm[-1]
            else:
                info.indirect = True
            return info

        info = InsnInfo()
        if mnem == "lui" and len(ops) == 2 and ops[1].type == _mips.MIPS_OP_IMM:
            self.regs[ops[0].reg] = ("const", (ops[1].imm & 0xFFFF) << 16)
            return info
        if mnem in ("addiu", "daddiu", "ori") and len(ops) == 3 and ops[2].type == _mips.MIPS_OP_IMM:
            if (st := self.regs.get(ops[1].reg)) and st[0] == "const":
                addr = (st[1] | (ops[2].imm & 0xFFFF)) if mnem == "ori" else st[1] + ops[2].imm
                self.regs[ops[0].reg] = ("const", addr)
                if self._plausible_address(addr):
                    info.addr_refs.append(addr)
                return info
        for op in ops:
            if op.type == _mips.MIPS_OP_MEM and (st := self.regs.get(op.mem.base)) and st[0] == "const":
                addr = st[1] + op.mem.disp
                if self.image.is_mapped(addr):
                    is_store = mnem.startswith("s")
                    info.mem_refs.append((addr, ACCESS_WRITE if is_store else ACCESS_READ, 0))
        self._kill_first_reg(cs_insn, _mips.MIPS_OP_REG)
        return info

    # ------------------------------------------------------------------ #
    # PowerPC
    # ------------------------------------------------------------------ #
    def _ppc(self, cs_insn: Any) -> InsnInfo:
        groups = set(cs_insn.groups)
        mnem = cs_insn.mnemonic.rstrip("+-")
        ops = cs_insn.operands
        imm = [op.imm for op in ops if op.type == _ppc.PPC_OP_IMM]

        if mnem in ("trap",):
            self.reset()
            return InsnInfo(Flow.TRAP)
        if mnem == "blr":
            self.reset()
            return InsnInfo(Flow.RET)
        if mnem.endswith("lr") and mnem.startswith("b") and not mnem.endswith("lrl"):
            return InsnInfo(Flow.CRET)  # beqlr, bnelr, bdnzlr...
        if GRP_CALL in groups or mnem in ("bl", "bla", "bctrl", "blrl"):
            info = InsnInfo(Flow.CALL)
            if mnem in ("bl", "bla") and imm:
                info.target = imm[-1]
            else:
                info.indirect = True
            self.reset()
            return info
        if mnem in ("bctr",):
            return InsnInfo(Flow.JUMP, indirect=True)
        if GRP_JUMP in groups or (mnem.startswith("b") and imm):
            uncond = mnem in ("b", "ba")
            info = InsnInfo(Flow.JUMP if uncond else Flow.CJUMP)
            if imm:
                info.target = imm[-1]
            else:
                info.indirect = True
            return info

        info = InsnInfo()
        if mnem in ("lis",) and len(ops) == 2 and ops[1].type == _ppc.PPC_OP_IMM:
            self.regs[ops[0].reg] = ("const", (ops[1].imm & 0xFFFF) << 16)
            return info
        if mnem in ("addi", "ori") and len(ops) == 3 and ops[2].type == _ppc.PPC_OP_IMM:
            if (st := self.regs.get(ops[1].reg)) and st[0] == "const":
                addr = (st[1] | (ops[2].imm & 0xFFFF)) if mnem == "ori" else st[1] + ops[2].imm
                self.regs[ops[0].reg] = ("const", addr)
                if self._plausible_address(addr):
                    info.addr_refs.append(addr)
                return info
        for op in ops:
            if op.type == _ppc.PPC_OP_MEM and (st := self.regs.get(op.mem.base)) and st[0] == "const":
                addr = st[1] + op.mem.disp
                if self.image.is_mapped(addr):
                    is_store = mnem.startswith("st")
                    info.mem_refs.append((addr, ACCESS_WRITE if is_store else ACCESS_READ, 0))
        self._kill_first_reg(cs_insn, _ppc.PPC_OP_REG)
        return info


__all__ = ["ACCESS_READ", "ACCESS_WRITE", "Flow", "FlowAnalyzer", "InsnInfo"]
