# kuna backend

[kuna](https://github.com/Noelo-Lab/kuna) is a stand-alone, SLEIGH based
decompiler shipped as a single CLI. It keeps no database. The kuna
backend (`quokka.backends.kuna`) drives this CLI and writes a regular
`.quokka` file. Any Quokka consumer can load that file, for example
[qbindiff](https://github.com/quarkslab/qbindiff). You don't need IDA,
Ghidra or Binary Ninja.

## Requirements

- The `kuna` executable, either on `PATH` or pointed to by `$KUNA_PATH`
- [LIEF](https://lief.re), installed with the `kuna` extra:

```commandline
$ pip install 'quokka-project[kuna]'
```

## How it works

kuna provides the function inventory (names, sizes, kinds), the
disassembly listing, string literals and, optionally, the decompiled code
and prototypes. The backend adds the rest:

- **Binary layout** (with LIEF): segments, permissions, file offsets,
  exported symbols, data symbols, import slots (ELF `JUMP_SLOT`/`GLOB_DAT`
  relocations, Mach-O bindings, PE IAT) and ARM/Thumb mapping symbols.
- **Basic blocks and CFG** (with capstone): rebuilt by recursive descent
  from each function entry, following IDA's conventions. Calls don't end
  a block, calls to non-returning functions end it as `NORET`, and both
  edges of a conditional branch are `JUMP_COND`.
- **References**: direct calls and jumps, calls through import or
  function-pointer slots, and data reads and writes, with addresses
  resolved by tracking constants within each block (`adrp`/`add`,
  `lea`/RIP-relative, ARM literal pools...).
- **Data**: strings (`char[]` / `wchar_t[]`), data symbols, and the slots
  that the code references.

## Usage

```python
import quokka
from quokka.types import Disassembler

prog = quokka.Program.from_binary("/bin/ls", disassembler=Disassembler.KUNA)
```

Or directly through the backend, for more control:

```python
from quokka.backends.kuna import export_binary

export_binary(
    "/bin/ls",
    "ls.quokka",
    decompiled=True,          # store decompiled code and prototypes
    jobs="auto",              # kuna workers when decompiling
    kuna_args=["--mode", "fast"],
)
```

From the command line:

```commandline
$ quokka-cli --backend kuna -t 8 dir/
$ quokka-cli --backend kuna --kuna-path ~/bin/kuna -m full --decompiled binary
```

`auto` backend detection prefers IDA, then Ghidra, then kuna.

### With qbindiff

```python
from qbindiff import Program, QBinDiff, Distance
from qbindiff.features import WeisfeilerLehman, Constant

primary = Program.from_quokka(export_binary("v1"), "v1")
secondary = Program.from_quokka(export_binary("v2"), "v2")
differ = QBinDiff(primary, secondary, distance=Distance.canberra)
differ.register_feature_extractor(WeisfeilerLehman, 1.0, distance=Distance.cosine)
differ.register_feature_extractor(Constant, 1.0)
differ.process()
matches = differ.compute_matching()
```

## Supported targets

| Format          | Architectures                         | Status       |
|-----------------|---------------------------------------|--------------|
| ELF (exec, DSO) | x86, x86-64, ARM (ARM/Thumb), AArch64 | tested       |
| Mach-O          | x86-64, AArch64                       | tested       |
| ELF             | MIPS, PowerPC                         | experimental |
| PE              | x86, x86-64, AArch64                  | experimental |

Validation on the repository dataset, compared with the IDA exports:

- `many_types_cpp`: identical basic blocks, CFG edges and instructions for
  all 38 normal functions.
- `puraUpdate` (ARM32): identical for all 18 common functions, and
  112 of the 113 call edges match.

## Limitations

- **Relocatable objects** (`.o`, `.ko`) are refused, because their
  addresses aren't final.
- **Jump tables** aren't resolved. Blocks only reachable through an
  indirect jump are still exported (seeded from the function extent), but
  without an edge from the dispatch block.
- **Imports** are represented as functions without body at their
  GOT/IAT slot, instead of in a synthetic `extern` segment as IDA does.
  Calls through the slot (`call [rip+x]`, `blr x16` after `ldr`) and PLT
  stubs reference them.
- **Data typing** is shallow: strings, scalar data symbols and pointer
  slots. No structures are recovered. In FULL mode with `--decompiled`,
  the prototypes come from kuna.
- **Function boundaries** follow kuna. kuna sometimes merges a function
  into the one before it, where IDA would split them.
- kuna keeps no database, so `Program.commit()` isn't supported. Edits
  can't be written back.
