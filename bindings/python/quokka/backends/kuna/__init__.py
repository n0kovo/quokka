"""Kuna backend: export a binary to Quokka with the kuna decompiler.

kuna (https://github.com/Noelo-Lab/kuna) is a stand-alone SLEIGH based
decompiler CLI. This backend drives it, rebuilds the basic blocks from its
instruction listing and writes a regular ``.quokka`` file, usable by any
Quokka consumer (qbindiff, ...).

Requires the ``kuna`` executable (on ``PATH`` or in ``$KUNA_PATH``) and LIEF
(``pip install 'quokka-project[kuna]'``).
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

from .cli import KUNA_PATH_ENV, KunaCli, KunaError, find_kuna
from .exporter import KunaExporter, export_binary

__all__ = ["KUNA_PATH_ENV", "KunaCli", "KunaError", "KunaExporter", "export_binary", "find_kuna"]
