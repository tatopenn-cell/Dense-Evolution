import re
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Data model
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class QASMCircuit:
    """
    Parsed representation of an OpenQASM 2.0 / 3.0 circuit.

    Attributes
    ----------
    n_qubits : total qubit count declared in qreg / qubit statements
    n_cbits  : total classical bit count declared in creg / bit statements
    ops      : list of gate dicts — each dict has keys:
                 'type'   : 'gate'
                 'name'   : lowercase gate name (aliases resolved)
                 'qubits' : list[int]  — absolute qubit indices
                 'params' : list[float] — evaluated rotation angles
    """
    n_qubits: int = 0
    n_cbits:  int = 0
    ops: List[Dict] = field(default_factory=list)

    def to_tuples(self) -> List[Tuple]:
        """
        Convert ops to the tuple format expected by DenseSVSimulator.run_circuit:
            (name, qubit0[, qubit1, ...][, param0, ...])

        BUG FIX (original): the original returned
            (name,) + tuple(qubits) + tuple(params)
        which placed params *after* qubits, but run_circuit expects
        params interleaved or trailing depending on gate type.
        For the standard (name, qubit, param) convention used throughout
        the simulator, this ordering is correct — preserved here but
        documented explicitly so callers know what to expect.
        """
        out = []
        for op in self.ops:
            row = (op['name'],) + tuple(op['qubits']) + tuple(op['params'])
            out.append(row)
        return out


# ─────────────────────────────────────────────────────────────────────────────
# Parser
# ─────────────────────────────────────────────────────────────────────────────

class QASMParser:
    """
    Robust OpenQASM 2.0 / 3.0 parser.

    Supported features
    ------------------
    - qreg / creg  (QASM 2.0)
    - qubit / bit  (QASM 3.0)
    - Parametric gates: rx, ry, rz, p, u1, u2, u3, cp, crz, ...
    - Compound parameter expressions: pi/2, sqrt(2), cos(0.3), ...
    - Block comments  /* ... */  and line comments  // ...
    - Gate aliases: cu1→cp, u1→p, toffoli→ccx, cnot→cx, ...
    - Range syntax  q[0:3]  expanded to individual qubits
    - Bare register name (no index) resolved to qubit 0 of that register
    - Silent fallback (0.0) for unparseable parameter expressions
    """

    # ── compiled regexes ────────────────────────────────────────────
    _RE_BLOCK_CMT  = re.compile(r'/\*.*?\*/', re.DOTALL)
    _RE_LINE_CMT   = re.compile(r'//[^\n]*')
    _RE_INDEX      = re.compile(r'\[(\d+)\]')
    _RE_RANGE      = re.compile(r'^([a-zA-Z_]\w*)\[(\d+):(\d+)\]$')  # q[0:3]
    _RE_QREG2      = re.compile(r'^qreg\s+([a-zA-Z_]\w*)\s*\[(\d+)\]')
    _RE_CREG2      = re.compile(r'^creg\s+([a-zA-Z_]\w*)\s*\[(\d+)\]')
    _RE_QREG3      = re.compile(r'^qubit(?:\s*\[(\d+)\])?\s+([a-zA-Z_]\w*)')
    _RE_CREG3      = re.compile(r'^bit(?:\s*\[(\d+)\])?\s+([a-zA-Z_]\w*)')
    _RE_GATE_HEAD  = re.compile(r'^([a-zA-Z_]\w*)(?:\((.*)\))?$')

    # ── gate name aliases ────────────────────────────────────────────
    _ALIAS: Dict[str, str] = {
        'cu1':     'cp',
        'u1':      'p',
        'toffoli': 'ccx',
        'fredkin': 'cswap',
        'cnot':    'cx',
        'not':     'x',
        'id':      'i',
        'cx':      'cx',    # explicit identity mappings for safety
        'cz':      'cz',
        'ccx':     'ccx',
    }

    # ── gate names that only ever take one qubit ─────────────────────
    # Used to expand range syntax (q[0:3]) into one op per qubit instead
    # of a single op with multiple qubits attached — see BUG FIX 3 note
    # on parse(): the original fix resolved q[0:3] to the qubit list
    # [0,1,2], but nothing expanded that list into separate applications
    # for gates that are only ever single-qubit, so e.g. `h q[0:3]` ended
    # up applying H to qubit 0 only, silently dropping qubits 1 and 2.
    _SINGLE_QUBIT_GATES = frozenset((
        'h', 'x', 'y', 'z', 's', 'sdg', 't', 'tdg', 'sx', 'id',
        'rx', 'ry', 'rz', 'p', 'u1', 'u2', 'u3',
    ))

    # ── statements to skip entirely ──────────────────────────────────
    # BUG FIX (original): 'gate ' had a trailing space making it miss
    # 'gate foo(...)' where the token is 'gate' followed by space.
    # Using startswith on lowercased tokens is correct but the original
    # also skipped 'def ' and 'for ' which are QASM 3.0 keywords —
    # kept here for forward compatibility.
    _SKIP = frozenset((
        'openqasm', 'include', 'barrier', 'measure',
        'reset', 'gate', 'def', 'if', 'for', 'while',
    ))

    # ── safe math environment for eval() ────────────────────────────
    _MATH_ENV: Dict = {
        '__builtins__': {},
        'pi':     np.pi,
        'tau':    2.0 * np.pi,
        'euler':  np.e,
        'np':     np,
        'sin':    np.sin,   'cos':    np.cos,   'tan':    np.tan,
        'sqrt':   np.sqrt,  'exp':    np.exp,   'log':    np.log,
        'asin':   np.arcsin,'acos':   np.arccos,'atan':   np.arctan,
        'arcsin': np.arcsin,'arccos': np.arccos,'arctan': np.arctan,
        'abs':    abs,      'round':  round,
    }

    # ────────────────────────────────────────────────────────────────
    # Public interface
    # ────────────────────────────────────────────────────────────────

    def parse(self, qasm_str: str) -> QASMCircuit:
        """
        Parse an OpenQASM 2.0 or 3.0 string into a QASMCircuit.

        BUG FIX 1 (original): the original joined all lines with a single
        space then split on ';'.  Multi-line gate definitions (gate foo ...)
        were not stripped before joining, causing 'gate foo ...' to appear
        as a runnable instruction.  Fixed by stripping comments *before*
        joining and by using the frozenset _SKIP check on the first token.

        BUG FIX 2 (original): bare register names (e.g. 'h q' instead of
        'h q[0]') were silently dropped if the register had more than one
        qubit, because qubit_map only stored 'name[0]' → 0 for size-1
        registers.  Fixed: bare names always map to qubit 0 of that register
        regardless of register size.

        BUG FIX 3 (original): range syntax q[0:3] was never handled —
        such tokens fell through to the digit-extraction fallback which
        returned only the last digit.  Fixed in _resolve_qubits.
        """
        qubit_map: Dict[str, int] = {}
        cbit_map:  Dict[str, int] = {}
        n_qubits = 0
        n_cbits  = 0
        ops: List[Dict] = []

        # ── strip comments ───────────────────────────────────────────
        cleaned = self._RE_BLOCK_CMT.sub(' ', qasm_str)
        cleaned = self._RE_LINE_CMT.sub(' ', cleaned)

        # ── split into statements ─────────────────────────────────────
        statements = [s.strip() for s in cleaned.split(';') if s.strip()]

        for instr in statements:
            # collapse internal whitespace runs to a single space
            instr = re.sub(r'\s+', ' ', instr).strip()
            if not instr:
                continue

            # first token (before any space or '(') for keyword detection
            first_token = re.split(r'[\s(]', instr)[0].lower()
            if first_token in self._SKIP:
                continue

            # ── qreg (QASM 2.0) ─────────────────────────────────────
            m = self._RE_QREG2.match(instr)
            if m:
                reg_name, sz = m.group(1), int(m.group(2))
                for i in range(sz):
                    qubit_map[f'{reg_name}[{i}]'] = n_qubits + i
                qubit_map[reg_name] = n_qubits   # bare name → first qubit
                n_qubits += sz
                continue

            # ── creg (QASM 2.0) ─────────────────────────────────────
            m = self._RE_CREG2.match(instr)
            if m:
                reg_name, sz = m.group(1), int(m.group(2))
                for i in range(sz):
                    cbit_map[f'{reg_name}[{i}]'] = n_cbits + i
                cbit_map[reg_name] = n_cbits
                n_cbits += sz
                continue

            # ── qubit (QASM 3.0) ─────────────────────────────────────
            m = self._RE_QREG3.match(instr)
            if m:
                sz_s, reg_name = m.group(1), m.group(2)
                sz = int(sz_s) if sz_s else 1
                for i in range(sz):
                    qubit_map[f'{reg_name}[{i}]'] = n_qubits + i
                qubit_map[reg_name] = n_qubits
                n_qubits += sz
                continue

            # ── bit (QASM 3.0) ───────────────────────────────────────
            m = self._RE_CREG3.match(instr)
            if m:
                sz_s, reg_name = m.group(1), m.group(2)
                sz = int(sz_s) if sz_s else 1
                for i in range(sz):
                    cbit_map[f'{reg_name}[{i}]'] = n_cbits + i
                cbit_map[reg_name] = n_cbits
                n_cbits += sz
                continue

            # ── gate application ─────────────────────────────────────
            op = self._parse_gate(instr, qubit_map)
            if op is not None:
                if op['name'] in self._SINGLE_QUBIT_GATES and len(op['qubits']) > 1:
                    # range syntax on an inherently single-qubit gate
                    # (e.g. `h q[0:3]`) — expand into one op per qubit.
                    for q in op['qubits']:
                        ops.append({
                            'type': op['type'], 'name': op['name'],
                            'qubits': [q], 'params': list(op['params']),
                        })
                        n_qubits = max(n_qubits, q + 1)
                else:
                    ops.append(op)
                    # update n_qubits from seen qubit indices
                    # (handles circuits without explicit qreg declarations)
                    if op['qubits']:
                        n_qubits = max(n_qubits, max(op['qubits']) + 1)

        return QASMCircuit(n_qubits, n_cbits, ops)

    def validate(self, circ: QASMCircuit) -> Tuple[bool, str]:
        """Light structural validation — does not verify gate semantics."""
        if circ.n_qubits <= 0:
            return False, 'n_qubits must be > 0.'
        if not circ.ops:
            return False, 'No gate operations found in circuit.'
        # check for out-of-range qubit references
        for i, op in enumerate(circ.ops):
            for q in op.get('qubits', []):
                if not (0 <= q < circ.n_qubits):
                    return False, (
                        f"Gate '{op['name']}' at op[{i}] references "
                        f"qubit {q} but n_qubits={circ.n_qubits}.")
        return True, 'OK'

    # ────────────────────────────────────────────────────────────────
    # Private helpers
    # ────────────────────────────────────────────────────────────────

    def _parse_gate(self,
                    instr:     str,
                    qubit_map: Dict[str, int]) -> Optional[Dict]:
        """
        Parse a single gate instruction into an op dict.

        BUG FIX 4 (original): the original code had two independent
        code paths for extracting param_str — one using _RE_GATE_HEAD
        and one rescanning for '(' — that could disagree, leaving
        param_str as the group(2) of an earlier (shorter) match while
        paren_start/paren_end referred to a different range.  Unified
        into a single pass that:
          1. finds the parameter parentheses (balanced),
          2. extracts everything before '(' as the gate name,
          3. extracts everything after the closing ')' as the qubit list.

        BUG FIX 5 (original): split_at was found by scanning for the
        first space at depth==0 *in the whole instruction*, so for
            rx(pi/2) q[0]
        split_at was -1 (no space outside parens in 'rx(pi/2)') and
        rest was '' — dropping the qubit entirely.  Fixed by splitting
        on the space after the closing ')'.
        """
        instr = instr.strip()

        # ── locate parameter block '(...)' ───────────────────────────
        paren_open  = instr.find('(')
        paren_close = -1
        param_str   = ''

        if paren_open != -1:
            depth = 0
            for idx in range(paren_open, len(instr)):
                if instr[idx] == '(':
                    depth += 1
                elif instr[idx] == ')':
                    depth -= 1
                    if depth == 0:
                        paren_close = idx
                        break
            if paren_close == -1:
                # Unbalanced parentheses — skip this instruction
                return None
            param_str = instr[paren_open + 1 : paren_close].strip()
            # gate_head = everything before '(', qubit_part = everything after ')'
            gate_head  = instr[:paren_open].strip()
            qubit_part = instr[paren_close + 1:].strip()
        else:
            # No parameters: split on first whitespace
            parts      = instr.split(None, 1)
            gate_head  = parts[0]
            qubit_part = parts[1] if len(parts) > 1 else ''

        gate_name_raw = gate_head.strip().lower()
        if not gate_name_raw:
            return None

        gate_name = self._ALIAS.get(gate_name_raw, gate_name_raw)

        # ── parse parameters ─────────────────────────────────────────
        params: List[float] = []
        if param_str:
            for tok in self._split_params(param_str):
                tok = tok.strip()
                if not tok:
                    continue
                params.append(self._eval_param(tok))

        # ── resolve qubits ───────────────────────────────────────────
        qubits = self._resolve_qubits(
            qubit_part.replace(' ', ''), qubit_map)

        if not qubits:
            return None

        return {
            'type':   'gate',
            'name':   gate_name,
            'qubits': qubits,
            'params': params,
        }

    def _eval_param(self, tok: str) -> float:
        """
        Evaluate a parameter token to float.

        Handles: numeric literals, pi, pi/2, sqrt(2), cos(0.3), etc.
        Returns 0.0 on any evaluation error (silent fallback).
        """
        try:
            return float(eval(tok, self._MATH_ENV))   # noqa: S307
        except Exception:
            return 0.0

    @staticmethod
    def _split_params(s: str) -> List[str]:
        """
        Split a comma-separated parameter string respecting nested
        parentheses.  e.g. 'pi/2, atan(1,0)' → ['pi/2', 'atan(1,0)']
        """
        tokens: List[str] = []
        cur:    List[str] = []
        depth = 0
        for ch in s:
            if ch == '(':
                depth += 1
                cur.append(ch)
            elif ch == ')':
                depth -= 1
                cur.append(ch)
            elif ch == ',' and depth == 0:
                tokens.append(''.join(cur).strip())
                cur = []
            else:
                cur.append(ch)
        if cur:
            tokens.append(''.join(cur).strip())
        return [t for t in tokens if t]

    def _resolve_qubits(self,
                         s:    str,
                         qmap: Dict[str, int]) -> List[int]:
        r"""
        Resolve a comma-separated qubit argument string to absolute indices.

        Handles
        -------
        - Indexed:  q[0], q[1]
        - Bare:     q  → qmap['q']  (first qubit of that register)
        - Range:    q[0:3]  → [qmap['q[0]'], qmap['q[1]'], qmap['q[2]']]

        BUG FIX 6 (original): range syntax q[0:3] was not handled and
        fell through to the digit-extraction fallback, returning only
        the last number found (e.g., 3 instead of [0,1,2]).

        BUG FIX 7 (original): the fallback `digits = re.findall(r'\d+', tok)`
        was used as a last resort — this could silently map unknown tokens
        to arbitrary integers.  Now the fallback is gated on the absence of
        any letter character to avoid mapping named registers that are simply
        not yet in qmap to wrong indices.
        """
        out: List[int] = []
        for tok in s.split(','):
            tok = tok.strip()
            if not tok:
                continue

            # ── range syntax: q[start:end] ───────────────────────────
            m = self._RE_RANGE.match(tok)
            if m:
                base  = m.group(1)
                start = int(m.group(2))
                end   = int(m.group(3))   # exclusive upper bound
                for i in range(start, end):
                    key = f'{base}[{i}]'
                    if key in qmap:
                        out.append(qmap[key])
                continue

            # ── direct map lookup ─────────────────────────────────────
            if tok in qmap:
                out.append(qmap[tok])
                continue

            # ── indexed: base[n] ─────────────────────────────────────
            bracket = self._RE_INDEX.search(tok)
            if bracket:
                base = tok[:tok.index('[')]
                key  = f'{base}[{bracket.group(1)}]'
                if key in qmap:
                    out.append(qmap[key])
                    continue
                # index not in map — try numeric fallback
                out.append(int(bracket.group(1)))
                continue

            # ── bare name not in map: try stripping to digits ─────────
            # Only do this when the token contains no letters (pure numeric)
            # to avoid misidentifying unknown register names.
            digits = re.findall(r'\d+', tok)
            if digits and not re.search(r'[a-zA-Z_]', tok):
                out.append(int(digits[-1]))

        return out
