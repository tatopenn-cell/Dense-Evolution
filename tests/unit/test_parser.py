"""
Unit tests for dense_evolution/parser.py -- QASMParser (2.0/3.0 parsing,
range syntax, for-loops, validate(), AST-sandboxed expression evaluation)
and QASMCircuit's iterability.

Split out of the original monolithic test_dense_evolution.py -- see
test_simulator.py's module docstring for why.
"""
import numpy as np
import pytest

from dense_evolution import DenseSVSimulator, QASMParser, QASMCircuit, Chunk, QuantumTranspiler

from _helpers import probs


def test_backward_compat_shim_parser_reexports_qasmparser_and_qasmcircuit():
    # dense_evolution.parser is the Phase 2 backward-compat shim left at
    # the old top-level path -- nothing else in this suite imports through
    # it (this file, like everything else, gets QASMParser/QASMCircuit via
    # the top-level dense_evolution package, which now sources them from
    # dense_evolution.circuits.parser directly), so without this the
    # shim'''s own lines go uncovered and a broken shim would go undetected
    # by CI.
    from dense_evolution.parser import QASMParser as shim_qasmparser, QASMCircuit as shim_qasmcircuit
    assert shim_qasmparser is QASMParser
    assert shim_qasmcircuit is QASMCircuit


# ─────────────────────────────────────────────────────────────
# QASM RANGE SYNTAX
# ─────────────────────────────────────────────────────────────

class TestQASMRangeSyntax:
    """Regression guard for audit finding #2: `gate q[a:b]` on an inherently
    single-qubit gate used to attach all resolved qubits to ONE op, so only
    the first qubit was ever actually gated — the rest were silently dropped
    with no error, and probabilities still summed to 1. The parser's own
    docstring already promised "range syntax expanded to individual qubits";
    parse() now honors that by emitting one op per qubit instead of one op
    carrying the whole list."""

    def test_range_syntax_expands_to_separate_ops(self):
        qasm = 'OPENQASM 2.0; include "qelib1.inc"; qreg q[4]; h q[0:3];'
        circ = QASMParser().parse(qasm)
        assert len(circ.ops) == 3
        assert [op['qubits'] for op in circ.ops] == [[0], [1], [2]]
        assert all(op['name'] == 'h' for op in circ.ops)

    def test_range_syntax_produces_correct_superposition(self, sim4):
        qasm = 'OPENQASM 2.0; include "qelib1.inc"; qreg q[4]; h q[0:3];'
        circ = QASMParser().parse(qasm)
        sim4.run_circuit_jit([[op['name'], op['qubits'][0], -1] for op in circ.ops])
        p = probs(sim4)
        # q0,q1,q2 uniform superposition, q3 untouched -> 8 equally likely states
        nonzero = np.where(p > 1e-9)[0]
        assert len(nonzero) == 8
        assert np.allclose(p[nonzero], 1.0 / 8, atol=1e-9)

    def test_two_qubit_gate_qubit_list_is_not_expanded(self):
        # sanity check the fix is scoped to single-qubit gate names only —
        # a genuine 2-qubit gate must keep both its qubits on one op
        qasm = 'OPENQASM 2.0; include "qelib1.inc"; qreg q[2]; cx q[0],q[1];'
        circ = QASMParser().parse(qasm)
        assert len(circ.ops) == 1
        assert circ.ops[0]['qubits'] == [0, 1]


class TestQASMForLoop:
    """QASM 3.0 `for`-loops are brace-delimited, not ';'-terminated — the
    parser used to split statements on ';' alone, so a `for ... { ... }`
    block both lost its own body (never extracted) AND corrupted whatever
    real statement followed it on the same line (the stray closing '}'
    merged with the next statement's text into one garbage op). Verified
    directly: `for int i in [0:2] { h q[i]; } cx q[0],q[1];` used to produce
    a single ghost op named '}' and silently drop both the loop body and
    the real cx — the executed circuit stayed |000> at 100% probability
    with no error. _process_block_constructs now unrolls resolvable `for`
    loops and cleanly strips `if`/`while`/`def` blocks before the ';'-split
    ever runs, needed for VQE ansätze written with a loop over qubits."""

    def test_for_loop_body_extracted_and_following_gate_preserved(self):
        qasm = '''
        qreg q[3];
        for int i in [0:2] { h q[i]; }
        cx q[0], q[1];
        '''
        circ = QASMParser().parse(qasm)
        assert len(circ.ops) == 4
        assert [op['name'] for op in circ.ops] == ['h', 'h', 'h', 'cx']
        assert [op['qubits'] for op in circ.ops] == [[0], [1], [2], [0, 1]]

    def test_for_loop_executes_to_real_ghz_not_ghost_op(self, sim3):
        qasm = '''
        qreg q[3];
        for int i in [0:2] { h q[i]; }
        cx q[0], q[1];
        '''
        circ = QASMParser().parse(qasm)
        sim3.run_circuit(circ.to_tuples())
        p = probs(sim3)
        # not the pre-fix bug (|000> at 100%): real superposition present
        assert p[0] < 0.99

    def test_for_loop_bound_resolved_from_declared_int_variable(self):
        qasm = '''
        int n = 3;
        qreg q[3];
        for int i in [0:n-1] { rx(0.5) q[i]; }
        '''
        circ = QASMParser().parse(qasm)
        assert len(circ.ops) == 3
        assert all(op['name'] == 'rx' and op['params'] == [0.5] for op in circ.ops)
        assert [op['qubits'] for op in circ.ops] == [[0], [1], [2]]

    def test_for_range_is_inclusive_of_end_bound(self):
        # QASM3 for-range [0:2] must cover indices 0,1,2 (three iterations) —
        # unlike this parser's own EXCLUSIVE q[a:b] qubit-range syntax.
        qasm = 'qreg q[3]; for i in [0:2] { x q[i]; }'
        circ = QASMParser().parse(qasm)
        assert [op['qubits'][0] for op in circ.ops] == [0, 1, 2]

    def test_for_loop_body_with_multiple_statements_expands_all(self):
        qasm = 'qreg q[2]; for i in [0:1] { h q[i]; x q[i]; }'
        circ = QASMParser().parse(qasm)
        assert [op['name'] for op in circ.ops] == ['h', 'x', 'h', 'x']
        assert [op['qubits'][0] for op in circ.ops] == [0, 0, 1, 1]

    def test_if_block_does_not_corrupt_following_statement(self):
        qasm = 'qreg q[2]; if (c==1) { x q[0]; } h q[1];'
        circ = QASMParser().parse(qasm)
        assert len(circ.ops) == 1
        assert circ.ops[0] == {'type': 'gate', 'name': 'h', 'qubits': [1], 'params': []}

    def test_no_block_constructs_is_a_no_op(self):
        # plain circuits with no for/if/while/def must be completely
        # unaffected by _process_block_constructs
        qasm = 'qreg q[2]; h q[0]; cx q[0], q[1];'
        circ = QASMParser().parse(qasm)
        assert [op['name'] for op in circ.ops] == ['h', 'cx']

    # -- unresolvable-bound / multi-construct coverage --------------------
    # Area verified separately (RAM-unconstrained environment): an
    # unresolvable `for` bound falls through to the exact same
    # `replacement = ''` strip path as if/while/def (see
    # _process_block_constructs docstring) -- these tests exercise that
    # specific trigger (an undeclared bound variable, so
    # _resolve_int_expr returns None) rather than assuming the shared
    # code path is equivalent without checking.

    def test_unresolvable_for_loop_bound_stripped_following_gate_preserved(self):
        # 'n' is never declared -- _resolve_int_expr must return None for
        # it (confirmed by reading _eval_ast_node: an ast.Name not in env
        # falls through to the final `raise`, caught by _resolve_int_expr's
        # except-Exception), so this for loop takes the strip path, not
        # the unroll path.
        qasm = '''
        qreg q[3];
        for int i in [0:n] { h q[i]; }
        cx q[0], q[1];
        '''
        circ = QASMParser().parse(qasm)
        assert [op['name'] for op in circ.ops] == ['cx']
        assert circ.ops[0]['qubits'] == [0, 1]

    def test_unresolvable_for_loop_stripped_execution_matches_bare_circuit(self, sim3):
        # Same pattern as the v8.1.13 regression tests: compare actual
        # probabilities, not just the op list, against an equivalent
        # circuit written without the unresolvable loop at all.
        qasm_with_loop = '''
        qreg q[3];
        for int i in [0:n] { h q[i]; }
        cx q[0], q[1];
        '''
        circ = QASMParser().parse(qasm_with_loop)
        sim3.run_circuit(circ.to_tuples())
        p_with_loop = probs(sim3)

        ref = DenseSVSimulator(n_qubits=3, use_gpu=False, use_float32=False)
        ref_circ = QASMParser().parse('qreg q[3]; cx q[0], q[1];')
        ref.run_circuit(ref_circ.to_tuples())
        p_ref = probs(ref)

        np.testing.assert_allclose(p_with_loop, p_ref, atol=1e-12)

    def test_while_block_does_not_corrupt_following_statement(self):
        qasm = 'qreg q[2]; while (c==1) { x q[0]; } h q[1];'
        circ = QASMParser().parse(qasm)
        assert len(circ.ops) == 1
        assert circ.ops[0] == {'type': 'gate', 'name': 'h', 'qubits': [1], 'params': []}

    def test_multiple_unresolvable_constructs_in_sequence(self):
        # for (unresolvable) + if + while, each stripped in turn, valid
        # gates interleaved between and after every one of them survive.
        qasm = '''
        qreg q[3];
        h q[0];
        for int i in [0:n] { x q[i]; }
        x q[1];
        if (c==1) { y q[0]; }
        y q[2];
        while (c==1) { z q[0]; }
        cx q[0], q[2];
        '''
        circ = QASMParser().parse(qasm)
        assert [op['name'] for op in circ.ops] == ['h', 'x', 'y', 'cx']
        assert [op['qubits'] for op in circ.ops] == [[0], [1], [2], [0, 2]]

    def test_resolvable_for_then_unresolvable_if_then_valid_code(self):
        # Combination the changelog's original fix never exercised: a
        # resolvable for-loop (real unrolling, not stripping) immediately
        # followed by an unresolvable-condition if (stripping) followed by
        # more valid code -- confirms the unroll doesn't shift/corrupt the
        # search position _process_block_constructs uses to find the next
        # block.
        qasm = '''
        qreg q[3];
        for int i in [0:1] { h q[i]; }
        if (some_undeclared_condition) { x q[2]; }
        cx q[0], q[1];
        '''
        circ = QASMParser().parse(qasm)
        assert [op['name'] for op in circ.ops] == ['h', 'h', 'cx']
        assert [op['qubits'] for op in circ.ops] == [[0], [1], [0, 1]]


class TestQASMParserValidateAndEdgeCases:
    """QASMParser.validate() was never called by any existing test (only
    parse() itself), plus a handful of parser edge-case branches (QASM3
    `bit[N]` classical register syntax, unbalanced braces/parentheses,
    multi-parameter gates, and out-of-declared-range indexed qubits)."""

    def test_validate_accepts_well_formed_circuit(self):
        qasm = 'OPENQASM 2.0; include "qelib1.inc"; qreg q[2]; h q[0]; cx q[0],q[1];'
        circ = QASMParser().parse(qasm)
        ok, msg = QASMParser().validate(circ)
        assert ok is True
        assert msg == 'OK'

    def test_validate_rejects_zero_qubits(self):
        empty = QASMCircuit(0, 0, [])
        ok, msg = QASMParser().validate(empty)
        assert ok is False
        assert 'n_qubits' in msg

    def test_validate_rejects_no_ops(self):
        no_ops = QASMCircuit(2, 0, [])
        ok, msg = QASMParser().validate(no_ops)
        assert ok is False
        assert 'No gate operations' in msg

    def test_validate_rejects_out_of_range_qubit_reference(self):
        bad = QASMCircuit(2, 0, [{'type': 'gate', 'name': 'h', 'qubits': [5], 'params': []}])
        ok, msg = QASMParser().validate(bad)
        assert ok is False
        assert 'qubit 5' in msg

    def test_qasm3_bit_declaration(self):
        qasm = 'OPENQASM 3.0; qreg q[2]; bit[2] c; h q[0]; cx q[0],q[1];'
        circ = QASMParser().parse(qasm)
        assert circ.n_cbits == 2
        assert [op['name'] for op in circ.ops] == ['h', 'cx']

    def test_unbalanced_parentheses_in_gate_call_skipped(self):
        # 'rx(0.5 q[0];' -- missing closing paren -- must be skipped
        # (returns None internally, no op emitted), not raise.
        qasm = 'OPENQASM 2.0; include "qelib1.inc"; qreg q[2]; rx(0.5 q[0]; h q[1];'
        circ = QASMParser().parse(qasm)
        assert [op['name'] for op in circ.ops] == ['h']

    def test_unbalanced_braces_in_for_loop_bail_out(self):
        # A for-loop construct with a missing closing brace must not hang
        # or raise -- _process_block_constructs bails out and the rest is
        # handled by whatever the existing fallback does.
        qasm = '''
        OPENQASM 3.0;
        qreg q[2];
        for int i in [0:1] { h q[i];
        cx q[0], q[1];
        '''
        circ = QASMParser().parse(qasm)  # must not hang/raise
        assert isinstance(circ.ops, list)

    def test_multi_parameter_gate_comma_split(self):
        # u2(phi, lam) -- a real 2-parameter gate, exercises _split_params's
        # depth==0 comma-splitting branch (every other parametric-gate test
        # elsewhere in this file uses a single parameter).
        qasm = 'OPENQASM 2.0; include "qelib1.inc"; qreg q[1]; u2(0.1,0.2) q[0];'
        circ = QASMParser().parse(qasm)
        assert circ.ops[0]['name'] == 'u2'
        assert circ.ops[0]['params'] == pytest.approx([0.1, 0.2])

    def test_indexed_qubit_beyond_declared_range_uses_numeric_fallback(self):
        # q[5] on a 2-qubit qreg -- not in qmap, falls back to the literal
        # index rather than being dropped (BUG FIX 7's documented gate,
        # only for tokens with no letters).
        qasm = 'OPENQASM 2.0; include "qelib1.inc"; qreg q[2]; h q[5];'
        circ = QASMParser().parse(qasm)
        assert circ.ops[0]['qubits'] == [5]


class TestQASMCircuitIterable:
    """Found via a user's own Colab testing: Chunk.run_chunk(circuit) (and
    QuantumTranspiler.transpile, which it calls internally) iterates
    directly over its `circuit` argument — `for cmd in circuit`. Passing a
    QASMCircuit straight from QASMParser().parse(...) (instead of calling
    .to_tuples() first) used to raise `TypeError: 'QASMCircuit' object is
    not iterable`, a real usability gap for a very natural usage pattern.
    Fixed by adding __iter__, duck-typing QASMCircuit as an iterable of the
    same tuples to_tuples() already returns — no existing call site inside
    dense_evolution relied on QASMCircuit being non-iterable."""

    def test_iterating_a_qasmcircuit_matches_to_tuples(self):
        circ = QASMParser().parse('qreg q[2]; h q[0]; cx q[0],q[1]; rz(0.5) q[1];')
        assert list(circ) == circ.to_tuples()

    def test_chunk_run_chunk_accepts_a_bare_qasmcircuit(self):
        circ = QASMParser().parse('qreg q[2]; h q[0]; cx q[0],q[1];')
        ch = Chunk(2)
        ch.run_chunk(circ)  # used to raise TypeError without __iter__
        probs_ = np.asarray(ch.get_probabilities())
        assert abs(probs_.sum() - 1.0) < 1e-9

    def test_transpile_accepts_a_bare_qasmcircuit(self):
        circ = QASMParser().parse('qreg q[3]; ccx q[0],q[1],q[2];')
        expanded = QuantumTranspiler.transpile(circ)
        assert len(expanded) == 15


class TestParserEvalSecurity:
    """_eval_param (gate parameters) and _resolve_int_expr (for-loop bounds)
    used to call raw eval() with only `{'__builtins__': {}}` as protection —
    that blocks bare builtin names (open, len, __import__...) but does
    NOT block attribute/dunder traversal of the live object graph, which
    needs no builtin name at all. Verified directly: a gate parameter of
    `().__class__.__bases__[0].__subclasses__().__len__()`, passed through
    the public QASMParser.parse() entry point, executed successfully and
    returned a real value (2200.0, the live subclass count) before this
    fix — a genuine code-execution vulnerability, not a hypothetical one.
    Both now go through _eval_ast_node, an AST node-type whitelist with no
    eval()/exec() anywhere — an attribute access is an ast.Attribute node,
    which is never one of the handled cases, so '.' in an expression always
    lands in the rejection branch structurally, not via a blocklist."""

    _ESCAPE_PAYLOADS = [
        '().__class__.__bases__[0].__subclasses__().__len__()',
        '__import__("os").system("echo pwned")',
        'getattr(1, "__class__")',
        '[x for x in range(10)][0]',
        '(lambda: 1)()',
        'exec("1")',
        'globals()',
        '().__class__.__init__.__globals__',
    ]

    @pytest.mark.parametrize('payload', _ESCAPE_PAYLOADS)
    def test_eval_param_blocks_sandbox_escapes(self, payload):
        # _eval_param used to swallow every rejected expression into a
        # silent 0.0 (same fallback a genuine typo like 'pi * / 2' hit
        # too); it now raises ValueError instead -- still structurally
        # blocked (the AST whitelist never reaches these nodes), just
        # explicit about it instead of silent, same as a malformed
        # expression from a typo.
        with pytest.raises(ValueError):
            QASMParser()._eval_param(payload)

    @pytest.mark.parametrize('payload', _ESCAPE_PAYLOADS)
    def test_resolve_int_expr_blocks_sandbox_escapes(self, payload):
        assert QASMParser()._resolve_int_expr(payload, {}) is None

    def test_original_exploit_through_full_parse_raises(self):
        # end-to-end through the actual public entry point, not just the
        # internal method directly
        qasm = ('OPENQASM 3.0; qubit[1] q; '
                'rx(().__class__.__bases__[0].__subclasses__().__len__()) q[0];')
        with pytest.raises(ValueError):
            QASMParser().parse(qasm)

    def test_original_exploit_in_for_loop_bound_yields_no_ops(self):
        qasm = ('OPENQASM 3.0; qubit[1] q; '
                'for int i in [0:().__class__.__bases__[0].__subclasses__().__len__()] '
                '{ x q[0]; }')
        circ = QASMParser().parse(qasm)
        assert circ.ops == []

    @pytest.mark.parametrize('expr,expected', [
        ('pi', np.pi), ('pi/2', np.pi / 2), ('-pi/4', -np.pi / 4),
        ('pi/8', np.pi / 8), ('0.5', 0.5), ('-0.5', -0.5),
        ('sqrt(2)', np.sqrt(2)), ('cos(0.3)', np.cos(0.3)),
        ('sin(pi/4)*2', np.sin(np.pi / 4) * 2),
        ('2*pi/3', 2 * np.pi / 3), ('1+2*3', 7.0),
    ])
    def test_legitimate_expressions_unchanged(self, expr, expected):
        assert QASMParser()._eval_param(expr) == pytest.approx(expected)

    @pytest.mark.parametrize('malformed', [
        'pi * / 2', 'pi +', '(pi', 'pi 2', '**pi', 'pi // 2',
    ])
    def test_malformed_expression_raises_instead_of_silent_zero(self, malformed):
        # Found via an external code-review report, reproduced directly:
        # 'rx(pi * / 2) q[0];' used to parse successfully and silently
        # produce rx(0.0) -- a different, valid circuit, no signal a typo
        # happened. Now raises instead of hiding the mistake.
        with pytest.raises(ValueError):
            QASMParser()._eval_param(malformed)

    def test_malformed_expression_through_full_parse_raises(self):
        qasm = 'OPENQASM 2.0; include "qelib1.inc"; qreg q[1]; rx(pi * / 2) q[0];'
        with pytest.raises(ValueError):
            QASMParser().parse(qasm)

    def test_legitimate_for_loop_bounds_unchanged(self):
        qasm = 'qreg q[3]; int n = 3; for int i in [0:n-1] { x q[i]; }'
        circ = QASMParser().parse(qasm)
        assert [op['qubits'][0] for op in circ.ops] == [0, 1, 2]
