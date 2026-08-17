# Security Policy

## Supported Versions

Only the latest version published on [PyPI](https://pypi.org/project/dense-evolution/) receives fixes. There's no separate long-term-support branch — if you're on an older version, upgrade before reporting an issue that might already be fixed.

## Reporting a Vulnerability

**Do not open a public GitHub issue for security vulnerabilities.**

Instead, email the maintainer directly: **jtatopenn@libero.it**

Include:
- A description of the vulnerability and its potential impact
- Steps to reproduce (a minimal script/circuit if applicable)
- Which version(s) you've confirmed it on

This is a scientific-computing library (a quantum circuit simulator) rather than a network-facing service — the most relevant risk class is untrusted QASM/circuit input reaching code that shouldn't be reachable from it, or dependency vulnerabilities pulled in transitively. `dense_evolution/parser.py` evaluates QASM gate parameters and `for`-loop bounds through `_eval_ast_node`, an explicit AST node-type whitelist (no `eval()`/`exec()` anywhere in the codebase, confirmed via full-repo search) — this replaced a raw `eval()` call in **v8.1.19**, after a real code-execution vulnerability was found and fixed (attribute/dunder object-graph traversal bypassed the previous `{'__builtins__': {}}` restriction entirely; see the v8.1.19 changelog entry in `README.md`). If you find another way to make untrusted QASM/circuit input execute arbitrary code, that's exactly the kind of report this policy is for. Report anything that looks like it could do that, not just numerical correctness bugs (those belong in a regular issue instead).

You should expect an initial response within a few days. Since this is a single-maintainer project, please be patient — but security reports are prioritized over regular feature work.

Encrypted reports
-----------------
If your report contains sensitive exploit details (PoC that enables remote code execution, leaked secrets, or any data you prefer not to publish publicly), please request the maintainer's PGP public key by emailing jtatopenn@libero.it with a short note. Alternatively, use GitHub's private Security Advisories feature to submit an encrypted/private disclosure.

Disclosure timeline
-------------------
- Acknowledgement: within 3 business days when possible.
- Resolution: the goal is to provide a security fix or mitigation within 90 days of a confirmed report. If a fix requires more time, the maintainer will communicate timelines and may coordinate with the reporter on an embargo period.

If you need a faster (or more formal) channel for coordinated disclosure, include that request in your initial email and the maintainer will try to accommodate it.
