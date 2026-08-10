# Dashboard Core — Engine

The real simulation engine for the dashboard: runs an actual
[`DenseSVSimulator`](simulator.md) circuit, not a mocked/placeholder
result, and is what Composer's UI and the MCP server's simulation tools
both call underneath.

::: dashboard_core.engine
