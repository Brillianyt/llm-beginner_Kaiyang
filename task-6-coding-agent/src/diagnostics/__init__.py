"""Diagnostics module — offline / debugging helpers.

NOT IMPORTED BY ``src/agent.py``.  ``src/agent.py`` must never reach into
the diagnostic surface; ``message.tool_calls`` from the vLLM server is
the canonical source of truth for tool calls.  See
``src/diagnostics/text_tool_parser.py`` for the architectural invariant.
"""
