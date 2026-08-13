"""task-6: Mini Coding Agent.

A minimal local Claude Code clone — three-layer capability stack:

  Tools (atomic, stateless) → Skills (workflow + progressive disclosure)
      → Subagents (isolated context, independent max_steps)
  plus the main CodingAgent loop that orchestrates them.

Public entry points (referenced by ``eval/run.py``):

* ``src.mcp_server.list_tools`` — enumeration of MCP-exposed tools
* ``src.skill_loader.SkillLoader`` — progressive-disclosure skill catalogue
* ``src.agent.CodingAgent`` — main agentic loop returning a ``Trace`` dict
"""
__version__ = "0.1.0"
