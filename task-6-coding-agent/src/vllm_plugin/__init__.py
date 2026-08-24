"""vLLM plugin package — Coder-specific tool-call parsers.

Currently exposes :class:`QwenCoderToolParser` for Qwen2.5-Coder-7B-Instruct
under the parser name ``qwen_coder_json``.  Loaded at vLLM server startup via::

    --tool-parser-plugin src/vllm_plugin/qwen_coder_tool_parser.py \\
    --tool-call-parser qwen_coder_json
"""
