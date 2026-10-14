"""Shared memory: one NeoCore memory for every AI on every machine.

``server`` keeps the memory and answers over HTTP (``neocore server serve``); ``client`` is what
a device's hooks and MCP server use instead of a local store once ``neocore connect`` wrote
``remote.json``; ``keys`` are the per-device API keys; ``tools`` are the memory tools an AI calls
on purpose; ``admin`` is the server operator's view (scopes, audit, export, forget).
"""
