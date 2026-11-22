# NeoCore's memory server (MCP over stdio), for MCP directories and container clients.
# Day-to-day use doesn't need this: `pipx install "neocore-memory[embed]"` then `neocore setup`.
FROM python:3.12-slim
RUN pip install --no-cache-dir "neocore-memory==0.3.11"
ENV NEOCORE_HOME=/data
VOLUME ["/data"]
ENTRYPOINT ["neocore", "mcp"]
