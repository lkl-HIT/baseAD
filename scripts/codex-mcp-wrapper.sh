#!/usr/bin/env bash
set -euo pipefail
cd /root/FoundAD
exec /root/.nvm/versions/node/v24.16.0/bin/codex mcp-server 2>>/tmp/codex-mcp.stderr.log