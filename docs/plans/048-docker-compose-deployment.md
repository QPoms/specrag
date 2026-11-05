# 048 Docker Compose Deployment

## Objective

Package the supported local SpecRAG topology behind one Docker Compose command
without collapsing the API, MCP research runtime, and browser client into one
process. Preserve the existing ownership rule: the API owns the Conversation
Worker, while MCP owns Research and Lifecycle Workers.

## Frozen decisions

- Use one multi-target root `Dockerfile`: a reusable Python backend target and
  an Nginx-hosted frontend target.
- Run three long-lived Compose services: `api`, `mcp`, and `web`.
- Reuse the same backend image for `api`, `mcp`, and an on-demand `cli` service.
- Publish only Web and API on host loopback. MCP is exposed only to the private
  Compose network.
- Keep MCP loopback-only by default outside containers. A new explicit CLI flag
  is required before it may bind a non-loopback address.
- Store `.local/` and the model cache in Docker named volumes. Bind the local
  `data/3gpp` corpus into runtime services read-only and into the on-demand CLI
  read-write for explicit corpus synchronization.
- Load model, LangSmith, and delegation credentials from the untracked `.env`;
  no secret enters an image layer or generated artifact.
- Keep corpus activation and administrator bootstrap explicit. They are run
  through `docker compose run --rm cli ...` before the long-lived services.
- Docker Compose is the supported container workflow. A single-container
  supervisor topology is not supported.

## Work packages

1. Add the multi-target Dockerfile, build context exclusions, Nginx reverse
   proxy configuration, and Compose topology.
2. Add the explicit MCP non-loopback opt-in while preserving the default
   security rejection and add tests for both paths.
3. Add static tests that parse the Compose file, check service commands,
   mounts, port exposure, health checks, and secret handling without requiring
   a Docker daemon.
4. Update `.env.example`, README, and current status with the one-command
   container workflow and its initialization steps.
5. Run the standard quality gate and frontend production build. Run Docker
   build/config/health validation when a Docker daemon is available; otherwise
   record that external validation gap explicitly.

## Acceptance criteria

- `docker compose up -d --build` describes the complete API/MCP/Web runtime.
- API reaches MCP through `http://mcp:8766/mcp`; MCP has no published host port.
- API and Web publish only to `127.0.0.1` by default.
- Persistent database/index and model-cache paths use named volumes.
- Runtime services mount the corpus read-only; the on-demand CLI is the only
  Compose service with a read-write corpus mount.
- The frontend reverse proxy supports `/api`, `/healthz`, and unbuffered SSE.
- MCP still rejects non-loopback binding unless the explicit opt-in is present.
- No `.env`, credential, local corpus, index, Run, or evaluation artifact is
  copied into an image.
- `python scripts/check.py` and `npm run build` pass.

## Verification limitation

The implementation workstation did not have the Docker CLI or daemon installed
when this plan was created. Static Compose tests, application tests, and the
frontend build remain mandatory; image build and live health verification must
be run on the first Docker-enabled machine before treating the container path
as production-validated.
