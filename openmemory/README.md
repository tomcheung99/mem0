# OpenMemory

OpenMemory is your personal memory layer for LLMs - private, portable, and open-source. Your memories live locally, giving you complete control over your data. Build AI applications with personalized memories while keeping your data secure.

![OpenMemory](https://github.com/user-attachments/assets/3c701757-ad82-4afa-bfbe-e049c2b4320b)

## Easy Setup

### Prerequisites
- Docker
- OpenAI API Key

You can quickly run OpenMemory by running the following command:

```bash
curl -sL https://raw.githubusercontent.com/mem0ai/mem0/main/openmemory/run.sh | bash
```

You should set the `OPENAI_API_KEY` as a global environment variable:

```bash
export OPENAI_API_KEY=your_api_key
```

You can also set the `OPENAI_API_KEY` as a parameter to the script:

```bash
curl -sL https://raw.githubusercontent.com/mem0ai/mem0/main/openmemory/run.sh | OPENAI_API_KEY=your_api_key bash
```

## Prerequisites

- Docker and Docker Compose
- Python 3.9+ (for backend development)
- Node.js (for frontend development)
- OpenAI API Key (required for LLM interactions, run `cp api/.env.example api/.env` then change **OPENAI_API_KEY** to yours)

## Quickstart

### 1. Set Up Environment Variables

Before running the project, you need to configure environment variables for both the API and the UI.

You can do this in one of the following ways:

- **Manually**:  
  Create a `.env` file in each of the following directories:
  - `/api/.env`
  - `/ui/.env`

- **Using `.env.example` files**:  
  Copy and rename the example files:

  ```bash
  cp api/.env.example api/.env
  cp ui/.env.example ui/.env
  ```

 - **Using Makefile** (if supported):  
    Run:
  
   ```bash
   make env
   ```
- #### Example `/api/.env`

```env
OPENAI_API_KEY=sk-xxx
USER=<user-id> # The User Id you want to associate the memories with 
ALLOWED_ORIGINS=http://localhost:3000 # Comma-separated origins. Use * only for local development.
MCP_API_KEY=change-me # Optional. When set, MCP endpoints require Bearer auth.
MCP_RATE_LIMIT=100/minute
```
- #### Example `/ui/.env`

```env
NEXT_PUBLIC_API_URL=http://localhost:8765
NEXT_PUBLIC_USER_ID=<user-id> # Same as the user id for environment variable in api
```

### 2. Build and Run the Project
You can run the project using the following two commands:
```bash
make build # builds the mcp server and ui
make up  # runs openmemory mcp server and ui
```

After running these commands, you will have:
- OpenMemory MCP server running at: http://localhost:8765 (API documentation available at http://localhost:8765/docs)
- OpenMemory UI running at: http://localhost:3000

#### UI not working on `localhost:3000`?

If the UI does not start properly on [http://localhost:3000](http://localhost:3000), try running it manually:

```bash
cd ui
pnpm install
pnpm dev
```

### MCP Client Setup

Use the following one step command to configure OpenMemory Local MCP to a client. The general command format is as follows:

```bash
npx @openmemory/install local http://localhost:8765/mcp/<client-name>/sse/<user-id> --client <client-name>
```

Replace `<client-name>` with the desired client name and `<user-id>` with the value specified in your environment variables.

### Securing Remote MCP Access

OpenMemory already isolates memory by user ID and app-level permissions. For an internet-exposed deployment, you should also enable transport-level controls:

- Set `MCP_API_KEY` to require authentication on MCP SSE and message endpoints.
- Set `ALLOWED_ORIGINS` to a comma-separated allowlist instead of `*`.
- Tune `MCP_RATE_LIMIT` if you expect higher per-user throughput.

Bearer auth is the preferred mode:

```http
Authorization: Bearer <MCP_API_KEY>
```

For clients that cannot set an Authorization header on SSE, the server also accepts `X-API-Key` or an `api_key` query parameter.

### Railway Deployment

An example Railway service manifest is available in [openmemory/railway.yaml](railway.yaml). It is aligned with the variables that the current backend actually reads:

- `DATABASE_URL` for the relational metadata database.
- `PG_HOST`, `PG_PORT`, `PG_DB`, `PG_USER`, `PG_PASSWORD` for pgvector.
- `OPENAI_API_KEY`, `USER`, `MCP_API_KEY`, `ALLOWED_ORIGINS`, `MCP_RATE_LIMIT` for application runtime.

The service also exposes a `/health` endpoint for Railway health checks.

### MCP Client Examples

VANE:

```json
{
  "mcpServers": {
    "mem0-memory": {
      "command": "npx",
      "args": [
        "@openmemory/install",
        "remote",
        "https://your-app.railway.app/mcp/vane/sse/${USER_ID}",
        "--client",
        "vane"
      ],
      "env": {
        "MCP_API_KEY": "${MEM0_MCP_API_KEY}"
      }
    }
  }
}
```

OpenClaw:

```json
{
  "plugins": {
    "entries": {
      "openclaw-mem0": {
        "enabled": true,
        "config": {
          "mode": "open-source",
          "userId": "${USER_ID}",
          "apiKey": "${MEM0_MCP_API_KEY}",
          "baseUrl": "https://your-app.railway.app",
          "autoRecall": true,
          "autoCapture": true,
          "topK": 5
        }
      }
    }
  }
}
```


## Project Structure

- `api/` - Backend APIs + MCP server
- `ui/` - Frontend React application

## Contributing

We are a team of developers passionate about the future of AI and open-source software. With years of experience in both fields, we believe in the power of community-driven development and are excited to build tools that make AI more accessible and personalized.

We welcome all forms of contributions:
- Bug reports and feature requests
- Documentation improvements
- Code contributions
- Testing and feedback
- Community support

How to contribute:

1. Fork the repository
2. Create your feature branch (`git checkout -b openmemory/feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin openmemory/feature/amazing-feature`)
5. Open a Pull Request

Join us in building the future of AI memory management! Your contributions help make OpenMemory better for everyone.
