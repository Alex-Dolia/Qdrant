# Qdrant Setup Guide

Setup guide for Qdrant vector database.

## What is Qdrant?

Qdrant is a vector similarity search engine. We use it to store and search document embeddings.

## Prerequisites

- Docker Desktop installed and running
- Port 6333 available

## Quick Start

### Using the Script (Windows)

```bash
start_qdrant.bat
```

### Using Docker Directly

```bash
docker run -p 6333:6333 qdrant/qdrant
```

## Verify Qdrant is Running

1. Open browser: http://localhost:6333/dashboard
2. You should see the Qdrant dashboard

## Configuration

### Default Settings

- **Port:** 6333 (HTTP API)
- **Port:** 6334 (gRPC API)
- **Storage:** In-memory (data lost on restart)

### Persistent Storage

To persist data:

```bash
docker run -p 6333:6333 -v $(pwd)/qdrant_storage:/qdrant/storage qdrant/qdrant
```

Windows:
```cmd
docker run -p 6333:6333 -v %cd%\qdrant_storage:/qdrant/storage qdrant/qdrant
```

## Environment Variables

The app uses these defaults:
- `QDRANT_URL=http://localhost:6333`
- `QDRANT_API_KEY=` (optional, not needed for local)

## Troubleshooting

### Docker not running
- Start Docker Desktop
- Wait for it to fully start
- Verify: `docker ps`

### Port 6333 already in use
- Stop other Qdrant instances: `docker ps` then `docker stop <container_id>`
- Or change port: `docker run -p 6334:6333 qdrant/qdrant`
- Update `QDRANT_URL` environment variable

### Connection refused
- Check Docker Desktop is running
- Verify container is running: `docker ps`
- Check firewall settings

### Data not persisting
- Use volume mount (see Persistent Storage above)
- Check volume permissions

## Performance Optimization

See `documents/09_qdrant_performance_optimizations.md` for performance tips.

## Next Steps

- See `documents/05_running_the_app.md` for using the app
- See `documents/09_qdrant_performance_optimizations.md` for optimization

