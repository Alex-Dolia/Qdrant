# Qdrant Performance Optimizations

Tips and best practices for optimizing Qdrant performance.

## Collection Configuration

### HNSW Parameters

For better search performance, configure HNSW:

```python
from qdrant_client.models import VectorParams, Distance

collection_config = VectorParams(
    size=4096,  # Llama 3.1 embedding dimension
    distance=Distance.COSINE,
    hnsw_config={
        "m": 16,        # Number of connections
        "ef_construct": 200,  # Construction time accuracy
    }
)
```

### Search Parameters

Optimize query-time performance:

```python
search_params = SearchParams(
    hnsw_ef=64  # Query-time accuracy (lower = faster)
)
```

## Docker Configuration

### Resource Limits

Allocate sufficient resources:

```bash
docker run -p 6333:6333 \
  --memory="4g" \
  --cpus="2" \
  qdrant/qdrant
```

### Persistent Storage

Use volumes for better I/O:

```bash
docker run -p 6333:6333 \
  -v qdrant_storage:/qdrant/storage \
  qdrant/qdrant
```

## Indexing Optimization

### Batch Insertion

Insert documents in batches:

```python
# Good: Batch insertion
points = [point1, point2, ..., point100]
client.upsert(collection_name, points)

# Avoid: Single insertions
for point in points:
    client.upsert(collection_name, [point])
```

### Payload Indexing

Index frequently filtered fields:

```python
client.create_payload_index(
    collection_name="my_collection",
    field_name="file_name",
    field_schema="keyword"
)
```

## Query Optimization

### Limit Results

Only retrieve what you need:

```python
results = client.search(
    collection_name="my_collection",
    query_vector=vector,
    limit=5  # Don't retrieve more than needed
)
```

### Use Filters

Filter before search when possible:

```python
from qdrant_client.models import Filter, FieldCondition, MatchValue

filter = Filter(
    must=[
        FieldCondition(key="file_name", match=MatchValue(value="document.pdf"))
    ]
)
```

## Memory Management

### Collection Management

- Delete unused collections
- Archive old data
- Use separate collections for different projects

### Payload Size

- Keep payloads small
- Store large text separately
- Use references instead of full content

## Monitoring

### Check Collection Info

```python
collection_info = client.get_collection("my_collection")
print(collection_info)
```

### Monitor Performance

- Check query times
- Monitor memory usage
- Track collection sizes

## Best Practices

1. **Use appropriate HNSW parameters** for your use case
2. **Index frequently filtered fields** in payload
3. **Batch operations** when possible
4. **Monitor resource usage** regularly
5. **Clean up unused data** periodically

## References

- [Qdrant Documentation](https://qdrant.tech/documentation/)
- [Qdrant Performance Guide](https://qdrant.tech/documentation/guides/performance/)

