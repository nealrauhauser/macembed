curl -s http://127.0.0.1:8000/v1/embeddings \
  -H "Authorization: Bearer changeme" \
  -H "Content-Type: application/json" \
  -d '{"model":"all-mpnet-base-v2","input":["hello","world"]}' | jq '.data[].embedding | length'

