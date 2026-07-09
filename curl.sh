curl -X POST http://192.168.13.153:8900/v1/completions   -H "Content-Type: application/json"     -d '{
        "model": "kimi25",
           "prompt": "The quick brown fox jumps over the lazy dog",
          "max_tokens": 160
         }'
