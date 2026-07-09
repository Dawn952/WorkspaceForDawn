curl -X POST http://192.168.13.153:8900/v1/completions   -H "Content-Type: application/json"     -d '{
        "model": "kimi25",
           "prompt": "The quick brown fox jumps over the lazy dog",
          "max_tokens": 160
         }'

curl -H "Accept: application/json" -H "Content-type: application/json" -X POST -d '{  
"model": "kimi_k26",  
"messages": [{  
"role": "user",  
"content": "The quick brown fox jumps over the lazy dog"  
}],  
"stream": false,  
"ignore_eos": false,  
"max_tokens": 200  
}' http://80.5.17.104:8089/v1/chat/completions
