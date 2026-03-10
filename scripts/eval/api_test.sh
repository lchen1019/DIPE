curl http://${LOCAL_IP}:7777/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer sk-abc123" \
  -d '{
    "model": "api_qwen2_5_siglip_3b_instruct",
    "messages": [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": [
          {"type": "image_url", "image_url": {"url": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg"}},
          {"type": "text", "text": "Describe this image."}
      ]}
    ],
    "max_tokens": 1024,
    "temperature": 0.1,
    "presence_penalty": 2.0,
    "top_k": 40,
    "top_p": 1.0
  }'
