msmodelslim quant --model_path /mnt/weight/Kimi_K2_5/ --save_path /mnt/weight/Kimi-K2_5-w4a8c8 --device npu --model_type Kimi-K2.5 --config_path /mnt/share/z00955625/msmodelslim/w4a8c8_config.yaml --trust_remote_code True

apiversion: multimodal_vlm_modelslim_v1
metadata:
  config_id: kimi_k2_6_w4a8c8
  score: 90
  verified_tags:
    Kimi-K2.6:
      - - vLLM_Ascend
        - Atlas_A2_Inference
      - - vLLM_Ascend
        - Atlas_A3_Inference
  label:
    w_bit: 4
    a_bit: 8
    is_sparse: False
    kv_cache: True  # 【核心修改】：开启 C8 (KV Cache 8-bit) 量化

default_w8a8: &default_w8a8
  act:
    scope: "per_tensor"
    dtype: "int8"
    symmetric: False
    method: "minmax"
  weight:
    scope: "per_channel"
    dtype: "int8"
    symmetric: True
    method: "minmax"

default_w8a8_dynamic: &default_w8a8_dynamic
  act:
    scope: "per_token"
    dtype: "int8"
    symmetric: True
    method: "minmax"
  weight:
    scope: "per_channel"
    dtype: "int8"
    symmetric: True
    method: "minmax"

default_w4a8_dynamic: &default_w4a8_dynamic
  act:
    scope: "per_token"
    dtype: "int8"
    symmetric: True
    method: "minmax"
  weight:
    scope: "per_channel"
    dtype: "int4"
    symmetric: True
    method: "ssz"

spec:
  process:
    # 1. 保持原有的 W4A8 权重和激活量化逻辑 (针对文本 LLM 部分)
    - type: "group"
      configs:
        - type: "linear_quant"
          qconfig: *default_w8a8
          include:
            - "*self_attn*"
          exclude:
            - "*kv_b_proj*"  # 保留：Kimi 使用了 MLA 架构，kv_b_proj 是 MLA 特有的下投影，不能量化
        - type: "linear_quant"
          qconfig: *default_w8a8_dynamic
          include:
            - "*mlp*"
          exclude:
            - "*mlp.experts.*"
        - type: "linear_quant"
          qconfig: *default_w4a8_dynamic
          include:
            - "*mlp.experts.*"

    # 2. 新增 fa3_quant，实现 C8 (KV Cache 8-bit) 量化
    - type: "fa3_quant"
      include:
        - "*"

  dataset: calibImages
  save:
    - type: "ascendv1_saver"
      part_file_size: 4
