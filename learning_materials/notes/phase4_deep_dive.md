# 阶段四：深入模块（2-4 周）

目标：掌握关键组件实现，能读懂核心模块细节。

## Diffusion 深入顺序

1. `vllm_omni/diffusion/data.py`
2. `vllm_omni/diffusion/scheduler.py`
3. `vllm_omni/diffusion/worker/`
4. `vllm_omni/diffusion/models/`（任选 1-2 个模型实现深入）
5. `vllm_omni/diffusion/attention/`
6. `vllm_omni/diffusion/cache/`

## 分布式与连接器

- `vllm_omni/distributed/omni_connectors/`
- 重点理解 shared_memory / mooncake / yuanrong

## 建议产出

- 画一张“Diffusion 推理时序图”
- 记录每个阶段进程/线程模型
- 对比 AR 与 Diffusion 的调度差异点
