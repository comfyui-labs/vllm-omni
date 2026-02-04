# 阶段三：代码阅读（2-3 周）

目标：理解 vLLM-Omni 的整体架构与调用链。

## 建议阅读顺序（从入口到核心）

1. `vllm_omni/__init__.py`
2. `vllm_omni/entrypoints/omni.py`
3. `docs/design/architecture_overview.md`
4. `docs/design/module/ar_module.md`
5. `docs/design/module/dit_module.md`
6. `vllm_omni/diffusion/diffusion_engine.py`
7. `vllm_omni/diffusion/scheduler.py`
8. `vllm_omni/config/model.py`
9. `vllm_omni/distributed/omni_connectors/`
10. `vllm_omni/outputs.py`

## 建议记录的问题

- `Omni` 如何组织多个 `OmniStage`
- stage 之间如何通过 `OmniConnector` 传递 payload
- Diffusion 的请求、调度、worker 进程模型
- AR 与 Diffusion 在输入/输出处理上的差异
