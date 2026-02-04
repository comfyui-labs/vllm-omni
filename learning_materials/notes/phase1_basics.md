# 阶段一：基础准备（1-2 周）

目标：建立 LLM 推理系统 + Transformer + Diffusion 的基础认知。

## 阅读顺序

1. vLLM 论文（PagedAttention 与推理系统设计）
   - `../papers/vllm_pagedattention_2309.06180.pdf`
2. vLLM 官方文档（概念与架构）
   - https://docs.vllm.ai/
3. Transformer 原始论文
   - `../papers/attention_is_all_you_need_1706.03762.pdf`
4. DDPM 论文
   - `../papers/ddpm_2006.11239.pdf`
5. DiT 论文
   - `../papers/dit_2212.09748.pdf`

## 建议笔记要点

- vLLM：Scheduler/Worker/ModelRunner 之间的关系、KV Cache/PagedAttention 的意义
- Transformer：自注意力、位置编码、encoder/decoder 的角色
- Diffusion：前向加噪/反向去噪过程、采样步骤与推理成本
- DiT：Transformer 如何替代 U-Net，patch/token 表达方式
