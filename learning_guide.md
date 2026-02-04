# vLLM-Omni 代码深度分析与学习指南

> 本文档为 vLLM-Omni 项目的代码分析和小白学习路线指南。

## 目录

- [项目定位](#项目定位)
- [核心架构概览](#核心架构概览)
- [五大核心模块详解](#五大核心模块详解)
  - [入口点模块](#1️⃣-入口点模块-entrypoints)
  - [Diffusion 模块](#2️⃣-diffusion-模块)
  - [AR 模块](#3️⃣-ar-autoregressive-模块)
  - [分布式连接器](#4️⃣-分布式连接器-omniconnector)
  - [配置系统](#5️⃣-配置系统)
- [小白学习路线建议](#小白学习路线建议)
- [项目亮点总结](#项目亮点总结)

---

## 项目定位

**vLLM-Omni** 是 vLLM 的扩展框架，专门用于支持全模态（Omni-modality）模型的推理和服务。它将 vLLM 从传统的文本自回归生成扩展到支持：

- **多模态输入/输出**：文本、图像、视频、音频
- **非自回归架构**：Diffusion Transformer (DiT) 等并行生成模型
- **异构输出**：从传统文本到多模态输出

### 支持的模型类型

根据当前流行开源模型的分析，大多数全模态模型都是 AR + DiT 的组合：

| 类型 | 示例 | 描述 |
|------|------|------|
| **DiT 为主，AR 为文本编码器** | Qwen-Image | 强大的图像生成基础模型 |
| **AR 为主，DiT 为多模态生成器** | BAGEL | 统一的多模态理解和生成模型 |
| **AR + DiT 混合** | Qwen-Omni | 端到端全模态 LLM |

---

## 核心架构概览

### 目录结构

```
vllm-omni/
├── vllm_omni/                    # 核心源码
│   ├── __init__.py              # 包入口
│   ├── config/                   # 配置模块
│   │   ├── __init__.py
│   │   ├── lora.py              # LoRA 配置
│   │   └── model.py             # OmniModelConfig
│   ├── core/                     # 调度器核心
│   │   └── sched/               # 调度器实现
│   ├── diffusion/                # Diffusion 模块 (核心!)
│   │   ├── attention/           # 注意力机制
│   │   ├── cache/               # 缓存加速
│   │   ├── distributed/         # 分布式
│   │   ├── models/              # 模型实现
│   │   ├── worker/              # Worker
│   │   ├── diffusion_engine.py  # 主引擎
│   │   └── scheduler.py         # 调度器
│   ├── distributed/              # 分布式通信
│   │   ├── omni_connectors/     # 连接器实现
│   │   └── ray_utils/           # Ray 工具
│   ├── engine/                   # 引擎层
│   │   ├── arg_utils.py
│   │   ├── input_processor.py
│   │   └── output_processor.py
│   ├── entrypoints/              # 入口点 (API 层)
│   │   ├── omni.py              # 主入口 Omni 类
│   │   ├── async_omni.py        # 异步入口
│   │   ├── cli/                 # 命令行工具
│   │   └── openai/              # OpenAI 兼容 API
│   ├── inputs/                   # 输入处理
│   │   ├── data.py              # 数据类型定义
│   │   ├── parse.py             # 解析
│   │   └── preprocess.py        # 预处理
│   ├── model_executor/           # 模型执行器
│   │   ├── models/              # 模型实现
│   │   └── stage_configs/       # 阶段配置 YAML
│   ├── platforms/                # 多平台支持
│   │   ├── cuda/                # CUDA
│   │   ├── npu/                 # NPU (华为)
│   │   ├── rocm/                # ROCm (AMD)
│   │   └── xpu/                 # XPU (Intel)
│   ├── worker/                   # Worker 实现
│   ├── outputs.py               # 输出数据结构
│   └── request.py               # 请求数据结构
├── examples/                     # 示例代码
│   ├── offline_inference/       # 离线推理示例
│   └── online_serving/          # 在线服务示例
├── docs/                         # 设计文档
│   ├── design/                  # 设计文档
│   ├── getting_started/         # 入门指南
│   └── user_guide/              # 用户指南
├── tests/                        # 测试用例
├── benchmarks/                   # 基准测试
└── pyproject.toml               # 项目配置
```

### 主要组件关系图

```
┌─────────────────────────────────────────────────────────────────┐
│                         用户请求                                 │
└───────────────────────────┬─────────────────────────────────────┘
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Omni / AsyncOmni                             │
│                    (entrypoints/omni.py)                        │
│              统一入口，管道编排，请求调度                          │
└───────────────────────────┬─────────────────────────────────────┘
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                      OmniStage (多个)                            │
│              每个阶段可以是 AR 或 Diffusion                       │
├─────────────────────────────────────────────────────────────────┤
│ ┌─────────────────┐    ┌─────────────────┐    ┌───────────────┐ │
│ │   Stage 0 (AR)  │───▶│  Stage 1 (AR)   │───▶│Stage 2 (Conv) │ │
│ │    Thinker      │    │     Talker      │    │   Code2wav    │ │
│ └─────────────────┘    └─────────────────┘    └───────────────┘ │
│         │                      │                      │         │
│         └──────────────────────┴──────────────────────┘         │
│                    OmniConnector (数据传输)                       │
└─────────────────────────────────────────────────────────────────┘
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                    OmniRequestOutput                            │
│              包含文本、图像、音频等多模态输出                       │
└─────────────────────────────────────────────────────────────────┘
```

---

## 五大核心模块详解

### 1️⃣ 入口点模块 (entrypoints)

**位置**: `vllm_omni/entrypoints/`

**核心类**:

```python
# vllm_omni/entrypoints/omni.py

class OmniBase:
    """Base class for serving Omni models.

    Args:
        model: Model name or path to load.
        **kwargs: Arbitrary keyword arguments.
            - stage_configs_path: 阶段配置 YAML 路径
            - log_stats: 是否启用统计日志
            - stage_init_timeout: 阶段初始化超时时间
            - shm_threshold_bytes: 共享内存阈值
            - worker_backend: Worker 后端 ("multi_process" 或 "ray")
            - ray_address: Ray 集群地址
            - batch_timeout: 批处理超时时间
            - init_timeout: 初始化超时时间
    """

class Omni(OmniBase):
    """统一入口，支持 LLM 和 Diffusion 模型"""
    
    def generate(
        self,
        prompts: OmniPromptType | Sequence[OmniPromptType],
        sampling_params_list: OmniSamplingParams | Sequence[OmniSamplingParams] | None = None,
        *,
        py_generator: bool = False,
        use_tqdm: bool | Callable[..., tqdm] = True,
    ) -> Generator[OmniRequestOutput, None, None] | list[OmniRequestOutput]:
        """生成输出"""
        ...
```

**核心职责**:
- 统一的推理入口（离线批量推理和在线服务）
- 多阶段管道编排（如 Thinker → Talker → Code2wav）
- 请求调度和结果收集
- 资源管理和清理

**关键方法**:

| 方法 | 描述 |
|------|------|
| `__init__()` | 初始化模型和阶段 |
| `generate()` | 执行推理生成 |
| `_initialize_stages()` | 初始化所有阶段 |
| `_start_stages()` | 启动所有阶段进程 |
| `start_profile()` / `stop_profile()` | 性能分析 |
| `close()` | 清理资源 |

---

### 2️⃣ Diffusion 模块

**位置**: `vllm_omni/diffusion/`

这是 vLLM-Omni 最核心的创新模块，实现了非自回归的 Diffusion 推理。

#### 目录结构

```
vllm_omni/diffusion/
├── __init__.py
├── attention/                    # 注意力机制
│   ├── backends/                # 后端实现
│   │   ├── abstract.py         # 抽象基类
│   │   ├── flash_attn.py       # FlashAttention
│   │   ├── sdpa.py             # PyTorch SDPA
│   │   ├── sage_attn.py        # SageAttention
│   │   └── registry.py         # 后端注册
│   ├── parallel/                # 并行注意力
│   │   ├── ring.py             # Ring Attention
│   │   └── ulysses.py          # Ulysses SP
│   ├── layer.py                 # Attention 层
│   └── selector.py              # 后端选择器
├── cache/                        # 缓存加速
│   ├── base.py                  # 缓存基类
│   ├── cache_dit_backend.py    # cache-dit
│   ├── teacache/                # TeaCache
│   └── selector.py              # 缓存选择器
├── distributed/                  # 分布式
│   ├── parallel_state.py       # 并行状态管理
│   ├── cfg_parallel.py         # CFG 并行
│   ├── sp_plan.py              # 序列并行计划
│   └── comm.py                  # 通信
├── models/                       # 模型实现
│   ├── interface.py            # 模型接口
│   ├── qwen_image/             # Qwen-Image
│   ├── flux/                   # FLUX
│   ├── flux2_klein/            # FLUX2-Klein
│   ├── glm_image/              # GLM-Image
│   ├── wan2_2/                 # Wan2.2
│   ├── z_image/                # Z-Image
│   ├── bagel/                  # BAGEL
│   ├── stable_audio/           # Stable Audio
│   └── schedulers/             # 调度器
├── executor/                     # 执行器
│   ├── abstract.py
│   └── multiproc_executor.py
├── worker/                       # Worker
│   ├── diffusion_worker.py
│   └── diffusion_model_runner.py
├── layers/                       # 自定义层
│   ├── adalayernorm.py
│   ├── rope.py
│   └── custom_op.py
├── lora/                         # LoRA 支持
├── hooks/                        # 钩子系统
├── profiler/                     # 性能分析
├── model_loader/                 # 模型加载
├── diffusion_engine.py           # 主引擎
├── scheduler.py                  # 调度器
├── request.py                    # 请求定义
├── data.py                       # 数据结构
├── compile.py                    # 编译优化
├── forward_context.py            # 前向上下文
├── offload.py                    # 内存卸载
├── envs.py                       # 环境变量
├── registry.py                   # 模型注册
└── utils/                        # 工具函数
```

#### 核心组件

**1. DiffusionEngine (主引擎)**

```python
# vllm_omni/diffusion/diffusion_engine.py

class DiffusionEngine:
    """Diffusion 推理引擎，管理 Worker 进程和执行流程"""
    
    def __init__(self, od_config: OmniDiffusionConfig):
        self.od_config = od_config
        self.post_process_func = get_diffusion_post_process_func(od_config)
        self.pre_process_func = get_diffusion_pre_process_func(od_config)
        self._processes: list[mp.Process] = []
        self._make_client()
    
    def step(self, requests: list[OmniDiffusionRequest]):
        """执行一步推理"""
        # 1. 预处理请求
        requests = self.pre_process_func(requests)
        # 2. 发送到调度器并等待响应
        output = self.add_req_and_wait_for_response(requests)
        # 3. 后处理结果
        result = self.post_process_func(output.output)
        return result
```

**2. Scheduler (调度器)**

```python
# vllm_omni/diffusion/scheduler.py

class Scheduler:
    """单例调度器，协调所有 Worker"""
    
    _instance = None
    
    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def add_req(self, requests: list[OmniDiffusionRequest]) -> DiffusionOutput:
        """广播请求到所有 Worker"""
        self.mq.enqueue(requests)
        output = self.result_mq.dequeue()
        return output
```

**3. 注意力后端**

支持多种注意力实现：

| 后端 | 描述 | 适用场景 |
|------|------|----------|
| `FlashAttention` | 高性能 CUDA 内核 | NVIDIA GPU |
| `SDPA` | PyTorch 内置 | 跨平台默认 |
| `SageAttention` | 稀疏注意力 | 长序列 |
| `AscendAttention` | NPU 优化 | 华为昇腾 |

**4. 并行策略**

```python
# 初始化并行组
def initialize_model_parallel(
    data_parallel_size: int = 1,      # 数据并行
    cfg_parallel_size: int = 1,       # CFG 并行
    sequence_parallel_size: int = 1,  # 序列并行 (ulysses_degree × ring_degree)
    tensor_parallel_size: int = 1,    # 张量并行
    pipeline_parallel_size: int = 1,  # 流水线并行
):
    ...
```

**5. 缓存加速**

| 后端 | 特性 |
|------|------|
| `TeaCache` | 基于时间步嵌入相似度的缓存 |
| `cache-dit` | DBCache + SCM + TaylorSeer |

---

### 3️⃣ AR (AutoRegressive) 模块

**位置**: 分布在 `vllm_omni/core/`, `vllm_omni/worker/`, `vllm_omni/model_executor/`

AR 模块通过继承扩展 vLLM 的核心组件：

#### 继承层次

```
┌─────────────────────────────────────────────────────────────┐
│                        Scheduler                             │
├─────────────────────────────────────────────────────────────┤
│ vLLM Scheduler ──▶ OmniARScheduler                          │
│                 └─▶ OmniGenerationScheduler                  │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                         Worker                               │
├─────────────────────────────────────────────────────────────┤
│ GPUWorker ──▶ GPUARWorker                                   │
│           └─▶ GPUGenerationWorker                            │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                       ModelRunner                            │
├─────────────────────────────────────────────────────────────┤
│ GPUModelRunner ──▶ OmniGPUModelRunner ──▶ GPUARModelRunner  │
│                                       └─▶ GPUGenerationModelRunner │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                    Input/Output Processor                    │
├─────────────────────────────────────────────────────────────┤
│ InputProcessor ──▶ OmniInputProcessor                       │
│ OutputProcessor ──▶ MultimodalOutputProcessor               │
└─────────────────────────────────────────────────────────────┘
```

#### 主要扩展功能

| 功能 | 描述 |
|------|------|
| **Payload 支持** | 序列化 prompt embeddings 和附加信息，支持阶段间传递 |
| **多模态处理** | 扩展的输入/输出处理器支持图像、音频等 |
| **Hidden State 暴露** | AR 模型运行器通过 `pooler_output` 暴露 hidden states |
| **生成调度器** | 针对非自回归架构的快速调度路径 |

#### 请求流程

```
OmniInputProcessor
       │
       ▼ OmniEngineCoreRequest (带 payload)
OmniARScheduler
       │
       ▼ schedule: OmniNewRequestData
GPUARWorker
       │
       ▼ SchedulerOutput
GPUARModelRunner
       │
       ▼ execute_model
Model Forward Pass
       │
       ▼ hidden_states, logits
GPUARModelRunner
       │
       ▼ sample_tokens: OmniModelRunnerOutput
OmniARScheduler
       │
       ▼ update_from_output
MultimodalOutputProcessor
       │
       ▼ RequestOutput
Client/Downstream Stage
```

---

### 4️⃣ 分布式连接器 (OmniConnector)

**位置**: `vllm_omni/distributed/omni_connectors/`

实现阶段间的数据传输。

#### 核心接口

```python
class OmniConnectorBase(ABC):
    @abstractmethod
    def put(
        self, 
        from_stage: str, 
        to_stage: str, 
        put_key: str, 
        data: Any
    ) -> tuple[bool, int, Optional[dict]]:
        """
        存储数据
        Returns: (success, serialized_size, metadata)
        """
        pass

    @abstractmethod
    def get(
        self, 
        from_stage: str, 
        to_stage: str, 
        get_key: str, 
        metadata: Optional[dict] = None
    ) -> Optional[tuple[Any, int]]:
        """
        获取数据
        Returns: (object, serialized_size)
        """
        pass
```

#### 支持的连接器

| 连接器 | 使用场景 | 说明 |
|--------|----------|------|
| `SharedMemoryConnector` | 单节点 | 默认，无需配置 |
| `MooncakeConnector` | 多节点 | 需要 Mooncake Master |
| `YuanrongConnector` | 多节点 | 需要 Yuanrong Datasystem |

#### 配置示例

```yaml
runtime:
  connectors:
    connector_of_shared_memory:
      name: SharedMemoryConnector
      extra:
        shm_threshold_bytes: 65536

stage_args:
  - stage_id: 0
    output_connectors:
      to_stage_1: connector_of_shared_memory

  - stage_id: 1
    input_connectors:
      from_stage_0: connector_of_shared_memory
```

---

### 5️⃣ 配置系统

**位置**: `vllm_omni/config/`, `vllm_omni/model_executor/stage_configs/`

#### OmniModelConfig

```python
# vllm_omni/config/model.py

@dataclass
class OmniModelConfig(ModelConfig):
    """Omni 模型配置，扩展 vLLM ModelConfig"""
    
    stage_id: int = 0                    # 阶段 ID
    async_chunk: bool = False            # 异步分块
    model_stage: str = "thinker"         # 阶段类型
    model_arch: str = "Qwen2_5OmniForConditionalGeneration"  # 架构
    engine_output_type: str | None = None  # 输出类型
    hf_config_name: str | None = None    # HF 配置名
    stage_connector_config: dict = field(
        default_factory=lambda: {
            "name": "SharedMemoryConnector",
            "extra": {},
        }
    )
    omni_kv_config: dict | None = None   # KV 配置
```

#### 阶段配置 YAML 示例

```yaml
# Qwen3-Omni 三阶段配置示例
stage_args:
  - stage_id: 0
    stage_type: "ar"
    final_output: true
    final_output_type: "text"
    runtime:
      process: true
      devices: "0"
      max_batch_size: 1
    engine_args:
      model_stage: "thinker"
      model_arch: "Qwen3OmniThinkerForConditionalGeneration"
      hf_config_name: "thinker_config"
      max_model_len: 32768
      trust_remote_code: true
      limit_mm_per_prompt:
        image: 1
        video: 1
        audio: 1

  - stage_id: 1
    stage_type: "ar"
    runtime:
      process: true
      devices: "0"
    engine_args:
      model_stage: "talker"
      model_arch: "Qwen3OmniTalkerForConditionalGeneration"
      hf_config_name: "talker_config"

  - stage_id: 2
    stage_type: "generation"
    final_output: true
    final_output_type: "audio"
    runtime:
      process: true
      devices: "0"
    engine_args:
      model_stage: "code2wav"
      model_arch: "Qwen3OmniCode2WavForConditionalGeneration"
```

---

## 小白学习路线建议

### 📖 阶段一：基础准备（1-2 周）

**学习目标**：理解 vLLM 和大模型推理基础

#### 1. 先学习 vLLM 基础

```bash
# 阅读 vLLM 官方文档
https://docs.vllm.ai
```

**重点理解**：
- PagedAttention 和 KV Cache 机制
- vLLM 的架构设计（Scheduler、Worker、ModelRunner）
- 请求调度和批处理

#### 2. 理解 Transformer 和 Diffusion

**Transformer**:
- 自注意力机制
- 编码器-解码器架构
- 自回归生成

**Diffusion**:
- DDPM (Denoising Diffusion Probabilistic Models)
- Score Matching
- DiT (Diffusion Transformer) 架构

#### 3. 建议阅读资料

| 资料 | 描述 |
|------|------|
| 《Attention Is All You Need》 | Transformer 原始论文 |
| 《Denoising Diffusion Probabilistic Models》 | DDPM 论文 |
| 《Scalable Diffusion Models with Transformers》 | DiT 论文 |
| vLLM 技术博客 | 理解 PagedAttention |

---

### 📖 阶段二：快速上手（1 周）

**学习目标**：能够运行示例代码

#### 1. 安装环境

```bash
# 创建虚拟环境
uv venv --python 3.12 --seed
source .venv/bin/activate

# 安装 vLLM
uv pip install vllm==0.14.0 --torch-backend=auto

# 克隆并安装 vLLM-Omni
git clone https://github.com/vllm-project/vllm-omni.git
cd vllm-omni
uv pip install -e .
```

#### 2. 运行最简单的示例

**文生图 (Text-to-Image)**:

```python
from vllm_omni.entrypoints.omni import Omni

if __name__ == "__main__":
    # 创建 Omni 实例
    omni = Omni(model="Tongyi-MAI/Z-Image-Turbo")
    
    # 生成图片
    prompt = "a cup of coffee on the table"
    outputs = omni.generate(prompt)
    
    # 保存结果
    images = outputs[0].request_output[0].images
    images[0].save("coffee.png")
    print("Image saved to coffee.png")
```

**多模态对话 (Qwen3-Omni)**:

```python
from vllm import SamplingParams
from vllm_omni.entrypoints.omni import Omni

omni = Omni(model="Qwen/Qwen3-Omni-30B-A3B-Instruct")

# 准备输入
prompt = """<|im_start|>system
You are Qwen, a helpful assistant.<|im_end|>
<|im_start|>user
Hello, who are you?<|im_end|>
<|im_start|>assistant
"""

inputs = {"prompt": prompt}

# 采样参数
sampling_params = [
    SamplingParams(temperature=0.9, max_tokens=512),  # Thinker
    SamplingParams(temperature=0.9, max_tokens=4096), # Talker
    SamplingParams(temperature=0.0, max_tokens=65536), # Code2wav
]

outputs = omni.generate(inputs, sampling_params)
```

#### 3. 探索示例目录

```
examples/offline_inference/
├── text_to_image/      # 文生图 (入门推荐)
├── text_to_video/      # 文生视频
├── text_to_audio/      # 文生音频
├── image_to_image/     # 图像编辑
├── image_to_video/     # 图生视频
├── qwen2_5_omni/       # Qwen2.5-Omni
├── qwen3_omni/         # Qwen3-Omni (推荐)
├── qwen3_tts/          # Qwen3 TTS
├── bagel/              # BAGEL
└── lora_inference/     # LoRA 推理
```

---

### 📖 阶段三：代码阅读（2-3 周）

**学习目标**：理解核心架构

#### 推荐阅读顺序

| 优先级 | 文件/模块 | 目的 |
|--------|-----------|------|
| ⭐⭐⭐ | `vllm_omni/__init__.py` | 理解导出的核心类 |
| ⭐⭐⭐ | `vllm_omni/entrypoints/omni.py` | 理解主入口和管道编排 |
| ⭐⭐⭐ | `docs/design/architecture_overview.md` | 架构设计文档 |
| ⭐⭐ | `docs/design/module/ar_module.md` | AR 模块设计 |
| ⭐⭐ | `docs/design/module/dit_module.md` | Diffusion 模块设计 |
| ⭐⭐ | `vllm_omni/diffusion/diffusion_engine.py` | Diffusion 引擎 |
| ⭐⭐ | `vllm_omni/diffusion/scheduler.py` | Diffusion 调度器 |
| ⭐ | `vllm_omni/config/model.py` | 配置类定义 |
| ⭐ | `vllm_omni/distributed/omni_connectors/` | 分布式连接器 |
| ⭐ | `vllm_omni/outputs.py` | 输出数据结构 |

#### 关键代码路径

```
请求进来
    │
    ▼
Omni.__init__()
    │ 加载模型配置
    │ 创建 OmniStage 列表
    │ 启动 Worker 进程
    ▼
Omni.generate()
    │ 验证输入和采样参数
    │ 生成 request_id
    ▼
_run_generation()
    │ 将请求放入 stage-0 队列
    ▼
while completed < total:
    │ 轮询各阶段输出队列
    │
    ├── stage.try_collect()
    │       │ 获取该阶段的输出
    │       ▼
    │   if final_output:
    │       yield OmniRequestOutput
    │
    └── 转发到下一阶段
            │ process_engine_inputs()
            │ connector.put() / stage.submit()
            ▼
        下一阶段处理
```

#### 调试技巧

```python
# 启用详细日志
import logging
logging.basicConfig(level=logging.DEBUG)

# 或者设置环境变量
import os
os.environ["VLLM_LOGGING_LEVEL"] = "DEBUG"
```

---

### 📖 阶段四：深入模块（2-4 周）

**学习目标**：掌握关键组件实现

#### 1. Diffusion 模块深入

**学习顺序**：

```
1. diffusion/data.py              # 数据结构定义
   - OmniDiffusionConfig
   - OmniDiffusionRequest
   - DiffusionParallelConfig

2. diffusion/scheduler.py         # 调度器实现
   - Scheduler 单例模式
   - MessageQueue 通信

3. diffusion/worker/              # Worker 实现
   - diffusion_worker.py          # 进程入口
   - diffusion_model_runner.py    # 模型执行

4. diffusion/models/qwen_image/   # 具体模型实现
   - pipeline.py                  # 管道定义
   - transformer.py               # Transformer 实现

5. diffusion/attention/           # 注意力机制
   - backends/                    # 各种后端
   - parallel/                    # 并行策略

6. diffusion/cache/               # 缓存加速
   - teacache/                    # TeaCache
   - cache_dit_backend.py         # cache-dit
```

#### 2. 理解多阶段管道

查看 `model_executor/stage_configs/` 下的 YAML 配置：

```bash
ls vllm_omni/model_executor/stage_configs/
# qwen2_5_omni.yaml
# qwen3_omni.yaml
# bagel.yaml
# ...
```

**理解数据流**：

```
Qwen3-Omni 三阶段流程:

用户输入 (文本/图像/视频/音频)
         │
         ▼
    ┌─────────┐
    │ Thinker │  Stage 0 (AR)
    │  思考器  │  输出: 文本 tokens + hidden states
    └────┬────┘
         │ OmniConnector
         ▼
    ┌─────────┐
    │ Talker  │  Stage 1 (AR)
    │  说话器  │  输出: 音频 codec tokens
    └────┬────┘
         │ OmniConnector
         ▼
    ┌─────────┐
    │Code2wav │  Stage 2 (Generation)
    │ 波形生成 │  输出: 音频波形
    └────┬────┘
         │
         ▼
    音频文件输出
```

#### 3. 分布式系统

学习 `distributed/omni_connectors/` 的实现：

```python
# 共享内存连接器
vllm_omni/distributed/omni_connectors/
├── base.py                  # 基类 OmniConnectorBase
├── shared_memory.py         # SharedMemoryConnector
├── mooncake.py              # MooncakeConnector
├── yuanrong.py              # YuanrongConnector
├── adapter.py               # 适配器
└── utils/                   # 工具函数
```

---

### 📖 阶段五：实践项目（持续）

**学习目标**：能够贡献代码

#### 1. 尝试修改示例

```python
# 修改采样参数
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

sampling_params = OmniDiffusionSamplingParams(
    height=1024,
    width=1024,
    num_inference_steps=30,  # 减少步数
    guidance_scale=7.5,      # 调整引导强度
    seed=42,
)
```

#### 2. 阅读测试用例

```bash
tests/
├── e2e/                # 端到端测试 (推荐)
│   ├── test_omni_diffusion.py
│   └── test_omni_ar.py
├── diffusion/          # Diffusion 模块测试
│   ├── test_attention.py
│   └── test_cache.py
├── distributed/        # 分布式测试
└── entrypoints/        # 入口点测试
```

#### 3. 尝试添加新模型

参考现有模型实现：

```bash
vllm_omni/diffusion/models/
├── interface.py         # 模型接口 (必读)
├── qwen_image/          # Qwen-Image (参考)
│   ├── __init__.py
│   ├── pipeline.py      # 管道实现
│   └── transformer.py   # 模型实现
└── flux/                # FLUX (另一个参考)
```

阅读官方文档: `docs/contributing/model/adding_diffusion_model.md`

---

## 学习技巧

### 1. 善用设计文档

```
docs/design/
├── architecture_overview.md    # 必读！整体架构
├── feature/
│   ├── disaggregated_inference.md  # 分离式推理
│   └── ray_based_execution.md      # Ray 执行
└── module/
    ├── ar_module.md            # AR 模块详解
    └── dit_module.md           # Diffusion 详解
```

### 2. 断点调试

在关键位置设置断点：

```python
# vllm_omni/entrypoints/omni.py
def generate(self, prompts, ...):
    # 在这里设置断点
    ...

# vllm_omni/diffusion/diffusion_engine.py
def step(self, requests):
    # 在这里设置断点
    ...
```

### 3. 日志输出

```python
# 方式 1: Python logging
import logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# 方式 2: 环境变量
import os
os.environ["VLLM_LOGGING_LEVEL"] = "DEBUG"
os.environ["VLLM_TRACE_FUNCTION"] = "1"  # 函数调用追踪
```

### 4. 关注核心概念

| 概念 | 描述 | 位置 |
|------|------|------|
| `OmniStage` | 阶段抽象 | `entrypoints/omni_stage.py` |
| `OmniRequestOutput` | 输出结构 | `outputs.py` |
| `OmniConnector` | 阶段间通信 | `distributed/omni_connectors/` |
| `SamplingParams` | AR 采样参数 | vLLM |
| `OmniDiffusionSamplingParams` | Diffusion 采样参数 | `inputs/data.py` |
| `OmniModelConfig` | 模型配置 | `config/model.py` |

### 5. 使用 IDE 功能

- **Go to Definition**: 跳转到定义
- **Find References**: 查找引用
- **Call Hierarchy**: 调用层次
- **Type Hierarchy**: 类型层次

---

## 项目亮点总结

| 特性 | 描述 |
|------|------|
| 🔥 **多模态支持** | 文本、图像、视频、音频输入输出 |
| 🚀 **高性能** | 继承 vLLM 的 KV Cache 优化 + Diffusion 加速 |
| 🔧 **灵活管道** | 可配置的多阶段异构管道 |
| 🌐 **分布式** | 支持多节点分布式推理 |
| 🔌 **易扩展** | 模块化设计，易于添加新模型 |
| 📊 **多平台** | 支持 CUDA、ROCm、NPU、XPU |
| 🎯 **OpenAI 兼容** | 提供 OpenAI 兼容的 API 服务器 |

---

## 常见问题 FAQ

### Q1: 如何选择合适的模型？

| 任务 | 推荐模型 |
|------|----------|
| 文生图 | Z-Image-Turbo, Qwen-Image, FLUX |
| 文生视频 | Wan2.2 |
| 多模态对话 | Qwen3-Omni, Qwen2.5-Omni |
| 图像理解+生成 | BAGEL |

### Q2: 内存不够怎么办？

```python
# 启用 CPU 卸载
omni = Omni(
    model="...",
    enable_cpu_offload=True,
)

# 或者使用分层卸载
omni = Omni(
    model="...",
    enable_layerwise_offload=True,
    layerwise_num_gpu_layers=1,
)

# VAE 优化
omni = Omni(
    model="...",
    vae_use_slicing=True,
    vae_use_tiling=True,
)
```

### Q3: 如何使用多 GPU？

```python
from vllm_omni.diffusion.data import DiffusionParallelConfig

parallel_config = DiffusionParallelConfig(
    tensor_parallel_size=2,      # 张量并行
    ulysses_degree=2,            # Ulysses 序列并行
    ring_degree=1,               # Ring 序列并行
    cfg_parallel_size=2,         # CFG 并行
)

omni = Omni(
    model="...",
    parallel_config=parallel_config,
)
```

### Q4: 如何使用缓存加速？

```python
# TeaCache
omni = Omni(
    model="...",
    cache_backend="tea_cache",
    cache_config={"rel_l1_thresh": 0.2},
)

# cache-dit
omni = Omni(
    model="...",
    cache_backend="cache_dit",
    cache_config={
        "Fn_compute_blocks": 1,
        "max_warmup_steps": 4,
        "residual_diff_threshold": 0.24,
    },
)
```

---

## 参考资源

### 官方资源

- [vLLM-Omni GitHub](https://github.com/vllm-project/vllm-omni)
- [vLLM-Omni 文档](https://vllm-omni.readthedocs.io/)
- [vLLM 官方文档](https://docs.vllm.ai/)

### 论文

- [vLLM-Omni Paper](https://arxiv.org/abs/2602.02204)
- [vLLM Paper](https://arxiv.org/abs/2309.06180)

### 社区

- Slack: `#sig-omni` @ [slack.vllm.ai](https://slack.vllm.ai)
- 论坛: [discuss.vllm.ai](https://discuss.vllm.ai)

---

*本文档由 AI 助手生成，最后更新: 2026-02-03*

