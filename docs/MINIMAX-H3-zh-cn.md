# 在 Apple Silicon 上运行 MiniMax H3

[English](MINIMAX-H3.md) | 简体中文

**MiniMax H3（FL2VA）** 是一个 330 亿参数的视频模型，只需一段提示词，就能**同时
生成画面和它的配乐**。vpipe 通过自己的 **metal-compute** 后端在本机运行它：使用
vpipe 自己的 Metal kernel，前向计算中不使用 Python，也不使用第三方张量库。

特别之处在于这个**同时**：视频和音频不是两个模型在最后拼在一起。一个去噪循环跑在
同一条打包序列上，画面和声音都在其中，所以声音是在「知道」画面的情况下生成的，而
不是事后配上去的。你要雨声，它给你的就是听得见的雨。

它还是**蒸馏过的**（guidance-distilled），这也是它在这里实用的原因：每
一步不需要第二次无条件前向计算，**8–16 步**就能出可用的结果，而不是 30 步以上。

<a id="contents"></a>
## 目录

- [准备条件](#what-you-need)
  - [磁盘空间](#disk-space)
- [流水线](#the-pipelines)
- [第 1 步——准备模型](#step-1--prepare-the-model)
  - [先选好工作目录](#first-choose-a-work-directory)
  - [然后运行流水线](#then-run-the-pipeline)
- [第 2 步——文本生成视频**与音频**](#step-2--text-to-video-and-audio)
  - [值得了解的设置](#the-settings-worth-knowing)
  - [耗时](#how-long-it-takes)
  - [看着它成形——实时预览](#watching-it-form--live-previews)
  - [不只是文本输入](#more-than-text-in)
  - [以参考素材为条件（Ref2VA）](#conditioning-on-references-ref2va)
    - [两种传入参考素材的方式](#two-ways-to-hand-it-a-reference)
    - [这次不用参考——只用提示词](#no-references-this-time--prompt-only)
    - [示例](#the-example)
    - [代价](#what-it-costs)
    - [参考素材是怎么被读取的](#how-a-reference-is-read)
    - [不是文件的参考素材](#references-that-are-not-files)
    - [准备 Ref2VA 模型文件](#preparing-the-ref2va-checkpoint)
    - [类 Ref2VA——在 FL2VA 权重上使用参考图](#ref2va-like--references-on-the-fl2va-weights)
  - [更长的片段——一个故事分四段](#longer-clips--one-story-in-four-parts)
    - [按顺序运行整条链](#running-the-chain)
    - [每一段如何取得它的引导片段](#how-a-part-takes-its-guide)
    - [撰写提示词](#writing-the-prompts)
    - [把四段接起来](#joining-the-parts)
    - [改成你自己的](#making-it-your-own)
  - [直接运行发布的权重（两个模式都可以）](#the-released-weights-either-partition)
  - [更少的步数——Turbo LoRA](#fewer-steps--the-turbo-lora)
    - [获取](#get-it)
    - [运行](#run-it)
    - [运行时应用（推荐）](#runtime-recommended)
    - [同时用两个](#two-at-once)
    - [合并，以及为什么它会丢掉这个适配器的大部分](#merging-and-why-it-loses-most-of-this-adapter)
    - [哪些 Turbo 适配器可用](#which-turbo-adapters-work)
    - [社区 LoRA——Civitai、musubi-tuner、ai-toolkit](#community-loras--civitai-musubi-tuner-ai-toolkit)
  - [八步——HyperFlow](#eight-steps--hyperflow)
    - [获取，然后指定它](#fetch-it-then-name-it)
    - [它的不同之处](#what-makes-it-different)
    - [代价](#what-it-costs-1)
  - [更快的注意力——VDN 线性分支](#faster-attention--the-vdn-linear-branch)
    - [获取并运行](#get-it-and-run-it)
    - [能省多少](#what-it-saves)
  - [更省的注意力——SageAttention 的 int8 QK](#cheaper-attention--sageattentions-int8-qk)
  - [更快的注意力——Sol-Attn 路由](#faster-attention--sol-attn-routing)
    - [能省多少](#what-it-saves-1)
    - [与 VDN 分支的对比](#against-the-vdn-branch)
    - [可调项](#the-knobs)
  - [神经引擎（ANE）——面向 M4 系列 Mac](#the-neural-engine--for-m4-family-macs)
    - [在 M4 上能省多少](#what-it-saves-on-an-m4)
    - [它的行为](#how-it-behaves)
- [内存](#memory)
- [排查](#troubleshooting)
- [实现内幕](#under-the-hood)
- [参考资料与许可协议](#references-and-licences)

<a id="what-you-need"></a>
## 准备条件

| | |
|---|---|
| **机器** | Apple Silicon Mac（M 系列）。 |
| **内存** | **最低 16 GB。** 越大越快——见[内存](#memory)。 |
| **磁盘** | 准备阶段需 **约 155 GB**，完成后保留 **约 45 GB**。见下文。 |
| **构建** | Apple Silicon 版 vpipe——在 arm64 macOS 上默认即是。详见主 [README](../README.md)。 |

<a id="disk-space"></a>
### 磁盘空间

发布的模型文件是 bf16 格式，体积很大；vpipe 会预先一次性把它量化为 8-bit。量化
运行期间两份副本同时存在：

| | |
|---|---|
| 源重打包版本（`Comfy-Org/MiniMax-H3`，bf16） | **约 115 GB** |
| 准备期间峰值（源 + 输出） | **约 180 GB** |
| 8-bit 模型（删除源文件后） | **约 65 GB** |
| 追加 Ref2VA 模式 | 源 **+66 GB**，8-bit **+60 GB**（4-bit 约 **+44 GB**） |

准备完成后即可删除源重打包版本，只保留那约 65 GB。115 GB 是一次性开销，不是长期
占用。

Ref2VA 那一行指的是**整个模型**，而不只是它的 transformer，这一点值得读两遍。
*下载*确实只多出 66 GB 的 transformer——提示词编码器和两个 VAE 都已在磁盘上，会
被跳过。但准备它的流水线**连编码器也一起量化**，输出到它自己的目录，所以你最终
持有的是第二份完整模型：8-bit 下约 33 GB 的 transformer 加约 27 GB 的编码器。
只有量化不去改动的组件才会做硬链接，而重新量化过的编码器是新的字节。4-bit 的
数字是把这同一个总和按 transformer 自身 8-bit 到 4-bit 的比例缩放得来的，并非
实测值；把它当作预算，而不是承诺。

**为什么选 8-bit 而不是 4-bit。** 为了质量，代价只是磁盘，其他几乎没有。两种
位宽都走流式加载，所以都不必装进内存，而 transformer 的时间花在从存储中读取权重
上，而不是花在权重有多宽上——所以换成 8-bit 大约多占 20 GB，但并不会明显改变一
段片段所需的时间。4-bit 路径依然可用；如果磁盘更重要，就走这条：在两个
`model-quantize` stage 中把 `bits` 设为 4，并相应地命名输出。

**你完全可以不做量化。** 下载下来的 bf16 重打包版本就是一个完整、可直接运行的
模型：把 `model-select` 直接指向 `Comfy-Org/MiniMax-H3-FL2VA`，
把[第 1 步](#step-1--prepare-the-model)里的量化 stage 全部跳过即可。不需要告诉
任何组件位宽是多少——加载器会从模型文件中读出每个张量的位宽，而不携带位宽信息的
模型文件就是稠密 bf16 路径。

它的代价是内存，而且这份代价是按每一步付、而不是一次付清。transformer 在 bf16
下约 66 GB，8-bit 下约 33 GB，所以流式加载它在每次前向计算中要重读大约两倍的
字节，而常驻集——它会增长到占满一切空余内存，见[内存](#memory)——在两步之间能
保留的分块按比例更少。在内存受限的机器上，差别就全在这里：同样的片段，更多时间
花在存储路径上。如果你打算生成不止一次，就做量化；如果你想在花掉那几个小时和
115 GB 之前先看看模型的效果，就直接运行下载来的重打包版本。

> **把下载文件和输出放在同一个文件系统上。** 量化不会触及的组件（两个 VAE）是
> **硬链接**进输出目录的，而不是复制过去的，这正是上面那些数字比看起来更小的
> 原因。跨两个卷时链接会失败，vpipe 退回为真正的复制，那些字节你就要付两遍。

<a id="the-pipelines"></a>
## 各个流水线

- **[`prepare-minimax-h3-8bit.vpipeline`](pipelines/prepare-minimax-h3-8bit.vpipeline)**
  ——下载模型文件并量化。只需运行一次。
- **[`prepare-minimax-h3-ref2va-8bit.vpipeline`](pipelines/prepare-minimax-h3-ref2va-8bit.vpipeline)**
  / **[`…-4bit`](pipelines/prepare-minimax-h3-ref2va-4bit.vpipeline)**
  ——对 **Ref2VA** 模式做同样的事（见
  [以参考素材作为条件](#conditioning-on-references-ref2va)）。只在你需要时运行；
  两者共用同一次下载。
- **[`minimax-h3-text-to-video.vpipeline`](pipelines/minimax-h3-text-to-video.vpipeline)**
  ——输入提示词，输出带声音的 `.mp4`。
- **[`minimax-h3-text-to-video-preview.vpipeline`](pipelines/minimax-h3-text-to-video-preview.vpipeline)**
  ——同上，外加片段逐步成形的**实时预览**（见[看着它成形](#watching-it-form--live-previews)）。
- **[`minimax-h3-first-last-to-video.vpipeline`](pipelines/minimax-h3-first-last-to-video.vpipeline)**
  ——同上，但首尾两端都锚定到一张图片（见[不止文本输入](#more-than-text-in)）。
- **[`minimax-h3-reference-to-video.vpipeline`](pipelines/minimax-h3-reference-to-video.vpipeline)**
  ——改用 **Ref2VA** 模式：输入参考图片、片段和音轨，输出 `.mp4`，每一项都会按你
  选择的尺寸预处理（见[以参考素材作为条件](#conditioning-on-references-ref2va)）。
- **[`minimax-h3-ref2va-like.vpipeline`](pipelines/minimax-h3-ref2va-like.vpipeline)**
  ——在 **FL2VA** 权重上使用参考图，并挂上 VDN 分支和 Turbo LoRA，因此不需要第二个
  66 GB 的 transformer（见[类 Ref2VA](#ref2va-like--references-on-the-fl2va-weights)）。
- **[`minimax-h3-extend-part1.vpipeline`](pipelines/minimax-h3-extend-part1.vpipeline)**
  … **[`…-part4`](pipelines/minimax-h3-extend-part4.vpipeline)**
  ——用四次运行生成一段 **40 秒**的片段：先是 FL2VA 文生视频，然后是三段 Ref2VA
  续接，每一段都从上一部分的最后 3.75 s 接着往下走（见
  [更长的片段](#longer-clips--one-story-in-four-parts)）。以 21 步、不加 Turbo 适配器
  运行**官方发布权重**的两个模式，并把每一部分**无损**写出——FFV1 视频、4:4:4、
  ALAC 声音——所以每一次续接读回的正是上一部分生成的内容。
- **[`minimax-h3-extend-concat.vpipeline`](pipelines/minimax-h3-extend-concat.vpipeline)**
  ——通过 concat demuxer 把这四部分拼接成一个 40 秒的文件，不需要模型，也不需要手写
  ffmpeg。这是整条链中唯一一次有损编码。
- **[`prepare-minimax-h3-vdn.vpipeline`](pipelines/prepare-minimax-h3-vdn.vpipeline)**
  / **[`minimax-h3-vdn.vpipeline`](pipelines/minimax-h3-vdn.vpipeline)**
  ——获取 **VDN** 混合注意力分支，并用它运行文生视频。**只支持 FL2VA 模式**——
  文本或首/末关键帧都可以，参考素材不行——片段**越长、画幅越大**，它的价值越高
  （见[更快的注意力](#faster-attention--the-vdn-linear-branch)）。

点开链接后用 **Raw ▸ Save as** 下载，或者直接从你克隆的仓库的 `docs/pipelines/`
里取。两种方式得到的文件都可以在终端用 `vpipe --launch <file>` 运行，也可以在
web UI 的 Pipeline Manager 里用 **Load** 打开（或在手机 UI 的 ⋯ 菜单里打开）；
下面的第 1 步用 CLI，第 2 步用 web UI，因为这分别是各自任务最合适的方式。两者都是
普通的 JSON——可以阅读、修改，并纳入版本控制。

<a id="step-1--prepare-the-model"></a>
## 第 1 步——准备模型

<a id="first-choose-a-work-directory"></a>
### 首先，选一个工作目录

vpipe 把**你启动它时所在的目录**当作自己的工作区，并在那里创建自己的状态文件：

| | |
|---|---|
| `models/` | 你下载或量化的所有模型 |
| `data.mdb`、`lock.mdb` | LMDB 数据库——模型注册表、日志、stage 输出 |
| `sandbox/` | 仅由 **`vpipe-web-ui`** 创建：它把 stage 的文件读写限制在这个目录内 |

由此有两点。请在有**约 155 GB** 空间（见[磁盘空间](#disk-space)）的卷上挑一个目录
——下载就落在那里。并且在第 2 步中使用**同一个**目录：你即将准备的模型记录在该目录
的注册表里，所以从别处启动的运行找不到它。

<a id="then-run-the-pipeline"></a>
### 然后运行流水线

```sh
cd ~/vpipe-work                                    # your work directory
cp ~/src/vpipe/docs/pipelines/prepare-minimax-h3-8bit.vpipeline .
~/src/vpipe/build/apps/vpipe/vpipe --launch prepare-minimax-h3-8bit.vpipeline
```

这项工作用 CLI 比用 web UI 更合适：它耗时长、无人值守、受磁盘限制，启动后也没有什么
可点的。它会为每个 stage 打印实时进度条，`Ctrl-C` 可以干净地停止——
`skip_existing_files` 意味着重新运行会从上次中断处继续。

四个 stage，依次是：

1. **`model-fetch`**——把 `Comfy-Org/MiniMax-H3` 拉取到 `./models`。
   `skip_existing_files` 已开启，所以中断后重新运行会跳过磁盘上每个大小正确的分片，
   而被截断的那个会从停下的那一字节续传，而不是重新开始——未完成的部分以
   `<name>.part` 的形式放在它旁边，直到下载完整且校验和匹配。这里一个分片有
   30–60 GB，所以这个差别是以小时计的。

   超过 256 MB 的分片是从 HuggingFace 的**内容存储（content store）**拉取的，
   而不是流式下载：仓库会发布一个可用于重建文件的哈希，而存储把它保存为去重、
   压缩后的分块，一次可以下载多个字节区间。对 bf16 权重来说，这相当于 0.873×
   的字节量——存储会先把字节平面分离再压缩，这才让浮点数有了可压缩性。在一个
   5.2 GB 的分片上端到端实测，这里快了 1.07×（中位数 94.4 对 88.3
   MB/s），因为单条流已经接近这条链路的上限；在单条流构成瓶颈的链路上，它的价值
   接近 2×。`xet_streams: 0` 可以关掉它。

   **`model_variant: fl2va` 是必需的，不是装饰。** 那一个仓库发布了*两个*模型
   ——FL2VA 与 Ref2VA 两种划分——它们锁定的 transformer 文件不同，tokenizer 也
   不同。没有说明要哪一个的拉取会被拒绝，并列出两者，而不是悄悄取第一个。

   **`model_key` 是下一个 stage 所引用的名字。** 两种划分在磁盘上共用一个目录，
   所以把它们区分开的是各自在模型数据库中注册所用的键——不是仓库路径，因为两者
   的仓库路径是同一个字符串。`model_key: Comfy-Org/MiniMax-H3-FL2VA` 锁定了它，
   而第 2 步的 `src_model` 必须就是这个键。在那里传入仓库路径
   （`Comfy-Org/MiniMax-H3`）是要避免的错误：拉取会成功，随后的量化却找不到这个
   名字的模型。不设置时，键会回退到模型目录条目的名称，在这里恰好是同一个字符串
   ——所以锁定它没有任何代价，还能让流水线说清它依赖什么。
2. **`model-quantize`**（`target: dit`）——把 33B 的 transformer 量化到 8-bit，
   group size 64。**不要改动 `quant_modulation: true`。** H3 的每个 block 的
   AdaLN 调制不像其他 DiT 那样只是一个小的旁路投影——它是 **33B 中的 13B**，
   所以把它留在 bf16 就意味着在一个原本 8-bit 的包里保留 26 GB 未量化的
   transformer，而这几乎就是量化的全部意义所在。加载器会从模型文件中读取每个张量
   的位宽，所以下游不需要被告知什么量化成了什么。
3. **`model-quantize`**（`target: text_encoder`）——把 Qwen3-VL-32B 提示词编码器
   量化到 8-bit。它的输出 `local/MiniMax-H3-FL2VA-8bit` 是一个**完整的模型**：
   量化后的部分加上所有未改动的部分。
4. **`model-remove`**——删除第 2 步产生的中间结果，它的存在只是为了供第 3 步使用。

所有东西都落在工作目录下的 `models/` 里。这需要一些时间，且主要受磁盘限制。完成后，
`local/MiniMax-H3-FL2VA-8bit` 是你唯一需要的东西；`Comfy-Org/MiniMax-H3` 的下载
文件可以删了。

> **受限访问（gated）仓库。** 如果 `model-fetch` 报告授权失败，请在该模型的
> Hugging Face 页面上接受其许可协议，并把令牌填进这个 stage 的 `hf_token`。

<a id="step-2--text-to-video-and-audio"></a>
## 第 2 步——文本生成视频**和音频**

```sh
cd ~/vpipe-work                                    # the SAME work directory
~/src/vpipe/build/apps/web-ui/vpipe-web-ui
```

请从你准备模型时所用的那个工作目录启动 web UI——保存着你的模型的注册表就在那里。
然后打开它打印出的 URL，加载 `minimax-h3-text-to-video.vpipeline`，编辑
`text-prompt` stage，按 Start。随附的提示词一次就要求了两半：

> *Cinematic video of a young Asian female pianist passionately playing a
> grand piano.*

配乐就在那句话里：点明乐器正是让模型给出音乐的原因。这里没有单独的音频提示词
——声音来自同一段文本——所以除了画面，也要描述**声音**。要么直接点明
（*"sound of the rain, with the occasional individual drop"*），要么像这里一样，
点明发出声音的东西。一条只说场景长什么样的提示词，得到的是模型认为那个场景听起来
像什么。

这张图有九个 stage：

```
text-prompt ──> diffusion-conditioner ──> generate-video ─┬─0─> vae-decode ──> rgb-to-video ─┐
                                                          │                                  ├─> save-video
                                                          └─1─> audio-vae-decode ────────────┘

model-select            ──> diffusion-conditioner, generate-video, vae-decode, audio-vae-decode
minimax-h3-model-config ──> generate-video (port 9)
```

`model-select` 只需指定模型一次，每个持有模型的 stage 都会锁定它，所以你是把**一个**
stage 指向模型文件，而不是四个。
`generate-video` 输出**两个**潜变量——视频在端口 0，音频在端口 1——它们分别解码，
在 `save-video` 处再次汇合，混流成一个 `.mp4`。

随附的 `output_url` 是**相对路径**——`minimax-h3-text-to-video.mp4`——所以片子会落在
你启动 vpipe 的位置旁边，同一个文件在 CLI 和 web UI 下都能用，无需改动。

如果你见到另一种写法，值得知道：在 web UI 下**开头的 `/` 指的是 sandbox 根目录**，
不是你的文件系统根目录，所以那里的 `/clip.mp4` 意思是工作目录里的
`sandbox/clip.mp4`。CLI 没有 sandbox，`/` 就是它通常的含义——在 Mac 上那是一个只读
卷，所以一张从 UI 带着绝对路径过来的图会生成完整的片子，然后写入失败。

<a id="the-settings-worth-knowing"></a>
### 值得了解的设置

来自 `generate-video` stage：

| 配置项 | 随附值 | 说明 |
|---|---|---|
| `width` / `height` | 960 × 544 | **向上取整**到最近的 **32** 的倍数——即视频 VAE 的 16× 空间步长乘以 DiT 的 2× patch。16 的倍数还不够：1360 是 16 的倍数，而它的潜变量是奇数 85，打包器无法对其做 patch。这个 stage 会把改动记进日志。 |
| `frames` | 120 | **向上取整**到 VAE 能分块的最近帧数——5、22、39、56、73、90、107、**124**……所以 120 变成 124。这个 stage 会把改动记进日志。 |
| `fps` | 24 | 124 帧 ≈ 5.2 s；56 帧 ≈ 2.3 s。 |
| `steps` | 8 | **8 步是草稿质量**——足以看出一条提示词的效果——而 **16 步给出良好质量**。少于 8 步是 [Turbo LoRA](#fewer-steps--the-turbo-lora) 的领域，不是这个模型本身的。`guidance_scale` 和负面提示词在这里是**无效的**——蒸馏模型没有无条件前向可以用来引导，所以 vpipe 直接跳过它，而不是在一个 33B 模型上白付 2× 的代价。 |
| `seed` | 6 | 相同的 seed + 相同的设置 ⇒ 相同的片子。 |
| `i8_gemm` | `true` | 一个需要显式开启的**有损**加速模式，本文随附的每个流水线都开着它。只有带矩阵核心的 GPU（M5 及更新）能用，所以在 M4 上它什么也不做——而在 M5 上关掉它更慢。它会让画面略有变化，所以当你在评判输出而非速度时，把它关掉。 |
| `sage_attn` | `false` | 一个需要显式开启的**有损**加速模式，与 `i8_gemm` 和 `sol_attn` 都互相独立，可以和任一个一起设置——它把注意力的 QK^T 乘积用 int8 加逐块 scale 来算，而 `sol_attn` 决定哪些块根本参与注意力。在视频几何尺寸下注意力提速 1.20×，cosine 与 f16 kernel 相同。仅限矩阵核心。见[更省的注意力——SageAttention 的 int8 QK](#cheaper-attention--sageattentions-int8-qk)。 |
| `ane_ffn` / `ane_qkv` | `false` | 需要显式开启的**有损**模式，把每个 block 的一部分放到 GPU 旁边的 **Apple Neural Engine** 上运行。在 **M4 系列** Mac 上值得开，在 M5 上通常不值得。见[Neural Engine](#the-neural-engine--for-m4-family-macs)。 |
| `sol_attn` | `false` | 又一个需要显式开启的**有损**加速模式，而且是独立的——它改变 GEMM 之间的注意力的计算方式，而 `i8_gemm` 改变的是 GEMM。在 832 × 480 的 124 帧上，挂钟时间提速 1.27×，且不需要额外权重；旁边的各个选项见[更快的注意力——Sol-Attn 路由](#faster-attention--sol-attn-routing)。 |
| `unload_when_idle` | `always` | 两次运行之间丢弃权重。在 16 GB 上，正是这一项让下一个 stage 能用上整台机器。 |

以及来自 **`minimax-h3-model-config`** stage 的设置，它连接到 `generate-video` 的
`model_config` 输入端口（iport）（端口 9）：

| 配置项 | 随附值 | 说明 |
|---|---|---|
| `video_shift` / `audio_shift` | 12.0 / 3.0 | 两条 sigma 调度。不可互换——这是发布的模型文件所用的值。 |
| `condition_timestep` | 1.0 | 被钉住的关键帧行所处的噪声水平。在这个模型的 `t = 1 − sigma` 约定下，`1.0` 表示**干净**。 |
| `condition_audio_timestep` | 1.0 | 同上，用于 Ref2VA 的参考**配乐**。 |
| `audio_seconds` | 0 | 音频长度由 `frames` 和 `fps` 推出；只在要覆盖它时才设置这一项。 |

这些是 H3 自己的选项，所以它们放在一个 H3 的 stage 里，而不是放在 `generate-video`
里——后者只保留每个视频模型都会响应的东西（几何尺寸、长度、步数、seed、驻留）。
不放这个 stage，上面的默认值就生效。给它一个**trigger** 输入端口（iport），它就会
在每个到来的节拍上重新发出一次，这样在一张持续生成的图里，设置可以逐片改变；没有
trigger 时，它在整次运行中只发出一次。

这里有意**没有 guidance scale**：H3 是蒸馏模型，而蒸馏模型没有无条件前向可以用来
引导。出于同样的原因，Wan 的 guidance 和专家边界住在 `wan2-model-config` 里——每个
模型家族都带着自己的那一份。

<a id="how-long-it-takes"></a>
### 耗时多久

在 8-bit 模型上实测，960 × 544（0.5 MP）、24 fps、6 步，运行时应用了
[Turbo LoRA](#fewer-steps--the-turbo-lora)，测试机是能跑起它的最小机器——一台
无风扇的 **MacBook Air 15 英寸（M5）**，10 核 CPU / 10 核 GPU、16 GB——以及一台
**MacBook Pro 16 英寸（M5 Pro）**、24 GB：

| 帧数 | 片长 | M5 Air，16 GB | M5 Pro，24 GB |
|---|---|---|---|
| 90 | 3.75 s | **9 min 26 s** | **3 min 20 s** |
| 124 | 5.2 s | **11 min 25 s** | **5 min 0 s** |

`steps` 是对此影响最大的设置，而 Turbo 适配器正是让步数能压到这么低的东西：没有它
的话，草稿按 8 步、成片按 16 步来计划，代价大致成正比。而改变*每一步*代价的是两个
注意力设置之一：[VDN 线性分支](#faster-attention--the-vdn-linear-branch)，它把 124
帧的 Pro 运行压到 **4 min 38 s**，片子越长省得越多；或者
[Sol-Attn 路由](#faster-attention--sol-attn-routing)，它不需要额外权重，把 832 × 480
下 124 帧的一次运行从 3 min 30 s 压到 **2 min 44 s**。

**这两列的可靠程度并不相同，值得说清哪一列是哪样。** M5 Pro 那一列是可重复的：有
风扇、一个钉住的时钟频率、每次运行都是同一个数。M5 Air 那一列不是。冰袋是手放的，
放在哪里既会改变 boost 窗口持续多久，也会改变之后时钟跌到多低，所以那些数字带有
本文未做量化的运行间波动。

于是差距读起来是 **90 帧时 2.8×、124 帧时 2.3×**，但这两者之间的差别不是结构性的
——主要是 Air 在动。缩放也是同理：90 → 124 帧是 1.38× 的长度，Air 花了 1.21× 的
时间，而 Pro 是 1.50×，其中只有 Pro 的数字算是一次测量，而不是一个噪声量的一次采样。
请把 Air 那一列当作一台散热良好的无风扇 M5 能达到的量级，把 Pro 那一列当作你可以
复现的数字。

> **M5 Air 那一列是把机身放在冰袋上测出来的**，即便如此，它在大部分运行时间里仍是
> 一台降频的机器。它以满频 **1578 MHz** 起步，大约保持**头两分钟**，然后降频并稳定
> 在 **1300 MHz** 附近波动——是这颗芯片的 82%——因为冰袋是一块会变热的散热片，
> 不是稳定的散热。
>
> 所以这个惩罚是任务**多长**的函数，不是一笔固定的税：一个两分钟的任务永远不会离开
> boost 窗口，而这些运行有**79–82% 的挂钟时间**在窗口之外。请把 Air 那一列理解为
> 一台开头很快、到结尾就不快了的机器。
>
> 它还很**嘈杂**，而另一列不是：冰袋是手放的，放在哪里既改变 boost 窗口的长度，
> 也改变它最终稳定到的时钟频率，所以重跑一次 Air 不会重现它的数字。放在温热的桌面
> 上还会跌得更多——那里 124 帧的运行大约要 15 分钟。
>
> **M5 Pro 那一列是那台机器的出厂状态**，靠它自己的风扇、下面什么也不放，而它的
> 风扇足够好，以至于这个负载把 GPU 钉在 **1620 MHz——也就是最高频——整次运行都在
> 100%**。那是 Air 持续频率的 **1.25×**，这还没算核心数的差别，所以时钟本身能解释
> 2.3–2.8× 差距的一部分，而不是大部分。

<a id="watching-it-form--live-previews"></a>
### 看着它成形——实时预览

这个模型的一次去噪要跑几分钟，没有预览的话，直到最后都没有东西可看。`generate-video`
可以让你看着片段成形：每走完一步，它就取出模型此刻对成片的最佳估计，用一个**小型自编码器**
（TAE）而不是真正的视频 VAE 解码，从 `generate-video` 的端口 2 送出。接在那里的 `preview`
阶段会在网页界面里播放它，每段循环播放，直到下一步的预览替换它。

**[`minimax-h3-text-to-video-preview.vpipeline`](pipelines/minimax-h3-text-to-video-preview.vpipeline)**
就是加上了预览的文本生成视频图。它需要一个小文件，
[`prepare-minimax-h3-preview.vpipeline`](pipelines/prepare-minimax-h3-preview.vpipeline)
会下载它（23 MB）并注册为 `madebyollin/taeh3`：

```sh
cd ~/vpipe-work                                    # 同一个工作目录
vpipe --launch ~/src/vpipe/docs/pipelines/prepare-minimax-h3-preview.vpipeline
```

这是 madebyollin 的 **`taeh3`**，为这个模型的潜空间训练。它发布在 GitHub 而不是
Hugging Face 上，模型目录知道去哪里取。它解出的帧数与真正的 VAE 相同，动作也保留下来。
**Kijai 的 `MiniMax-H3-TAE`** 也能用（`Kijai/MiniMax-H3-TAE`，10 MB，同样在目录里）。
不过它是逐帧的静态图解码器，每个潜帧只出一帧，放慢播放以填满同样的时长。在同一段片子上，
它与真正 VAE 的一致度也更低：23.3 dB，`taeh3` 是 25.3 dB。

这些选项在 `minimax-h3-model-config` 上，和 H3 的其它设置放在一起：

| 键 | 默认值 | 说明 |
|---|---|---|
| `preview_vae` | *（空）* | TAE：一个已注册的模型（`madebyollin/taeh3`）、一个只含一个 `.safetensors` 的目录，或指向它的路径。留空即关闭预览。 |
| `preview_every` | 1 | 每 *N* 步渲染一次；最后一步总会渲染。 |
| `preview_max_edge` | 512 | 预览画面的最长边。TAE 总是以原尺寸解码，之后再把画面缩小。实测先缩小潜变量会让画面变糊，颜色也会偏。 |
| `preview_frames` | 0 | 只预览前 *N* 帧；0 表示整段。这是让预览更省的那个选项。 |

实测代价（M4 Pro）：

- GPU 空闲时，完整的 90 帧、960 × 576 片段解码需 **0.94 秒**。
- DiT 运行时（模型常驻，每步 8 秒），一段 39 帧、512 × 288 的预览约需 **0.5 秒**。
  这一项和下面的 3% 是在完整片段还需 2.1 秒、解码器的 ReLU、残差加、上采样和拼接尚未
  并入卷积时测的，请把两者都看作上限。
- 解码不在生成线程上进行。下一次预览到期时如果上一次还在渲染，排队中的那段会被替换，
  而不是排在后面，所以预览不会让生成等待。
- 不过解码确实与去噪共用 GPU。**每一步**都预览，大约让去噪多花 **3%**：不预览 85 秒，
  预览 87 秒和 88 秒（7 步，交替运行）。调大 `preview_every` 可以少花一些。
- 最后一步的预览与真正 VAE 的解码结果一致度为 **30–32 dB**。

端口 2 不接，或 `preview_vae` 留空，就什么都不加载、不解码。有两点要知道：

- **只能接一个消费者。**没人读取的预览会被丢弃，而不是拖住生成，这种策略只支持一个消费者。
- **`preview` 阶段会一直运行，直到你停止流水线。**它是实时画面，所以含有它的图在命令行下
  不会自己结束。请在网页界面里使用它。

<a id="more-than-text-in"></a>
### 输入不止文本

这个模型文件名叫 **FL2VA**——*first-and-last to video and audio*（首尾帧生成视频与
音频）。给 `generate-video` 的端口 5 喂一张图片的 `vae-encode`，生成就会以它为开头
帧被锚定；再在端口 6 上加第二个 `vae-encode`，模型就会在两张静态图之间做插值。两个
锚点都必须按片子生成时所用的同一分辨率来编码。

[`minimax-h3-first-last-to-video.vpipeline`](pipelines/minimax-h3-first-last-to-video.vpipeline)
就是这张图，并且已经走通了。在它旁边放一张 `reference.jpg`，然后像你运行文本生成
视频那个流水线一样运行它：

```sh
cd ~/vpipe-work                                    # the work directory again
cp ~/src/vpipe/docs/pipelines/minimax-h3-first-last-to-video.vpipeline .
cp <your picture> reference.jpg
~/src/vpipe/build/apps/vpipe/vpipe --launch minimax-h3-first-last-to-video.vpipeline
```

要从那个目录运行，不要从别处：`models/` 和模型注册表都是相对于 vpipe 启动的位置来
解析的，所以一张在错误位置指名 `local/MiniMax-H3-FL2VA-8bit` 的图会报告说每个 stage
都未生效。

它**从一张图片构建出两个锚点**，这正是把一张静态图变成一次镜头运动而不是一次交叉
淡入淡出的原因。`load-image` 扇出到两个 `image-resample` stage：

| stage | `fit` | 它取的画面 |
|---|---|---|
| `first-frame` | `crop` | 整张图片，居中裁剪到 960x544 |
| `last-frame` | `manual` | 其中一个更紧的窗口，`scale` 为 1.6875 |

模型会补上它们之间约 5 秒的内容，所以片子读起来像一次缓慢的推进。把两者互换，它就
变成拉远。

**两个重采样器都用 `algorithm: lanczos`。** 锚点是源图片的细节进入模型的唯一入口
——它们之后的一切都是潜变量——所以这是整张图里唯一值得为之付代价的一次缩放。双线性
会让细纹理（笔触、树叶、织物）变软，而 VAE 随后会把这种变软当成你有意为之来编码。

**`manual` 的这几个数字描述的是一张 1024x1024 的源图。** `src_x`/`src_y` 是进入你
自己图片的绝对像素，`scale` 是重采样比例，所以那个窗口是从该角点开始的
`width / scale` 乘 `height / scale` 像素——这里 960 / 1.6875 = 569 宽。把它指向另一
个尺寸的图片，你取到的就是别的画面。两条出路：重新调这三个数字，或者给两个
`image-resample` stage **两张不同的图片**，两个都用 `fit: crop`，这不需要任何算术，
也是*首尾帧*最朴素的读法。

**它应用了 Turbo LoRA**，这正是它能以 6 步运行的原因——所以请先拉取这个适配器
（见[更少的步数](#fewer-steps--the-turbo-lora)）。如果你不想用，就从
`minimax-h3-model-config` 中删掉 `lora` 和 `lora_scale` 两项，并把 `steps` 调回 8；
这张图里其他一切不变。

`frames` 是 **124**，不是文本生成视频示例里的 120，因为 H3 按每次 17 帧对视频分块并
保留 5 个潜变量——任何帧数都会被向上取整到下一个 `17n + 5`。指定一个本来就合规的
帧数，意味着文件里的数字就是你得到的数字。

两个 `vae-encode` stage 一旦编码完自己的关键帧就会释放 VAE，这件事比听起来更重要：
H3 的视频 VAE 有 5.2 GB，而 DiT 紧接着就要用它的暂存空间。在一台 16 GB 的机器上，
同时持有两者就是能出片和被拒绝之间的差别。

<a id="conditioning-on-references-ref2va"></a>
### 以参考媒体为条件（Ref2VA）

MiniMax-H3 还发布了**第二个模型文件** `ref2va`，它不以关键帧为条件，而是以
一*组*参考媒体为条件：最多 **9 张图片**、**3 段视频**和 **3 条音轨**，总计
十二个，而且音频永远不能是唯一的种类。图片承载主体与风格，视频承载运动与
镜头，音轨承载一段人声或一段音乐。

两个模式是同一套架构，随附的 transformer 配置逐字节相同，所以**权重里没有
任何东西能把它们区分开**——vpipe 是从打包方式上读出来的。把 Ref2VA 模型文件
当作 FL2VA 接线会被拒绝，而不是照跑：那样它会加载成功、以完整 33B 的代价
去噪，然后生成一段不以任何东西为条件的视频。*有意*不给任何参考则是另一种
请求，它是接受的——见[只用提示词](#no-references-this-time--prompt-only)。

反过来那个方向则是一种真实存在的用法，而不是接错线：FL2VA 权重同样能经由这条
序列接收参考图片，这就是下面的[类 Ref2VA](#ref2va-like--references-on-the-fl2va-weights)。
先读本节——接线、限额以及一个参考是怎么被读取的，两者完全相同——再去读那一节，
看有什么不一样。

> **参考不是关键帧，Ref2VA 也无法钉住某一帧。** 两个模式打包的序列不同——
> FL2VA 的是 `[text | keyframe conditions | target audio | target video]`，
> Ref2VA 的是 `[text | reference blocks | target audio | target video]`
> ——因此没有任何槽位能把一个参考绑定到输出的第 0 帧。一张参考图片作用于
> **整段视频**：它把主体、服装和风格带到每一处，却不特别落在某一处。
>
> 所以在提示词里提出这种要求（*"use `<Picture 2>` as the opening frame"*）
> 不会有任何效果。这不是模型不听话；根本没有可供这条指令作用的机制，露出
> 马脚的地方是：一次运行里服装和面容都迁移过来了，首帧却没有。锚定开头一帧
> ——比如从上一段视频的最后一帧继续——是
> [FL2VA 的职责](#more-than-text-in)，接在 `generate-video` 的 5 号端口上，
> 而两者互斥：用了锚定帧就要放弃参考列表。在 Ref2VA 图上接入关键帧时，
> `generate-video` 会**警告**，而不是悄悄丢掉它。
>
> 续接一段**视频**是另一种请求，而这个请求 Ref2VA 是接受的。一段被描述为
> `[video continuation]` 来源的参考视频，会从它的结尾继续下去：它的运动、
> 它的主体和它的声音都会延续。那是模型学到的行为，不是钉住的一帧；参见
> [更长的视频](#longer-clips--one-story-in-four-parts)。

接入一个 **`video-ref-encoder`** stage：

| | |
| --- | --- |
| 0 号输入 | 提示词 |
| 1 号输入 | 可选的 `model-select` |
| 2–7 号输入 | 可选的 `ref1`..`ref6`——以张量形式给出的参考 |
| 0 号输出 | 条件信息 → `generate-video` 的 0 号端口 |
| 1 号输出 | 参考视频行 → `generate-video` 的 **7** 号端口 |
| 2 号输出 | 参考音频行 → `generate-video` 的 **8** 号端口 |

<a id="two-ways-to-hand-it-a-reference"></a>
#### 交付参考的两种方式

**一份文件列表**，写在 stage 的 `references` 配置里。它们是要打开的路径，
所以编排器的文件浏览器会替你填好——**一次选中多个**，它们就会按你选择的
顺序进入列表：

```json
"references": ["subject.png", "motion.mp4", "voice.wav"]
```

**或者一个张量**，走六个 `ref` 输入端口（iport）中的一个。这是为那些路径
无法指称的参考准备的——你的图刚刚生成出来的一张静帧、一个裁剪出来的画面、
一段从未写到磁盘上的视频——也是为了由你自己决定每个参考的几何尺寸。约定见
下文[不是文件的参考](#references-that-are-not-files)。

两者可以自由混用：端口参考的编号排在列表*之后*，而各项上限（9 张图片、
3 段视频、3 条音轨、总计 12 个）作用于两者的并集。当参考就是你手头的文件、
并且你愿意让模型自己的规则来决定尺寸时，用列表；当参考由某个 stage 生成、
或者你想自己设定尺寸时，用端口。

<a id="no-references-this-time--prompt-only"></a>
#### 这次不用参考——只用提示词

**空列表**本身就是一种请求：只凭提示词生成，走的是已经加载好的 Ref2VA
模型文件。

```json
"references": []
```

这让同一张图既能处理带参考的请求，也能处理不带参考的请求——去掉最后一个
参考不再意味着换成 FL2VA 模型文件（在别处也用到 Ref2VA 的机器上，就是两份
都得常驻），之后再加一个参考也只是改列表，而不是改图。一个已接线的 `ref`
端口若发来**空张量**，表达的是同一个意思，因为那正是它约定的“本轮没有内容”
的写法。

实际运行的是文生视频自己的序列 `[text | target audio | target video]`，由
Ref2VA 权重来读：条件只有提示词本身，与 `diffusion-conditioner` 为它产出的
逐字节相同，不打包任何参考行。两边的日志都会写明：

```
VideoRefEncoderStage('refenc'): prompt only (an explicitly empty reference
  list) -> 34 conditioning rows, no reference rows
GenerateVideoStage('gen'): prompt-only Ref2VA -- the request's reference list
  is explicitly empty, so the Ref2VA weights denoise from the prompt alone
  over the text-to-video layout
```

**省略这个键不是同一个请求。** 既没有 `references` 键、也没有接任何 `ref`
端口时，编码器等于什么都没拿到——这正是一张没接好线的图的样子——它会给出
警告并跳过这个请求，而不是为它花掉一次 33B 的去噪。`generate-video` 也守着
同一条线：在 Ref2VA 模型文件上，一个不带参考列表的条件（比如来自
`diffusion-conditioner` 的）会被拒绝，只有明确为空的列表才会运行。

在 **FL2VA** 模型文件上，空列表就是普通的文生视频，接在 5 号端口上的关键帧
照常生效。在 Ref2VA 上则不然：只用提示词的请求仍然是 Ref2VA 的序列，而它
没有关键帧的位置。

这是让 Ref2VA 权重去做文生视频：序列允许这样做，但 Ref2VA 的配方并没有描述
它。FL2VA 模型文件才是为此训练的路线；这条路是给已经持有 Ref2VA 的图用的。

<a id="the-example"></a>
#### 示例

[`minimax-h3-reference-to-video.vpipeline`](pipelines/minimax-h3-reference-to-video.vpipeline)
用的都是你已经有的素材：

| 参考 | 承载什么 | 从哪里来 |
|---|---|---|
| `minimax-h3-reference-subject.jpg` | 主体 | [随本仓库发布](images/minimax-h3-reference-subject.jpg)——用 vpipe 自己的 FLUX.2 文生图流水线做出来的，所以不附带任何许可问题 |
| `minimax-h3-text-to-video.mp4` | 镜头运动**以及**音轨 | [第 2 步](#step-2--text-to-video-and-audio)写出来的那个文件。带音频的视频仍然是**一个**参考、同时承载两者，标记为 `<Video 1>` 和 `<Audio 1>`，无论你怎么喂给它 |

它走的是**端口**路线，用普通的 stage 来准备每一个参考：

```
load-image → image-resample(1024×1024)                              → ref1
load-video ┬→ video-to-rgb → image-resample(1344×768) → temporal-stack → ref2
           └→ audio-to-pcm(32000, stereo) → temporal-stack             → ref3
```

列表只要三行的事，这里用了十七个 stage，而它换来一件东西：两个
`image-resample` 的尺寸由你决定。端口默认 `short_edge: 0`，所以编码器直接
采用递给它的东西，而不是重新解析一遍尺寸——改这些数字，参考的几何尺寸就跟着
改。如果你更想要那三行，把整条链换成一份 `references` 列表即可；这个 stage
两种都收。

这个形状里有四个细节是承重的。**一个 `load-video` 同时喂两条流**，所以视频
和它的音轨依然是从同一个容器里一起出来的——`references` 列表赖以成立的同步
论证在这里被保留了下来，而不是被交换掉。编码器上的 **`attach_audio: [3]`**
让 **ref3** 的音频——也就是 4 号输入端口上的 PCM、第三个参考——成为它前一个
参考的音轨，这样它仍然是一对 `<Video 1>` + `<Audio 1>`，而不会变成第三个
独立的参考。这个数字是 `1..6` 里的**参考**编号，不是输入端口的索引：ref1
是 2 号输入端口。**`channels: 2`** 承载真正的立体声，`references` 路线同样
如此，而单声道的链条会悄悄放弃它。至于视频堆叠器上的 **`max_mb: 384`**，
这个尺寸是有意设的：第 2 步写出的 124 帧视频在 1344 × 768 下是 366 MiB，
超过了默认的 256 MiB，否则会把这一组截到 86 帧并警告。注意 384 留下的余量
有多小——更长的参考视频需要更高的上限，或者在 `load-video` 上加一个
`duration_s`。

**实测**：这条链产生的请求，在每一个存在的计数上都与 `references` 列表完全
一致——2 个参考、3,115 行条件信息、13,120 行参考视频、130 行音频、一条
22,615 行的打包序列。在同一个种子下，它并不是同一段*视频*：缩放现在发生在
`image-resample` 里，而不是在编码器内部，而两套实现落在相同的尺寸上，并不
意味着落在相同的字节上。如果你要的是与文件列表逐位等价、而不是尺寸控制，
那就把两处重采样保持在源尺寸，并在视频堆叠器的 `sideband` 里写
`short_edge: 768`。

有一处低效值得知道：编码器会把视频截断到 `frames`，所以这条链会把参考视频
全部 124 帧都缩放一遍，然后只用 39 帧。`references` 列表是先截断、只缩放
它保留的那些。如果这一点要紧，就去限制源，而不是抬高 `max_mb`。

```sh
cd ~/vpipe-work                                    # the work directory again
cp ~/src/vpipe/docs/pipelines/minimax-h3-reference-to-video.vpipeline .
cp ~/src/vpipe/docs/images/minimax-h3-reference-subject.jpg .
~/src/vpipe/build/apps/vpipe/vpipe --launch minimax-h3-reference-to-video.vpipeline
```

如果你是在这个目录里跑的第 2 步，那个 `.mp4` 已经就在旁边了；任何带音轨的
视频都可以。如果你给 Ref2VA 目录起了别的名字，就把 `model-select` 指向它，
另外注意两处 `frames` 设置——编码器的和 `generate-video` 的——它们已经都是
39，而且必须一致。

<a id="what-it-costs"></a>
#### 它的代价

**Ref2VA 的代价比它的输出看起来要大**，示例的尺寸是围绕这一点设计的，而不是
围绕它产出的那段视频。一个参考会被打包进与被生成物**同一条**序列里，而参考
视频还会落到它自己的画布上：第 2 步那个 960 × 544 的文件会被解析成
**1344 × 768**，比输出的像素还多。**实测**于 16 GB 的 M5，请求 56 帧：

| | 行数 | |
|---|---|---|
| 参考视频 | 18,160 | 58% |
| 条件信息（文本 + 视觉） | 4,131 | 13% |
| **正在被生成的内容**（960 × 544，56 帧） | 8,856 | 28% |
| 参考音轨 | 188 | <1% |

31,335 行需要约 6.3 GB 的激活暂存空间，而 `generate-video` 在 16 GB 的机器
上会在这个尺寸**拒绝**运行，而不是去抖动——被 wire 住的 Metal 缓冲区无法被
换出，所以超额提交会让整台机器倒下，而不是让一个 stage 失败。

所以杠杆是 `frames`，不是画面尺寸：参考视频会被截断到被生成的长度，所以缩短
输出也就把参考一起缩短了。降到 39 会同时移动两半并且装得下——22,615 行，
其中 13,120 行是视频。只把 `width`/`height` 改小，只会削掉那 28%，那 58%
一动不动。

**另一个**杠杆是参考画布本身。那个 1344 × 768 是对一个 960 × 544 文件的
*上采样*——1.98 倍的像素，从同样的信息插值出来——拒绝它是值得测量的。**实测**
于一台空闲的 16 GB M5，同一种子下背靠背运行，用的是示例构造的那个双参考
请求：

| | `768`（默认） | `0`（视频自身尺寸） |
|---|---|---|
| 参考视频画布 | 1344 × 768 | 960 × 544 |
| 每个分块的 VAE 图块数 | 28 | 15 |
| 参考行数 | 13,120 | 7,144 |
| **条件信息行数** | **3,115** | **2,119** |
| 打包序列 | 22,615 | 15,643 |
| 墙上时钟 | 23 m 07 s | 14 m 17 s |

**它会改变三样东西，而第三样会让人意外。** VAE 编码和参考行数是显而易见的
一对。第三样是*条件编码器*：视觉塔拿到的是同一批归一化像素，并从它们出发做
smart-resize，所以这个画布也决定了一段视频贡献多少视觉 token——这里少了
996 个，全都是视频的（那张静帧按它自己的短边编码，不受影响）。降低它是对
**两条**通道同时做的保真度决定，不是白得的 1.6 倍。

而在随附的这个示例上，它明显不是免费的。在 `768` 下，生成出的猫穿着参考
静帧里那件海军蓝外套，坐在提示词要求的音乐厅里；在 `0` 下，同一种子，它
没穿外套，场景也朝参考*视频*那间暗房塌陷过去。不同的序列形状给出不同的
采样，所以一个种子不能证明存在系统性的质量损失——但它足以证明 `0` 是另一次
生成，而不是同一次生成的更省写法。

所以默认值仍然是 `768`：这是发布的模型文件自己的规则，而模型被训练成看到
上采样后的参考，它给出的也就是那样的结果。在参考本身已经小于 768、预算又
紧张的场合，`0` 值得*试*——而且在保留它之前值得**并排对比**。无论哪种方式，
总像素数上限对大于画布的视频依然生效。

**你该调哪个设定取决于路线。** `reference_video_short_edge` 是按种类的默认
值，而它只会在编码器必须自己解析尺寸的参考上被查询——也就是来自 `references`
**列表**的参考。走端口的参考到达时就已经带着 `short_edge: 0`（那是端口的
默认值，也正是端口的意义所在），所以按种类的默认值根本到不了。在**这个**
示例上——它全程走端口路线——上表里 `0` 那一列是你在 `size-clip` 里把视频
重采样到 960 × 544 得到的结果；在编码器上设 `reference_video_short_edge`
不会改变任何东西。`reference_image_short_edge` 和那张静帧同理。

按随附配置——960 × 544、39 帧、8 步、一张静帧加一段视频——在无风扇的 16 GB
M5 上是 **23 min 54 s**（`references` 列表在相同几何尺寸下实测 23 min
07 s：端口链会缩放每一个解码出来的帧，而列表只缩放它保留的那些）。其中大约
三分之一发生在第一个去噪步之前：32B 的条件编码器要加载并流式读取，而两个
参考都要被读两遍，一遍由视觉塔按它自己的画布读，一遍由视频 VAE 按
MiniMax-H3 的画布读。从这里再抬高 `frames`，参考行数会跟着一起涨，所以
下一档尺寸比看上去跳得更大。

<a id="how-a-reference-is-read"></a>
#### 一个参考是怎样被读取的

单个路径可以不加方括号、直接裸写。**顺序就是编号**：它决定了模型读到的
提示词里各个参考的编号，并把它们放在一个共享的时钟上，所以重排列表就是
另一次生成——而端口参考在最后一个文件之后接着往下编号。

**标记里的编号是在它自己的种类内部计数的，不是跨整个列表计数。**
`<Picture i>`、`<Video k>` 和 `<Audio j>` 各有自己的计数器，所以整体上的
第三个参考完全可以是 `<Audio 1>`——它是第一条*音轨*，不管它前面排着什么。
两张静帧加一条音轨，按这个顺序，呈现为：

```
<Picture 1>: … <Picture 2>: … <Audio 1>: … <your prompt text>
```

而一张静帧后面跟一段带声音的视频，则是 `<Picture 1>`、`<Audio 1>`、
`<Video 1>`——这段视频是第二个参考，但仍然是第一个视频和第一个音频。把这件
事搞反，写出的提示词就会指向一个并不存在的参考，而且没有任何东西会报告它：
对模型来说，一个对不上的标记就是普通文本。

一个参考自己的块是按列表顺序发出的，提示词文本排在所有块之后，所以你在
提示词里的哪个位置提到某个标记是自由的——这些标记回指的是模型已经读过的块。

正是这个顺序性，决定了这里为什么是一份列表、而不是一个参考一个端口。一个
请求的形状只有在它到达时才知道，而十二条 `load-image` 链无法表达「三段视频
加九张静帧」，除非每来一个请求就重写一次图。那六个端口是同一论证的另一半，
不是对它的推翻：一个*路径*无法指称一张还不存在的静帧。是六个而不是更多，
因为编号必须是静态的——一个可能沉默的端口会让它后面的每个参考都重新编号。

**文件不会说明自己是什么。** vpipe 打开它、从解码结果里读出这一点：只有一帧的
容器是*图片*参考，不带视频流的 `.mp4` 是*音频*参考，而一个动画 `.webp` 是
视频。扩展名只是一种声明，实际解码结果才是事实——而文件选择器递过来的是用户选中的
任何东西。搞错了并不导致报错：一张静帧被当成视频读，会让模型以一段凝固的视频
为条件；一段视频被当成静帧读，则会悄悄只保留它的第一帧。张量输入没有这种歧义，
这也是端口按维数定类型的原因之一：`[1, 3, H, W]` 是一段单帧视频，
`[3, H, W]` 是一张静帧，不用去猜。

一个视频参考在自己带音轨时，会以**它自己的音轨**为条件，而视频和它的音频
必须从同一个容器里一起出来才能保持同步。`references` 列表通过自己打开文件
做到这一点；端口路线则靠一个 `load-video` 同时喂它的两条流。无论哪种方式，
文件的**帧率**都必须在这一路上存活下来：MiniMax-H3 会把每个参考重采样到它
自己的 24 fps，所以在输入途中丢掉的帧率，意味着一次以错误速度为条件的生成，
而且没有任何东西会抱怨。这就是为什么端口上的视频必须声明 `fps`，缺了就会
被拒绝。

把这个 stage 的 `frames` 设成与 `generate-video` 的**同一个值**：它既是
参考被截断到的时长，也是 transformer 打包的那条序列的大小。两者会互相校验，
而不是互相信任。

参考从不决定生成物的几何尺寸。一张图片按它自己的短边编码
（`reference_image_short_edge`，2048），没有总像素数的上限，也包含上采样；一段
视频落到与目标相同的画布规则上（`reference_video_short_edge` 为 768，受
`reference_video_max_pixels` 1032192 约束），并从*它*的宽高比解析出来。两个
形状不同的参考会落到不同的画布上，这是预期行为。

这些默认值是发布的模型文件自带的，而它们全都值得了解，因为没有一个能在输出
里看出来。`reference_image_short_edge` 尤其如此：在 2048 下，一张静帧就是
121 个 VAE 图块，而允许九张；在 1024 下则是 25 个。任何通过 `references`
列表喂入静帧的图，都应该把它调低。

这两个键都是按种类的默认值，所以它们只作用于 `references` 列表——从端口到达
的参考已经自带 `short_edge`，永远到不了它们。示例改用一个 `image-resample`
stage 把静帧限制在 1024，这是端口路线的等价做法，也是它同时还带着一个冗余
的 `reference_image_short_edge: 1024` 的原因。

**编码器每改动一个参考的形状，都会输出文字警告**，并说明
保留下来的是什么：

```
reference 1 fitted -- rescaled 1920x1080 -> 1344x736 (48% of the pixels),
  resampled 30 -> 24 fps (18 frame(s) dropped),
  truncated 72 -> 39 frames (54% of the clip)
```

三种削减，特意分开计数，因为它们的对策不同：帧率重采样无论视频是否同时太长
都会丢掉整帧，截断要靠 `frames` 来解决，而缩放要靠画布相关的键来解决。一个
什么都不需要改的参考会以 debug 级别记为 *taken as given*——当你自己把尺寸
设好之后，这就是应该追求的结果。

<a id="references-that-are-not-files"></a>
#### 不是文件的参考

六个**张量输入端口**（`ref1`..`ref6`）排在 `prompt` 和 `model` 之后，服务
于文件列表无法指称的那些参考：你的图刚刚生成出来的一张静帧、一个裁剪出来的
画面、一段从未写到磁盘上的视频。它们是对 `references` 列表的补充而不是替代，
编号排在它之后。

| 你发送什么 | 维数 | sideband |
|---|---|---|
| 音频 | `[N]` 或 `[channels, N]` f32 | `sr`（或 `sample_rate`）——**必需** |
| 一张图片 | `[3, H, W]` u8 | — |
| 一段视频 | `[frames, 3, H, W]` u8 | `fps`——**必需** |

维数决定类型，这也顺带解决了容器唯一说不清的那种情况：单帧视频是
`[1, 3, H, W]`，静帧是 `[3, H, W]`，而这两者是不同的请求。速率是**必需的、
从不取默认值**——以错误速率读取的音轨会以错误的声音为条件，而以错误速度读取
的视频会生成出没有任何东西可抱怨的画面。

**发送音频时用音频 VAE 的速率：32000 Hz。** 文件参考会被直接解码到这个速率
上，但一段*节拍*到达时用的是它的生产者选的速率，所以要把生产它的
`audio-to-pcm` 的 `output_sample_rate` 设成 `32000`。别的速率并不会悄无声息
地出错——编码器会把它重采样到 32000 并警告——但那是对一段已经被生产者重采样
过一次的波形再走一遍滤波器，而修法只是一个配置键。这件事之所以重要，是因为
下游不会再有任何东西去看这个速率：一段*未被统一*的 44.1 kHz 音轨会被当成
32 kHz 来编码，也就是快了 1.38 倍、音高升高了四度，并且相对于与它共享同一个
旋转时钟的视频长了 1.38 倍——而所有形状依然合法。

一个可选的 sideband 键：`short_edge` 设定*这一个参考*的画布。

当速率不由你决定时——PCM 来自一个不是你配置的 stage，或者一条馈送必须同时
服务这个端口和一路 44.1 kHz 的混流——就在它前面放一个
**`audio-temporal-resample`**，并设 `output_sample_rate: 32000`。如果参考在
被模型听到之前还需要拉伸，那个 stage 同时也掌管音轨的*速度*和*音高*。

<a id="audio-that-belongs-to-a-clip"></a>
##### 属于某段视频的音频

一段被解复用成帧和 PCM 的视频会到达**两个**端口，但应该保持为**一个**参考。
编码器上的 `attach_audio` 指明哪些端口上的音频是音轨、而不是独立的参考：

```json
"attach_audio": [3]
```

它是**按位置**的：音频会折叠到紧排在它前面的那个参考上。只有一段视频时这是
无歧义的；有两段时，就要把端口排成让每条音轨紧跟自己的那段视频。没有办法
指名一个目标。

它是一个列表而不是一个单独的开关，因为一个请求完全可能同时带一条附属音轨和
一段独立的音乐，而这是两个不同的参考。

**它可以附着到 `references` 列表里的参考上。** 文件是在任何端口之前读取的，
所以它折叠上去的那个参考可能就是其中之一——这使得 `references:
["motion.mp4"]` 加上一个端口上生成出来的音轨成为一个完全合理的请求，也是给
你已有的一段视频配乐的最利落的办法。

一个音频节拍自己的 `attach` sideband 会为该节拍覆盖配置，这样一个比图更清楚
的生产者可以把这件事说出来——`true` 表示在配置没要求的地方附着，`false`
表示在配置要求了的地方拒绝。

附着到一张**静帧**上是允许的，并且会警告。这是一个真实的请求——一个带音频轨
的单帧 `.mp4` 通过 `references` 列表进来时就是这样——但更常见的情况是端口
接线顺序错了，所以这次运行会把它说出来，而不是拒绝另一条路线所允许的东西。

**一个接上线的端口必须在每个请求上都有产出**；要表示「这次什么都没有」就
发一个空张量。一个可能沉默的端口会让它之后的每个参考都重新编号，而编号就是
请求本身。

端口的默认值是 `short_edge: 0`——**按它到达时的尺寸编码**。这正是端口的
意义：如果你已经用一个 `image-resample` stage 重采样到了 768，或者裁剪到了
你自己选定的取景，那么拿按种类的默认值去重新解析一遍，会把它直接放大回去、
把你做的工作全部抹掉（还要额外付一次 Lanczos 的代价）。在 `0` 下，编码器只
可以**缩小**，绝不放大——它会把图片压到总像素数上限之下，把两个轴都向下取整到
DiT patch 和 VAE 步长要求的 32 的倍数，除此之外不去碰它。仍然有三件事它
跳不过去：那个网格（一个高 1080 的画面会变成 1056）、1:4 … 4:1 的宽高比
边界（这是拒绝，而不是去适配），以及对视频而言的 24 fps 重采样加上
`17n + 5` 对齐。

由于短边为 `0` 会移除图片参考原本唯一的边界，在喂入原始图片的图上要设
**`reference_image_max_pixels`**。它默认不设上限，与模型文件一致，而一张
不设上限的 4K 静帧约为 220 个 VAE 图块和 8,160 个 DiT 行——是一个典型请求
打包序列的一半，而且这还没算它的视觉 token。

这个 stage 在运行期间会同时持有提示词编码器、它的视觉塔和两个 VAE，所以在
内存受限的机器上，把 `unload_when_idle` 留在 `auto`——那些编码器会在去噪
开始之前被丢掉。

<a id="preparing-the-ref2va-checkpoint"></a>
#### 准备 Ref2VA 模型文件

完全照第 1 步那样运行
[`prepare-minimax-h3-ref2va-8bit.vpipeline`](pipelines/prepare-minimax-h3-ref2va-8bit.vpipeline)，
它会产出 `local/MiniMax-H3-Ref2VA-8bit`。把生成流水线的 `model-select` 指向
它即可。[`…-4bit`](pipelines/prepare-minimax-h3-ref2va-4bit.vpipeline)
是同一件事、只占一半磁盘，代价见[为什么是 8-bit 而不是 4](#disk-space)。

两个模式的模型**共享一个仓库和一次下载**。Ref2VA 只额外增加它自己的 66 GB
transformer；51 GB 的提示词编码器和两个 VAE 在第 1 步之后已经在磁盘上，会被
跳过。如果你只想要 Ref2VA，单独运行这个流水线就行——它会把自己需要的东西
取回来。

有一个配置键让这种共享是安全的，它就在流水线里：

| 键 | 为什么 |
|---|---|
| `model_variant: ref2va` | 要取仓库两个模型中的*哪一个*。文件不同，仓库路径相同。不声明的抓取会被拒绝，并把两者都列出来。 |

每个模式还各带自己的**注册键**——这里是 `Comfy-Org/MiniMax-H3-Ref2VA`
——所以两条记录可以在磁盘上同一个目录之上共存，而不会互相覆盖。那是模型目录
对这个条目的自有名称，所以 `model_key` 只在需要覆盖它时才必须设置。

之后下游的一切都通过那个键来解析，而不是靠检查目录——这一点很重要，因为那个
目录里放着**两个** transformer，它说不出你指的是哪一个。放任它去猜，它会挑
FL2VA，而一个 Ref2VA 请求就会加载成功、以完整 33B 的代价运行，然后生成一段
不以任何东西为条件的视频。

<a id="ref2va-like--references-on-the-fl2va-weights"></a>
#### 类 Ref2VA——在 FL2VA 权重上使用参考图

第二个模型文件并不是非有不可。**FL2VA** 权重同样能接收参考图片，走的是
Ref2VA 自己的那条序列，而不是关键帧。OpenVDN 在 2026-09-17
[公布了这种用法](https://github.com/OpenVDN/vdn-minimax-h3#supporting-ref2va-like-task)
并给出了渲染好的示例。vpipe 会运行它：照上面那样接一个 `video-ref-encoder`，
再把 `model-select` 指向一份 FL2VA 模型文件，而不是 Ref2VA 的。

[`minimax-h3-ref2va-like.vpipeline`](pipelines/minimax-h3-ref2va-like.vpipeline)
就是这张图，上面还挂了 [VDN 线性分支](#faster-attention--the-vdn-linear-branch)
和 [Turbo LoRA](#fewer-steps--the-turbo-lora)——这正是上游渲染时用的组合，而这
个组合正是靠这种用法才成为可能：**VDN 分支只在 FL2VA 上发布过**，所以在此之前，
线性注意力和参考图没法同时拥有。

**它到底是什么。** 权重什么都没变。参考图按 Ref2VA 的打包方式打包——
`[text | 每个参考一个 block | 目标音频 | 目标视频]`，每个参考占一个旋转位置槽，
各自保持在接近干净的噪声增强水平上——然后由 FL2VA 的 transformer 读取，而不是
MiniMax-H3 那个独立的 `transformer_ref`。两个模式随附的 transformer 配置逐字节
相同，所以这条序列两边都构建得出来；不同的只是由哪一份权重来读它。

> **这是零样本能力，报告结果时请照实说。** 这是 FL2VA 模型文件碰巧具备的能力，
> 而不是它训练过的任务。Ref2VA **就是**那个任务，权重也是为它训练的，当主体
> 还原度才是重点时，它仍然是更好的答案。每一次用到这种用法，`generate-video`
> 都会在日志里点明模式，因为它出问题的方式是：生成的片段悄悄忽略了参考图，
> 而看上去完全正常。

**和一张 Ref2VA 图相比有三处不同**，其中只有第一处需要你做什么：

- **只用图片。** 上游这种用法针对的是参考*图片*。参考视频或音轨也能打包、也能
  跑通——布局是同一套——但没有任何已发布结果覆盖它，所以 vpipe 会发出警告。
  视频和音轨是 Ref2VA 模式上训练过的输入。
- **短边 768，而不是 2048。** 这是上游的配方，vpipe 会自动采用：不设置时，
  `reference_image_short_edge` 在 **FL2VA 上是 768**，在 Ref2VA 上是 2048。
  它是每个参考四分之一的 token 数，而这些行处在 DiT **每一步**都要重读一遍的
  序列里——所以这个差别不是一个随便放在哪儿都安全的保真度旋钮。一旦设置了这个
  键，两个模式下都以设置为准。
- **仍然不能用关键帧。** 参考图和关键帧锚点依然互斥，而这一点从来就与模式无关：
  无论由哪份权重来读，参考布局里都没有放锚点的槽位。`generate-video` 会明确
  指出这一点，而不是悄悄把锚点丢掉。

<a id="longer-clips--one-story-in-four-parts"></a>
### 更长的片段——一个故事分成四部分

单次生成的开销随长度快速上升：每一帧都会给打包后的序列增加行数，而注意力的开销
按行数的平方计算。要得到 **40 秒**的片段，跑四次普通的 10 秒生成是更便宜的做法。
第一次之后的每一次都会把**前一次的尾段**作为要续接的片段交给 Ref2VA，于是故事由
画面和声音向前承接，而不是只靠提示词：

- **[`minimax-h3-extend-part1.vpipeline`](pipelines/minimax-h3-extend-part1.vpipeline)**
  ——在 **FL2VA** 模型文件上由文本生成视频与音频，243 帧（10.125 s），960 × 576。
  它写出 `minimax-h3-extend-part1.mp4`。
- **[`…-part2`](pipelines/minimax-h3-extend-part2.vpipeline)** /
  **[`…-part3`](pipelines/minimax-h3-extend-part3.vpipeline)** /
  **[`…-part4`](pipelines/minimax-h3-extend-part4.vpipeline)**
  ——**Ref2VA**，每一部分都以前一部分最后 3.75 s（90 帧）的画面和声音为条件。各 243
  帧。
- **[`…-concat`](pipelines/minimax-h3-extend-concat.vpipeline)**——把四部分合成
  一个文件，在 vpipe 内完成而不是手工拼接。见[拼接各部分](#joining-the-parts)。

每一次续接都是同一张流水线图，只是换了提示词和要读的文件，所以第五部分就是第四
部分的一份副本。

**随附的流水线图以 21 步运行官方发布的权重，不加适配器。**每一部分都指向发布者
自己的模型文件（见[发布的权重，两个模式都有](#the-released-weights-either-partition)）：

| | 模型文件 | 步数 | shift 值 |
|---|---|---|---|
| 第 1 部分 | `MiniMaxAI/MiniMax-H3-FL2VA` | 21 | 12 / 3 |
| 第 2–4 部分 | `MiniMaxAI/MiniMax-H3-Ref2VA` | 21 | 12 / 3 |

这是以时间换质量的选择，而一条链正是它值得的地方。Turbo 适配器换来更少的步数，代价
是牺牲一部分基础模型对构图的理解——某一部分可能返回主体位置很糟的画面、一只画错的
手，或者干脆漏掉提示词要求的某个节拍——而在一条链里，之后的每一部分都会从前一部分
出错的地方接着往下走。这些流水线图也关闭了 `sol_attn`；`i8_gemm` 在每一部分都保持
开启。

**想让这条链跑得更快**，就把随附流水线图省掉的东西放回去：准备好的 8-bit 重打包
（在 `model-select` 上用 `local/MiniMax-H3-FL2VA-8bit` 和
`local/MiniMax-H3-Ref2VA-8bit`）、每个 `generate-video` 上的 `sol_attn: true`，以及
在 `minimax-h3-model-config` 上为每个模式各加一个 Turbo 适配器——每个模式用**它自己
的**适配器，因为适配器是针对单一任务蒸馏出来的，模型目录中的 parent 链接会拒绝另一
个模式的适配器（见[哪些 Turbo 适配器可用](#which-turbo-adapters-work)）：

| | `lora` | `steps` | `video_shift` |
|---|---|---|---|
| 第 1 部分 | `larryvrh/MiniMax-H3-Turbo-Lora-v4-600-ema` | 8 | 12 |
| 第 2–4 部分 | `lightx2v/Minimax-h3-Turbo-ref2va-8step-768p` | 8 | **6.0** |

这里只有 Ref2VA 的适配器需要 `video_shift: 6.0`。

当某一部分返回的构图不理想时，在重写提示词之前先**换一个 `seed`**：它的改动更小，
而且常常就够了。流水线图的其他部分都不变，它之后的各部分仍然只读它的尾段。

随附的故事是*钟表匠的鸣鸟*：

| 部分 | 发生了什么 | 结束于 |
|---|---|---|
| 1 | 在一间烛光摇曳、四壁挂满时钟的工作室里，一位年老的钟表匠给一只黄铜鸣鸟上弦，低声说 *"Just one more turn,"*（再上一圈就好），并把它托在手指上。 | 她静静地举着，鸟一动不动 |
| 2 | 鸟醒了，唱起歌，在所有时钟一齐报时声中飞了一圈；她笑着说 *"You remembered the song!"*（你还记得这首歌！）它落在结霜的窗台上。 | 鸟仍停在窗台上，望着雪 |
| 3 | 她拨开窗闩，推起窗扇；雪吹进来，鸟跳上敞开的窗框回头张望。*"Go on, then. It's your sky."*（去吧。那是你的天空。） | 鸟立在窗框上，翅膀半张 |
| 4 | 它跃入夜色，越过屋檐向上攀升，在她从亮着灯的窗口注视时盘旋一圈。*"Goodnight, little one."*（晚安，小家伙。） | — |

<a id="running-the-chain"></a>
#### 运行这条链

你需要官方发布权重的**两个**模式：把 `MiniMaxAI/MiniMax-H3` 获取两次，分别用
`model_variant: fl2va` 和 `ref2va`（见
[发布的权重，两个模式都有](#the-released-weights-either-partition)）。若要改用在步骤 1
和[准备 Ref2VA 模型文件](#preparing-the-ref2va-checkpoint)中准备好的重打包，就按上文
所述修改每一部分的 `model-select`。按顺序运行各部分，并且在同一个工作目录下——每一
部分都按相对名称打开前一部分的 `.mp4`：

```sh
cd ~/vpipe-work                                    # the work directory again
cp ~/src/vpipe/docs/pipelines/minimax-h3-extend-part*.vpipeline .
for n in 1 2 3 4; do
  ~/src/vpipe/build/apps/vpipe/vpipe \
    --launch minimax-h3-extend-part$n.vpipeline || break
done
```

**一次只跑一个**，绝不要同时跑：每一个都持有一个 33B transformer（见
[内存](#memory)）。`|| break` 很重要——输入缺失的那一部分不会产生任何输出，而下一
部分就会以一个过期的文件为条件，而不是直接失败。

这四个部分文件是 **FFV1** 编码，QuickTime 和大多数浏览器都无法播放；它们是给这条链
读取的，不是用来观看的。要看就看拼接后的片段，或者用 `ffplay`、VLC 打开某一部分。

<a id="how-a-part-takes-its-guide"></a>
#### 一部分如何取得它的引导片段

```
load-video ─> video-to-rgb(u8) ─> temporal-slice(start −90)
                                ─> temporal-stack ─────────────> ref1
load-audio(start_s 6.375, duration_s 3.75) ─> audio-to-pcm(32000, stereo)
                                           ─> temporal-stack ────────────> ref2
```

**引导片段按解码出来的样子原样交给模型。** 切片直接接到堆叠——中间不做重采
样，也不做色阶调整。上一部分产出的是什么，下一部分就从什么上面往下接。

**续接从哪里接上，由提示词决定。**续接应当从引导片段的**末帧**往下走，而用随附的
提示词，每一部分都是如此：各部分之间的衔接没有跳变。

你自己的续接提示词请照这些示例的写法来写——把延续下来的主体一一命名并声明其被保
留，`summary` 以 `[video continuation + audio reference]` 开头，并给镜头一条明确的
时间线，说明什么时候发生什么。如果某处接缝确实出现跳变，**请改提示词**，而不是去修
正画面：针对某一条接缝调出来的重采样或提亮，换一条接缝就不适用了。

**各部分以无损方式写出，因为下一部分要读取它们。**每一部分的 `rgb-to-video` 输出
`full` 范围、标记为 `bt709` 的 `yuv444p`，它的 `save-video` 再用
`video_codec: ffv1` 和 `audio_codec: alac` 编码。于是续接读到的引导片段就是解码出来
的画面本身——色度没有被减半成 4:2:0，没有有损量化，用满 256 个码值而不是 219 个——
无论链条有多少部分，都不会有任何损失逐级累积。唯一的有损编码是最后的拼接。在
960 × 576 下，一个 10 s 的部分约 80–93 MB；它的 sink 上所带的 `video_bitrate` 对
FFV1 不起作用。

**色彩范围是被带着走的，而不是靠约定猜的。** `save-video` 会给写出的视频流打
上标记，`video-to-rgb` 会把这个标记读回来，因此一个片段经过解码、参与条件生成
再重新编码之后，对比度不变。这一点在这里尤其要紧，因为续接每做一部分就要把自
己的输出重新读一遍，而一个没有标记、又被按错误约定读取的文件，每经过一跳就会损
失 219/255 的对比度。

**`attach_audio: [2]`** 把那条声轨折叠到它前面的片段上，使模型读到的是一个
`<Video 1>` 及其 `<Audio 1>`，而不是两个互不相关的参考（见
[属于某个片段的音频](#audio-that-belongs-to-a-clip)）。

**为什么是 90 帧。**编码器会**从开头起**把参考片段向下取整到整数个 `17n + 5` 帧，
所以任何其他长度的引导片段都会丢掉它**最后**那几帧——恰恰是续接本该接上的那一刻。
24 fps 下两秒是 48 帧，会被截成 39 帧。90 是 `17 × 5 + 5`，所以什么都不会被丢掉；
而 3.75 s 比最短的整齐长度 56 帧（2.33 s）把更多的运动带过接缝。它编码为 27 个潜
变量帧、每帧 540 个单元：**14,580** 个参考视频行。MiniMax 记载的最短参考片段是
**2 s**。

**`reference_video_short_edge: 0`** 让引导片段保持在它自己的 960 × 576 画布上，也
就是输出本身的尺寸，而不是把它放大到 1344 × 768。这样参考就保持在 14,580 行，而不
是大约两倍于此。这就是[代价是什么](#what-it-costs)中描述的那个取舍，而且还没有在
这个故事上做过并排对比。

**`video-ref-encoder` 上的 `max_prompt_tokens`** 从默认的 16,384 调高了——第 2 部
分为 **65,536**，第 3、4 部分为 **131,072**。它是条件编码器序列长度的上限：提示词
加上视觉塔对引导片段的解读；超出时编码器会直接拒绝，而不是截断。它的 KV 是分页的，
按实际编码的 token 计每个约 200 KB，所以上限设得高并不会预先占用内存。当编码器报告
某次呈现超出其 token 池时，就把它调高。

**参考编码器和 `generate-video` 上的 `unload_when_idle: destroy`**，让每个模型在
它负责的阶段一结束就把内存交还回去。

<a id="writing-the-prompts"></a>
#### 编写提示词

这些提示词遵循 MiniMax 自己的提示词指南，那些指南描述的是模型训练时所用的格式。
这不是装饰：H3 的编码器逐字读取提示词，中间没有任何改写步骤。

- **第 1 部分**使用
  [基础指南](https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/docs/VIDEO_PROMPT_WRITING_GUIDE_base_en.md)
  的三个字段：`integrated_multimodal_description`、`overall_soundscape` 和
  `non_diegetic_music`。镜头标记为 `[Shot N] At MM:SS.mmm`，说话人有一个稳定的
  ID，她的台词写作 `<d>[English] Just one more turn.</d>`。
- **第 2 到 4 部分**使用
  [参考指南](https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/docs/VIDEO_PROMPT_WRITING_GUIDE_ref_en.md)
  的六个小节。`subject_definitions` 把钟表匠、鸣鸟和工作室命名为 `<Video 1>` 的
  `<Subject 1..3>`，并以 `<Audio 1>` 作为它的声轨。`summary` 以
  **`[video continuation + audio reference]`** 开头，这个任务类型表示该片段是要被
  续接的，而它的声音要被延续而不是被复制。随后是 `retention_analysis` 和逐镜头的
  `detailed_description`。

**除最后一部分之外，每一部分都结束在一个静止镜头上，而下一部分就从它开始。**第 1
部分从 6.0 s 起以一个静止的中近景收尾，其间除了烛焰和钟摆什么都不动；第 2 部分从那
个取景开始，并在鸟醒来之前保持两秒；第 2 部分反过来又结束在鸟静止于窗台上，依此沿
链而下。每一段引导片段都**结束**在这些静止段之内，所以每一处接缝都落在画面静止的地
方。3.75 s 的引导片段并不总是**开始**于静止段之内：第 2、3 部分分别在 7.5 s 和
7.0 s 切到最后一个镜头，所以第 3、4 部分读到的引导片段开头是前一个镜头的最后片刻
——这次切换也是模型看到的内容之一。从一开始就照这个思路规划故事：值得生成的节拍是
那些**接缝之间**的部分，而一段至少和引导片段一样长的静止，能让整段引导都落在它上面。

**为下一部分要用到的东西预先埋好伏笔。**第 3 部分要打开窗户，所以第 2 部分的提示词
给窗框加了一个能抬起窗扇的小黄铜窗扣，让窗户在它自己的镜头里一直关着，并让鸟落在
窗扣旁边。这样第 3 部分读到的引导片段里就已经有它要用的东西；一个直到需要它的那一
部分才第一次出现的细节，模型只能当场去编。

**逐字沿用主体定义。**第 2 到 4 部分用完全相同的词句描述那个女人、那只鸟和那个
房间，因为片段展示了它们而文本又命名了它们——措辞上的任何漂移，都是模型可以随意
施加到画面上的漂移。有些漂移还是会出现；见下文。

**Ref2VA 是在片段之后续接；它不会重放片段。**第 2 部分从第 1 部分**最后**一帧之后
的那一刻开始，而不是从引导窗口的第一帧开始。这是模型从任务类型中学到的续接，而不是
一个被钉住的帧。
[上面的注意事项](#conditioning-on-references-ref2va)仍然成立，这也是要在一个不动
的时刻上交接的又一个理由。

<a id="joining-the-parts"></a>
#### 拼接各部分

**各部分是前后相继的，所以要首尾相接地拼起来，什么都不裁掉。** 续接是在引导片段
**之后**继续，而不是把它重新渲染一遍，所以第 2 部分的第一帧接在第 1 部分的**最后**
一帧之后——而不是接在 3.75 s 之前、引导片段被切出来的那一帧之后。不存在重叠的素
材，因此也没有什么可以在其中剪切的。

**在 vpipe 内**，那就是
[`minimax-h3-extend-concat.vpipeline`](pipelines/minimax-h3-extend-concat.vpipeline)：
一个 `load-video` 拿到全部四个文件，并按给定的顺序把它们接起来。

```json
"config": {
  "input_url": [
    "minimax-h3-extend-part1.mp4",
    "minimax-h3-extend-part2.mp4",
    "minimax-h3-extend-part3.mp4",
    "minimax-h3-extend-part4.mp4"
  ],
  "enable_video": true,
  "enable_audio": true,
  "options": {
    "safe": "0"
  }
}
```

**在 web UI 里这就是一个文件选择器。** 在编辑器中打开这条流水线，点 `input_url`
上的「浏览」，在同一个对话框里把四个部分一次选中——它们会按选中的顺序落进数组。
不需要写列表文件，也不需要指定解复用器。

> **不要在拼接时裁剪各部分。** 在每一部分的引导片段起点处把它裁断看起来很自然
> ——那正是下一部分所依据的那一刻——但这样会**跳过该点与这一部分结尾之间已经渲染
> 好的画面**，按随附的引导片段计，每一处接缝丢掉 3.75 s 的故事，而这个跳跃是看得见
> 的。

```sh
vpipe --launch minimax-h3-extend-concat.vpipeline
```

这张流水线图就是普通的文件链——`load-video → video-to-rgb → rgb-to-video →
save-video`，声轨经过 `audio-to-pcm`——所以这次拼接就是按 sink 所设的
`video_bitrate` 重新编码一次，画面与声音在每一处边界上都保持锁定。这是整条链中唯
一有损的一步：采用 sink 的默认设置，H.264、4:2:0、有限范围，配 AAC 声音，也就是每
个播放器对一个交付文件的预期。`start_s` / `duration_s` 针对的是接好之后的时间线，
而不是其中某一个部分。

> **什么时候仍然需要列表文件。** 数组拼接的是完整的片段，这里正是这种情况。如果
> 想在拼接的**同时**裁剪每个片段——给每一项加 `inpoint` / `outpoint`——就要自己写
> 列表文件，并用 `format: "concat"` 指定解复用器；这时它还需要
> `options: {"safe": "0"}` 才愿意跟随绝对路径。把 `format` 和数组同时设置会报错，
> 而不是让某一方悄悄胜出。

结果是 **40.5 s**——四个部分完整保留，972 帧。

**代价是什么。** 每一部分静止的收尾和下一部分静止的开头都被保留了下来，所以一处
接缝会在它那个静止的节拍上停留两者的时长，而不是只停留其中之一。这种更长的停留就是
「什么都不丢」的代价，而且通常是划算的一笔交易：停顿看起来像是有意为之，跳帧看起来
则像是出了毛病。如果某一处接缝确实显得拖沓，就在那里裁掉**几**帧——绝不要裁掉整段
引导片段的长度，那样等于把跳跃又放了回去。

**手工拼接**时，如果接缝处想要交叉淡化而不是硬切——流水线里没有做这件事的
stage——就在每个接合处用五帧来淡化。各部分完整为 10.125 s，于是每个 `offset` 都是
「目前已构建的时间线减去淡化时长」：

```sh
ffmpeg -i minimax-h3-extend-part1.mp4 -i minimax-h3-extend-part2.mp4 \
       -i minimax-h3-extend-part3.mp4 -i minimax-h3-extend-part4.mp4 \
  -filter_complex "\
[0:v]setpts=PTS-STARTPTS[v0];[1:v]setpts=PTS-STARTPTS[v1];\
[2:v]setpts=PTS-STARTPTS[v2];[3:v]setpts=PTS-STARTPTS[v3];\
[v0][v1]xfade=transition=fade:duration=0.208:offset=9.917[a01];\
[a01][v2]xfade=transition=fade:duration=0.208:offset=19.834[a02];\
[a02][v3]xfade=transition=fade:duration=0.208:offset=29.751,format=yuv420p[v];\
[0:a]asetpts=PTS-STARTPTS[t0];[1:a]asetpts=PTS-STARTPTS[t1];\
[2:a]asetpts=PTS-STARTPTS[t2];[3:a]asetpts=PTS-STARTPTS[t3];\
[t0][t1]acrossfade=d=0.208[s01];[s01][t2]acrossfade=d=0.208[s02];\
[s02][t3]acrossfade=d=0.208[a]" \
  -map "[v]" -map "[a]" -c:v libx264 -crf 18 -c:a aac -b:a 192k \
  minimax-h3-extend.mp4
```

交叉淡化会吃掉它所混合的那些帧，所以这样得到的是 **39.9 s** 而不是 40.5。它是可选
的。

<a id="making-it-your-own"></a>
#### 改成你自己的

- **让每一部分的两个 `frames` 保持相等**，即它的 `generate-video` 和它的
  `video-ref-encoder` 上的那两个；它们会互相校验。各部分之间不必长度相同。
- **改变某一部分的长度，下一部分的音频窗口就会移动。**`start_s` 是
  `previous_part_frames / 24 − 90 / 24`。只要切片仍停在 90，`duration_s` 就保持
  3.75。
- **引导片段是 `17n + 5` 帧**：56（2.33 s）、73（3.04 s）或随附的 90（3.75 s）。改
  `temporal-slice` 的 `start`，并随之改音频窗口。更长的引导片段承载更多运动，在
  960 × 576 下每多一个潜变量帧约需 540 行。
- **保持各部分无损。**把某一部分的 sink 改回 H.264，之后的每一次续接都会以前一部分
  压缩过的 4:2:0 副本为条件——这种损失每经过一部分就累积一次。
- **要预期身份沿这条链漂移**，并针对它来写。每一部分只看到它之前的 3.75 s，所以
  引导片段没有展示的细节只能靠提示词承载。各部分之间保持描述完全一致，把可辨识的
  特征放在接缝附近的画面内，并把最后一部分与第一部分对比，而不是与它前面那一部分
  对比。
- **要加第 5 部分**，复制第 4 部分，把它的两个 `load-guide*` stage 指向
  `…-part4.mp4`，给新的一部分自己的 `output_url`，并把它加进拼接列表。
- **加上 Turbo 适配器会连带改变步数和 shift 值。**每一个都是按某个配方蒸馏出来
  的；上面的表格是那一对所需要的，而
  [哪些 Turbo 适配器可用](#which-turbo-adapters-work)列出了其余适配器及其配方。
  另一个模式的适配器会被拒绝，而不是被应用。

<a id="the-released-weights-either-partition"></a>
### 发布的权重，两个模式都有

两个模式也都从 **`MiniMaxAI/MiniMax-H3`**——发布者自己的 diffusers 检出——编入模型
目录，在那里每个模式都自成一套完整的流水线——`FL2VA/` 和 `Ref2VA/`，各自下面都有
transformer、提示词编码器和两个 VAE。选择方式相同，用 `model_variant: fl2va` 或
`ref2va`；那个仓库现在发布两个模型，所以抓取它时必须说明要哪一个。

它是更大的那个下载：每个模式约 **134 GB**，而整个重打包版约 115 GB。编码器和两个
VAE 在每个模式下是*重复*的而不是共享的，所以想从这里同时获得两个模式，就要为它们
付出两份副本——而且它的编码器提供全部 64 层，而重打包版的编码器在这个模型实际读取
的抽取点处被截断。它换来的是参照：这些就是重打包版所转换的源权重，而两者对
transformer 融合后的 qkv 投影的分组方式不同——这个差异在张量名或形状上都没有任何
标记，所以磁盘上同时有这两份，就把「我们的加载器对不对？」变成了一次 diff。

一个目录持有哪个模式，是从那个模式自己的 `model_index.json` 中读出的，而模型的
注册键则表明你要的是哪一个。

<a id="fewer-steps--the-turbo-lora"></a>
### 更少的步数——Turbo LoRA

步数是这个模型开销的主要部分，而原始模型要出好效果需要 16 步。
社区的 [Turbo LoRA](https://huggingface.co/larryvrh/MiniMax-H3-Turbo-Lora)
把它蒸馏下来：**4 步**就能得到可用的视频**以及**同步的音频，6–8 步效果更好。
超过 8 步它就不再有帮助，反而开始过度锐化，所以合适的范围是 4–8。`scale`
保持 **1.0**——这个适配器就是按它调好的。

它们大多适配 **FL2VA** 模式（文生视频与图生视频），而 lightx2v 那一系还覆盖
**Ref2VA**。模型目录里收录的不是一个而是八个适配器——共十二个条目，因为
lightx2v 把其中四个发布了两遍——因为它们之间的选择是实质性的，而不是版本号的
递进：训练分辨率不同、sigma 网格不同、模式不同、同一个文件的分解方式不同。

| 条目 | 模式 | 步数 | shift（视频/音频） | 训练分辨率 | 何时用 |
|---|---|---|---|---|---|
| `Turbo few-step v4-600 EMA`（larryvrh） | FL2VA | 4–8 | 12 / 3 | — | 默认选择。静态与小幅运动的镜头更好，微观细节更好。 |
| `Turbo few-step v1-850 EMA`（larryvrh） | FL2VA | 4 | 12 / 3 | — | 只适用于**4 步且有大幅快速运动**的情况，那时 v4 可能出现拖影或涂抹。6–8 步时优先用 v4。 |
| `Turbo 4-step v1.0 768p`（lightx2v） | FL2VA | 4 | **6** / 3 | 1344×768 | 另一次蒸馏，在 768p 上。**请设置 `video_shift: 6.0`**——见下文。 |
| `Turbo 4-step v1.2 768p`（lightx2v） | FL2VA | 4 | **6** / 3 | 1344×768 | 那一系里最新的，也是 4 步时该先试的。同时发布了**拆分（split）**版和融合版；取拆分版。 |
| `Turbo 8-step v1.0 544p`（lightx2v） | FL2VA | 8 或 4 | 12 / 3 | 544p | 那一系的 8 步版，使用模型文件自身的 shift。 |
| `Turbo 8-step v1.0 768p`（lightx2v） | FL2VA | 8 | **6** / 3 | 1344×768 | 上游自己的默认选择——LightX2V Studio 用的就是这个。也发布了**拆分**版，应优先选那一份；见下文。 |
| `Turbo 4-step v0.1`（lightx2v） | **Ref2VA** | 4 | 12 / 3 | 544p | 参考图模式这两个里更便宜的一个，也是使用模型文件自身 shift 的那个。也发布了**拆分**版。 |
| `Turbo 8-step v1.0 768p`（lightx2v） | **Ref2VA** | 8 | **6** / 3 | 1344×768 | 另一个，也是唯一需要 `video_shift: 6.0` 的 Ref2VA 适配器。同时发布了**拆分**版和融合版；取拆分版。 |

> **这些版本号是上游的，而且没有文档。** lightx2v 的模型卡只描述了它部署的那个
> 8 步 v1.0，所以 4 步 768p 那一系的 v1.0、v1.1 和 v1.2 之间的区别在任何地方都
> 没有说明。两个较新的里只收录了 **v1.2**——版本号的意义就是取最新的——但这只
> 意味着「最新」，不代表「在这里实测更好」。在把习惯切换过去之前，先在一个你
> 熟悉的 seed 上与 v1.0 做对比。
>
> 模型目录*记录*的内容也是如此。对 **v1.2** 和 **Ref2VA 8 步 768p** 而言，
> 除 shift 之外的每个字段（模式、步数、分辨率、dtype）都是从文件名读出来的，
> 而 shift 是从 768p 那一系的其余条目继承来的。如果上游哪天给这些补上文档，
> 这个继承就是值得重新核对的假设：shift 错了就是 sigma 网格错了，而没有任何
> 东西会报出来。

**shift 是适配器的一部分，不是个人偏好。** lightx2v 的 768p 模型文件是在视频
shift 为 **6** 的条件下蒸馏出来的，而这个模型的默认值——以及这里所有不是在
768p 上训练的适配器——是 12。蒸馏是拟合它训练时所用的那个 sigma 网格的，所以在
错误的网格上运行不是风格差异；那是另一套调度表，而且没有任何东西会报出来。
`video_shift` 和 `lora` 位于同一个 `minimax-h3-model-config` stage 上，这正是
两者要在同一个 beat 里一起改动的原因。

**`-split` 是 diffusers 那一份，而它是更好的文件。** lightx2v 把每个适配器都
发布两遍——一遍是给 ComfyUI 的融合版，一遍是原始的拆分 `to_q`/`to_k`/`to_v`
导出——而 vpipe 两者都能加载。请优先选拆分版：至于为什么一个已发布的融合版在
两个权重发布版之一上会无声地出错、而拆分文件在两者上都正确，见
[哪些 Turbo 适配器可用](#which-turbo-adapters-work)。

<a id="get-it"></a>
#### 获取

一个 `model-fetch` stage——
[`prepare-minimax-h3-turbo-lora.vpipeline`](pipelines/prepare-minimax-h3-turbo-lora.vpipeline)：

```sh
vpipe --launch docs/pipelines/prepare-minimax-h3-turbo-lora.vpipeline
```

约 744 MB，相对第 1 步要花的那几个小时来说只要十秒左右——LoRA 就按它发布的样子
使用，所以之后没有什么需要量化的。

lightx2v 那一系是
[`prepare-minimax-h3-turbo-lora-lightx2v.vpipeline`](pipelines/prepare-minimax-h3-turbo-lora-lightx2v.vpipeline)，
形状相同，锁定的是 `lightx2v/Minimax-h3-Turbo-4step-768p`。那个仓库的两种写法
都已收录——它把每个适配器发布两遍，给 ComfyUI 的融合版和给 diffusers 的拆分版，
vpipe 两者都能加载。`model_variant` 的完整取值：

| variant | 是什么 |
|---|---|
| `lightx2v/Minimax-h3-Turbo-4step-768p` | FL2VA 4 步，shift 6 |
| `lightx2v/Minimax-h3-Turbo-8step` | FL2VA 8 步，544p，shift 12 |
| `lightx2v/Minimax-h3-Turbo-8step-768p` | FL2VA 8 步，shift 6 |
| `lightx2v/Minimax-h3-Turbo-8step-768p-split` | 同一个的**拆分**版——优先选这个 |
| `lightx2v/Minimax-h3-Turbo-4step-768p-v1.2` | FL2VA 4 步 **v1.2**，shift 6，**拆分**版——优先选这个 |
| `lightx2v/Minimax-h3-Turbo-4step-768p-v1.2-comfyui` | 同一个的融合版 |
| `lightx2v/Minimax-h3-Turbo-ref2va-4step` | Ref2VA 4 步，544p，shift 12 |
| `lightx2v/Minimax-h3-Turbo-ref2va-4step-split` | 同一个的**拆分**版——优先选这个 |
| `lightx2v/Minimax-h3-Turbo-ref2va-8step-768p` | Ref2VA 8 步，shift 6，**拆分**版——优先选这个 |
| `lightx2v/Minimax-h3-Turbo-ref2va-8step-768p-comfyui` | 同一个的融合版 |

这里 `model_variant` 不是可选的，而且它的值是模型目录里的**名称**，不是标题里的
某个词。一个仓库的每个 Turbo 模型文件都是从那同一个仓库发布的，所以光写
`"turbo"` 会匹配到多个，于是下载会被拒绝并列出候选项，而不是悄悄取第一个。对
larryvrh 来说，那就是把另一个换成
`larryvrh/MiniMax-H3-Turbo-Lora-v1-850-ema`；它们在磁盘上共用一个目录，但注册
在各自独立的键下。

下载会把适配器**注册**在它的模型目录名称之下——一个形如
`larryvrh/MiniMax-H3-Turbo-Lora-v4-600-ema` 的 `owner/name` 键——而运行用的
流水线里写的就是这个键，不是路径。配置 stage 的 **Browse** 按钮提供的也是它，
所以通常的流程是：把准备流水线跑一次，然后从列表里挑适配器。`lora` 接受一个
已注册的模型、一个只放着一个 `.safetensors` 的目录，或者一个直接路径，优先
顺序就是这个次序。请优先用键：一个仓库的两个 Turbo 模型文件会落在同一个目录
里，所以只有键这种形式才说得清你指的是哪一个。

<a id="run-it"></a>
#### 运行

[`minimax-h3-text-to-video-turbo.vpipeline`](pipelines/minimax-h3-text-to-video-turbo.vpipeline)
就是第 2 步的那张图，只有两处关键改动：在配置 stage 上指定的适配器，以及把
`steps` 从 8 降到 **6**。

```sh
vpipe --launch docs/pipelines/minimax-h3-text-to-video-turbo.vpipeline
```

它仍然需要第 1 步得到的基础模型（`local/MiniMax-H3-FL2VA-8bit`）——适配器替换的
是步数，不是模型文件。图里其他一切都没有动，而这正是重点：应用一个 LoRA 是在
一个 stage 上改配置，而不是换一条流水线。

**它的开销。** 960 × 544 · 24 fps · **124 帧**（5.17 秒）· 6 步，从
`vpipe --launch` 到封装好的 mp4 的端到端时间。流水线要求的是 120 帧，而 stage
向**上取整到了 124**——视频 VAE 处理 17 帧的片段并从每段保留 5 个潜变量，所以
只有 17n+5 才有对应的潜变量形式，日志里会说明这一点。

| 机器 | |
|---|---|
| **M4 Pro** Mac mini，64 GB，模型放在外置 Thunderbolt SSD 上 | **21 分 44 秒** |
| **M5** MacBook Air 15"，16 GB，无风扇，垫在冰袋上 | **11 分 25 秒** |
| **M5 Pro** MacBook Pro 16"，24 GB，自带风扇，无额外散热辅助 | **5 分 0 秒** |

三行都是同一个流水线文件，所以它们可以直接比较：无风扇的 M5 比有风扇的 M4 Pro
**快 1.9×**，而且只用了四分之一的内存。在 16 GB 上 DiT 会流式读取它的权重，这
正是那台机器没能更快的原因。M5 Pro 又**快 2.3×**——是 M4 Pro 的 4.3×——它内存
更多，而且可以实测到它的机身守得住频率。

每一行都是那套机身的最佳表现，而对 Air 来说，这个最佳也维持不了多久。它以满频
**1578 MHz** 起步，保持约**两分钟**，然后降频到在 **1300 MHz** 附近波动——是该
芯片的 82%——因为冰袋是一块会变热的散热器，而不是稳定的散热。因此一次 11 分钟
的运行里有**约 82% 的时间处于降频状态**，这也是为什么在无风扇的 Mac 上最不该
相信的数字是短基准测试：它可能在机器变慢之前就跑完了。**同一次运行放在桌面上
要约 15 分钟**，所以在这里散热大约值墙上时间的**四分之一**，而它也是在从一个
计时里读出任何别的结论之前最该先确认的事。

出于同样的原因，这一行**噪声很大**。冰袋是手放上去的，它放在哪里既会改变加速
窗口的长度，也会改变之后的频率，所以 Air 的数字是一个样本，而不是一个可重复的
数值。把它当作数量级来看，并据此与它比较。

M5 Pro 那一行是机器的出厂状态，下面什么都没垫：它的风扇足够好，以至于这个负载
**把 GPU 钉在 1620 MHz——它的最高频率——并在整个运行过程中保持 100%**——所以和
Air 不同，它是可重复的，也是当一个数字必须站得住脚时该引用的那一行。因此两行
M5 并不是同一块芯片以同样的速率在跑：一台被顶在自己的上限上，另一台大部分时间
都在**比自己的上限低 18%** 的状态下运行。这在两者之间 2.3× 的差距里约占
**1.25×**，剩下大约 **1.8×** 归于核心数量和内存——值得分开来看，因为只有那
1.25× 是更好的散热能夺回来的。

和这里其他每条流水线一样，这一条也设置了 `i8_gemm`（见[值得了解的设置](#the-settings-worth-knowing)）。
它在 M4 Pro 上不起作用，而 M5 上不开它的运行会慢于 11 分 25 秒。它能和适配器
一起用，但它对画面的改变大致和适配器一样多，所以当你要评判的是画面而不是速度
时，请一次只开一个。

应用它有两种方式，而对这个适配器来说，两者并不等价。

<a id="runtime-recommended"></a>
#### 运行时应用（推荐）

在 `minimax-h3-model-config` stage 上指定它，DiT 就会在运行中应用它——每个被
适配的投影计算 `W x + scale * B (A x)`：

```json
{
  "id": "h3-config", "type": "minimax-h3-model-config", "iports": [],
  "config": {
    "lora": "larryvrh/MiniMax-H3-Turbo-Lora-v4-600-ema",
    "lora_scale": 1.0
  }
}
```

把它的输出接到 `generate-video` 的 `model_config` 输入端口（iport），把 `steps`
设为 4–8，改动就这些。磁盘上不写任何东西，基础模型文件也不会被改动，所以切换
适配器——或者关掉一个——只是改一处配置，而不是跑一遍 66 GB。

应用它的代价是一步里的百分之几，加上一块小的暂存缓冲，所以你在步数上省下的都
留得住。

**`lora_scale` 是实时的。** 它作为每次前向的一个值搭在一个 GEMM 常量上，而不是
被折叠进因子里，所以一个由触发器驱动的配置 stage 可以让它跨多个 beat 扫值，
每次改动的代价只是一次 setter 调用，而不是重新加载 33B 的权重。`0` 会完全跳过
适配器的那两个 GEMM，所以*关掉*就是彻底关掉，与未适配的模型做 A/B 只需改一处
配置。适配器自身的 `alpha/rank`（kohya 约定的文件里带着它；这两个没有）是
**文件**的属性，在加载时折叠进去一次，所以两者永远不会混淆。上游是按 `1.0`
调的：出现模糊重影时往上微调（约 1.05–1.2），出现过锐颗粒时往下（约
0.8–0.95）。

`lora` 这个路径本身是**加载期**参数，它与 `lora_scale` 之间的不对称是实实在在
的，而不是疏漏：一个被适配的 `mlp.fc1` 会改变 block 用哪些 kernel 构建，所以
它不能在运行中的 DiT 下面被替换。在 DiT 构建完成之后改变适配器的 beat 会被
报告并忽略，而不是悄悄应用到下一个片段；只改变强度的那种则会被应用。

<a id="two-at-once"></a>
#### 同时用两个

一共有**两个槽位**。`lora2`、`lora2_scale` 和 `lora2_qkv_layout` 是第二个槽位，
它们的工作方式和前三个完全一样：

```json
{
  "id": "h3-config", "type": "minimax-h3-model-config", "iports": [],
  "config": {
    "lora":  "larryvrh/MiniMax-H3-Turbo-Lora-v4-600-ema",
    "lora_scale": 1.0,
    "lora2": "my-style-adapter",
    "lora2_scale": 0.7
  }
}
```

此时每个被适配的投影计算 `W x + s1 B1 (A1 x) + s2 B2 (A2 x)`。这个机制所针对的
搭配是：第一个槽位放一个**少步数蒸馏**——那是对整个模型的一个修正，应当保持在
它训练时的强度——第二个槽位放一个**风格或身份适配器**，那才是你真正会去调的。
不过模型本身并不区分这两个槽位：任一个槽位都可以放任一类，而只用 `lora2`
（不设 `lora`）也是一个完全正常的请求。

**两个强度互相独立，而且都保持实时。** 这正是两者做成分开的槽位、而不是一组
合并后的因子的原因。合并在算术上是精确的——把因子堆在 rank 轴上，
`A = [s1 A1; s2 A2]`、`B = [B1 | B2]`，一个 rank 为 `r1 + r2` 的适配器就能免费
算出同样的和——但它会把两个强度都折叠进 `A`，于是对最可能被扫值的那个适配器来
说，实时可调的设定又变回了一次重建。

第二个适配器的代价是**它自己的一对瘦长 GEMM**，在 rank 64 时约为一个投影的
1.5%。基础权重仍然只读一次，而时间实际上就花在那里。如果你要比较不同的运行，
有一点要注意：第二个适配器的增量是在一个独立的 bf16 过程中累加到输出上的，所以
把*同一个*适配器以一半强度分放在两个槽位里，与只放一个槽位、用满强度，二者并非
逐位相同——**实测为输出的 5.7e-3，且在各个强度下保持不变**——这相当于 1.5 个
bf16 ULP，不是随尺度变化的误差。两个*不同*的适配器付出同样的代价，而且没有什么
可供对比。

`lora2` 和 `lora` 一样是加载期参数，而 `lora2_qkv_layout` 是按槽位设置的，因为
融合 `qkv` 的行顺序是*文件*的属性：来自不同发布方的两个适配器需要不同的答案。

<a id="merging-and-why-it-loses-most-of-this-adapter"></a>
#### 融合，以及为什么它会丢掉这个适配器的大部分作用

`lora-fuse` 会写出一个把增量折叠进去的新模型文件。对*风格类* LoRA 来说那是对的
工具，在这里则是错的。用 Turbo 适配器对其 bf16 基础模型**实测**：

| 张量 | 期望的 \|dW\|/\|W\| | 融合后留存 | 发生变化的元素 |
|---|---|---|---|
| `blocks.7.mlp.fc1` | 2.26e-4 | 46% | 5.8% |
| `blocks.23.attn.qkv_proj` | 2.65e-4 | 51% | 6.4% |
| `blocks.40.adaln_proj.linear` | 3.87e-4 | 78% | 13.8% |

这个更新相对于权重是 2–4e-4；而 bf16 的相对步长约为 4e-3。对**94% 的元素来说，
`W + dW` 会直接四舍五入回 `W`**——这个修正比存储分辨率低了一个数量级。量化后的
基础模型——这里是 8-bit，如果你选了就是 4-bit——则更粗。上游也顺带说了同样的话：
它的 ComfyUI 节点默认在运行时应用 LoRA，并称融合的效果「略软一些」。

如果你还是想要一个融合后的模型文件：

```json
{
  "id": "fuse", "type": "lora-fuse", "iports": [],
  "config": {
    "base_model": "<models>/Comfy-Org/MiniMax-H3/diffusion_models/minimax_h3_fl2va_bf16.safetensors",
    "lora": "larryvrh/MiniMax-H3-Turbo-Lora-v4-600-ema",
    "output_name": "MiniMax-H3-FL2VA-Turbo-bf16",
    "scale": 1.0
  }
}
```

`lora-fuse` 的 `scale` 是同一个强度设定，只在写出增量时应用一次。它不是实时的
——改一次就意味着再跑一遍 66 GB——这也是在你还在挑数值时应优先走运行时路径的
另一个原因。

**`base_model` 是 DiT 的文件，不是它所在的目录。** Comfy-Org 的重打包是每个
组件一个文件，而那个仓库的 `diffusion_models/` 里放着*两个*任务模式、每个
66 GB；指定目录会把两个模型合到同一套张量名下。指定文件还能保住模式信息——融合
的输出是一个分片目录，`fl2va` 之所以能保住，只是因为融合过程把它从源文件名里
提取出来，写进了输出的 `config.json`，与记录 Comfy-Org 扁平 qkv 分组的
`qkv_per_head` 标志放在一起。这两者在张量里都看不出来。请预留磁盘：它读 66 GB、
写 62 GB，在 SSD 上约九分钟。

<a id="which-turbo-adapters-work"></a>
#### 哪些 Turbo 适配器可用

两条路径都以模型自身的模块名为键，并容许在其上再加一层 `diffusion_model.`
容器前缀（ComfyUI 的约定）——或者 kohya 对这些名字的扁平化写法，社区 LoRA 用的
就是它（见[社区 LoRA](#community-loras--civitai-musubi-tuner-ai-toolkit)）。
针对 FL2VA 基础模型实测：

| 适配器 | 模块数 | 是否可用 |
|---|---|---|
| `larryvrh/MiniMax-H3-Turbo-Lora` | 259（多出 `adaln_proj`、`final_layer`） | 可用，两条路径都可以 |
| `lightx2v/Minimax-h3-Turbo`，`_comfyui_` 那些文件 | 208 | 可用，两条路径都可以 |
| `lightx2v/Minimax-h3-Turbo`，diffusers 那些文件 | 312 | 可用，两条路径都可以——**而且这些才是该留下的** |

diffusers 那套写法不是命名上的差异，而是另一种分解方式：独立的
`to_q`/`to_k`/`to_v` 适配器按块对角堆叠进融合后的 `qkv_proj`，以及一个按
diffusers 的 **value 在前**顺序排列的 `ff.net.0.proj`，它的两半要交换之后才对应
本模型 gate 在前的 `mlp.fc1`。vpipe 在加载时做这两种变换，并与上游自己对同一个
适配器的 ComfyUI 转换逐张量核对过：`A` 在 rank 轴上精确拼接，`B` 精确地是块
对角，`fc1` 的两半精确交换，`out_proj` 和 `fc2` 逐字节相同，而 rank 128 上的
alpha 8 与融合版 384 上的 24 是同样的强度。

**优先选 diffusers 那一份**，这也是两份都被收录的原因。一个*已发布*的融合版是
针对某一种 qkv 列分组构建的——在现存的每一个情形里都是扁平分组——所以它用在
另一个发布方的权重上会无声地出错。而从拆分文件做融合是针对实际加载的那个 DiT
进行的，所以它在两边都对。它的下载也更小，1.38 GB 对 1.96，差别就是块对角 `B`
中为零的那三分之二——而且模型也更小，原因相同：拆分的 q/k/v 是以*分带*方式融合的，
每个输出行只保留它自己那一部分的因子；而已发布的融合版把这些零带进来，在内存里
也照样占着。

lightx2v 的 `_comfyui_` 版 `qkv_proj` 是 rank 384——三个 rank-128 适配器堆叠
起来——而这种堆叠方式正好能看出它假定了哪个基础模型：它的 `B` 是在
`[all q | all k | all v]` 这个意义上块对角的，所以它针对的是 Comfy-Org 的扁平
分组，而不是它的 `base_model` 标签所写的那个 per-head 发布版。

**在 MiniMaxAI 发布的权重上，一个融合版适配器会是错的。** 它是针对 Comfy-Org
的重打包训练的，所以它的 `attn.qkv_proj` 增量假定了扁平分组。用在 per-head
发布版上，它会把一个头的 `q` 增量加到另一个头的 `k` 上，50 个 block 全都如此，
而且没有任何东西会报出来。这正是拆分文件不存在的问题：它们的 q、k、v 是分开
到达、在这里融合的，融进所加载的那个 DiT 实际采用的分组。

<a id="community-loras--civitai-musubi-tuner-ai-toolkit"></a>
#### 社区 LoRA——Civitai、musubi-tuner、ai-toolkit

社区训练的风格、运动或角色 LoRA，加载方式与 Turbo 适配器完全相同：把
`.safetensors` 填进 `lora`（或 `lora2`），用 `lora_scale` 设定强度。在 Web UI
里这个字段有两个按钮——模型选择器用于已编目的适配器，文件浏览器用于你下载到
沙盒里的 `.safetensors`。**不需要任何转换。** 读取器接受 H3 各训练工具已知会写出的每一种写法：

| 由谁写出 | 张量名 | 强度 |
|---|---|---|
| musubi-tuner、kohya sd-scripts | `lora_unet_blocks_0_attn_qkv_proj.lora_down.weight` / `.lora_up.weight` | 每个模块自带的 `.alpha` |
| ai-toolkit、diffusion-pipe、ComfyUI 转换版 | `diffusion_model.blocks.0.attn.qkv_proj.lora_A.weight` / `.lora_B.weight` | 有 `.alpha` 就用，否则按满强度 |
| 本 DiT 自己的名字 | `blocks.0.attn.qkv_proj.lora_A.weight` | 同上 |
| diffusers / peft | `transformer_blocks.0.attn.to_q.lora_A.default.weight` | 文件头里的 `alpha` |

kohya 的名字就是本模型自己的模块路径，只是把点换成了下划线。这一步从文件
那一侧无法还原——`qkv_proj` 和 `qkv.proj` 扁平化后一模一样——所以 vpipe 反过来
做：把模型的名字扁平化后去查找，这是精确的。`lora-fuse` 也以同样方式读取
kohya 的写法，所以对这样的文件，运行时路径和合并路径适配的是同一批投影。

日志会写明它识别出的是哪种写法——下面是一个按 musubi-tuner 默认目标训练的
LoRA，50 个 block 里各四个投影：

```
MetalMiniMaxH3Transformer: runtime LoRA 'my-style.safetensors' -- 200 modules
  at scale 1, rank <= 32, kohya lora_down/lora_up factors
```

**不要手工改键名。** 转换过的文件只有保住两样东西才能用，而随手写的脚本往往
会丢掉它们。第一是 `.alpha`：kohya 对每个模块按 `alpha / rank` 施加，丢掉它就
改变了强度——一个以 alpha 1、rank 32 保存的适配器会因此**强 32 倍**。第二是每对
因子所属的模块：被改名到错误投影上的一对因子会被搁置，适配器的其余部分照常
运行——新名字若指向另一形状的投影，日志报为 `SKIPPED (shape mismatch)`；若指向
模型里根本没有的名字，则什么都不报。请把训练工具写出的原始文件直接交给加载器。

一个仍然绑定不上任何模块的文件会被拒绝，错误信息里列出上面这些写法。那是命名
问题，而不是下载损坏——请附上它的几个张量名提一个 issue。

融合的 `qkv_proj` 带有一种行顺序，名字里看不出来。与 Turbo 适配器一样，vpipe
把融合的 `qkv_proj` 适配器按 Comfy-Org 的扁平分组来读，并为 per-head 的
MiniMaxAI 发布版重新排序。如果某个适配器是在 MiniMaxAI 自己的权重上训练的，
请设置 `lora_qkv_layout: per_head`（第二个槽位用 `lora2_qkv_layout`）。

<a id="eight-steps--hyperflow"></a>
### 八步——HyperFlow

[HyperFlow](https://huggingface.co/videorebirth/hyperflow)（Video Rebirth）
是把步数降下来的第二种办法，而且和上面的 Turbo 适配器不是同一类东西。它是一个
8 步的**流映射（flow map）**自蒸馏：每一步都以它所积分的区间 `(t, r)`——从哪里
出发、落到哪里——为条件，而不是只以时刻 `t` 为条件。它用**一个文件**覆盖全部三种
工作流（`t2va`、`fl2va`、`ref2va`）。

<a id="fetch-it-then-name-it"></a>
#### 获取，然后指定它

[`prepare-minimax-h3-hyperflow.vpipeline`](pipelines/prepare-minimax-h3-hyperflow.vpipeline)
负责下载（2.8 GB）并把它注册为 `videorebirth/hyperflow`：

```sh
vpipe --launch docs/pipelines/prepare-minimax-h3-hyperflow.vpipeline
```

[`minimax-h3-text-to-video-hyperflow.vpipeline`](pipelines/minimax-h3-text-to-video-hyperflow.vpipeline)
就是第 2 步的那张图，只是在配置 stage 上指定了这个适配器：

```json
"lora": "videorebirth/hyperflow",
"lora_scale": 1.0
```

**Ref2VA** 的图请改用 `videorebirth/hyperflow-ref2va`——同一个文件，在模型目录里
登记了第二次，好让 Browse 列表在参考图模式下也能列出它。任何 FL2VA 的图（首帧、
首尾帧）用普通的那个键即可。

改动就这么多。**你不需要设置步数。**适配器文件里写明了它蒸馏时用的 sigma 网格——
9 个点、8 次前向——只要适配器在生效，vpipe 就跑这个网格；日志会说明这一点，而图里
的 `steps` 在此期间不会被使用：

```
GenerateVideoStage('generate-video'): HyperFlow 1.0 on lora slot 0 --
8-step flow-map grid from the adapter (`steps: 8` is not used while it is
live), two-time embedding at gate 0.25
```

`video_shift` / `audio_shift` 保持 **12 / 3**，这是这个模型的默认值，也是适配器
训练时用的 shift。网格会按图里设置的 shift 来变换（上游也是这么做的），但不一致
时会给出警告：质量只在训练时的数值上验证过。

`lora_scale` 保持 **1.0**。设为 **0** 时模型就精确地回到原始模型——适配器的两半
一起关掉，所以一张图可以用同一个已加载的模型对比两者——但介于两者之间的数值是
适配器从未训练过的状态。HyperFlow 占一个适配器槽位，所以风格或人物适配器仍然可以
放在 `lora2`。同时加载两个流映射适配器会被拒绝。和 **Turbo** 适配器搭配不会被
拒绝，但也没有用：两者都是同一个模型的蒸馏，会叠加在一起。

<a id="what-makes-it-different"></a>
#### 它的不同之处

Turbo 适配器只是一个 LoRA：同样的 block、固定的调度，只是步数更少。HyperFlow 是
一个 LoRA 再加上恰好一处结构改动，位置在时间步 MLP。除了模型文件自带的时间嵌入器，
它还带有一个**终点（endpoint）**嵌入器——前者的一份拷贝，配有自己的适配器——
用来嵌入 `r`，两者在每个 AdaLN 读取之前按下式混合：

```
temb = emb(t) + gate * (emb_r(r) - emb(t))        gate = 0.25
```

所以每一步每一行需要两个数，而原来只需要一个。生成的视频行走
`(t_video, r_video)`，生成的音频行走 `(t_audio, r_audio)`，各自在自己变换后的网格
上。条件行——关键帧、参考帧、参考音轨——没有要去的地方，所以那里 `r == t`。每一步
的 AdaLN 表是按 `(t, r)` 对烘焙的，所以烘焙依然适用。

时间步 MLP 在主机上以 f32 运行，这个适配器属于它的那部分也一样：文件有意以 f32
发布这些因子，它们从不经过 bf16。与上游自己的 `TwoTimeEmbedder` 对照，在一次 8 步
运行用到的每一个 `(t, r)` 对上，嵌入的相对误差约为 **1e-6**。适配器本身让它变化了
**9%**，所以这个检查有五个数量级的余量。调度与上游**逐位一致**。

文件的其余部分是熟悉的 diffusers 分解——拆分的 `to_q`/`to_k`/`to_v`、value 在前的
`ff.net.0.proj`、refiner block——在这里融合进所加载的 DiT 实际采用的 qkv 分组，
和 lightx2v 的拆分文件完全一样。和这里的每个适配器一样，它在运行时应用。

<a id="what-it-costs-1"></a>
#### 代价

在一台 M4 Pro（64 GB）上测得，8 位 FL2VA 模型预加载，640 × 352 × 56 帧
（3943 行），在同一段时间里先后运行：

| | 前向次数 | 每次前向 | 去噪 |
|---|---:|---:|---:|
| 原始模型，`steps: 8` | 7 | 27.7 s | 194 s |
| 原始模型，`steps: 16`（良好质量） | 15 | 27.7 s | 约 415 s |
| **HyperFlow** | **8** | **30.2 s** | **241 s** |

适配器让每次前向**慢 9%**。它换来的是步数：原始模型要 15 步以上的片段，它 8 次
前向就完成。加载到内存里适配器约 **2.8 GB**，与磁盘上相同。它的 q/k/v 是同一个融合
投影上的三个秩 256 的适配器，每个输出行只与它自己那一部分做收缩；等价的单个秩 768
更新会有三分之二是零——多占 1.1 GB，在 M4 Pro 与 M5 Pro 上交错实测每次前向慢
1.4–2.2%。

同一个种子跑两次，得到逐位相同的视频。

**配合 Sol-Attn。**上游为这个 8 步网格发布了一份 Sol-Attn 配方：前 **2 步**和前
**2 个 block** 保持 dense，`tau` 取 1.0。在 vpipe 里就是在 `generate-video` 上
`sol_attn: true` 旁边设 `sol_dense_steps: 2`、`sol_dense_layers: 2` 与
`sol_tau: 1.0`。dense 的那一步已验证与不路由的模型完全一致，但两者组合后的画面质量
在这里没有测过——请在你熟悉的种子上判断。见
[更快的注意力——Sol-Attn 路由](#faster-attention--sol-attn-routing)。

<a id="faster-attention--the-vdn-linear-branch"></a>
### 更快的注意力——VDN 线性分支

上面的 Turbo LoRA 削减的是一个片段要花多少步。这里削减的是一步要花多少，而它的
做法是改变注意力*本身*——从开销随序列长度*平方*增长的**密集（dense）**注意力，
换成开销与序列长度同步增长的**线性**注意力。这里的序列是片段中的每一个视频行，
而这个行数随时长**和**分辨率**两者**上升——所以**更长的片段和更大的画面都会让
它更值得**，而在又短又小的片段上则几乎不值得开。

[**VideoDeltaNet on MiniMax H3**](https://openvdn.github.io/#vdn-h3)（VDN-H3）
不是一个新模型。它是一份放在你已有模型旁边的第二份模型文件，把每个主 block 的
密集注意力替换为由两半组成的**混合（hybrid）**注意力：

- 对 query 附近的帧做的**窗口化 softmax**——跨度是固定数量的完整分块，而不是
  整个片段；
- 一条**双向 delta 规则**，以在帧上递推的线性注意力承载那个窗口看不到的一切。

两者对 key 做**划分**。每一帧恰好被其中一半读到，这正是为什么这是另一种注意力，
而不是叠在原来那套之上的近似——也正是为什么这个分支的权重是为它训练的。被两半
都算到的帧会被计算两次；两半都没算到的帧则被无声丢弃。

**为什么节省会随片段变长而增大。** 密集注意力把每一行与其他每一行都比较一遍，
所以**片段长度翻倍，它的开销就变成四倍**——而这里的行是整个片段的视频，不是
一帧的。混合注意力的两半都不这样做。无论片段跑多长，窗口都是固定的帧跨度，
所以片段翻倍只会让窗口的数量翻倍；旁边的 delta 规则是在帧上的递推，出于同样的
道理也是线性的。平方与线性之间的差距只随长度拉开，所以在短片段上节省只有百分之
几，而到 10 秒或 20 秒时，大部分收益还在前面等着你。

在单个 block 上实测，两半都跑在矩阵核（matrix cores）上，取两种序列长度。请看
**增长**那一列：那就是平方与线性各自在做它们会做的事。

| | 18,887 行 | 81,617 行 | 增长 |
|---|---|---|---|
| 密集注意力 | 431 ms | 9662 ms | **22.4×** |
| 混合注意力的窗口那一半 | 222 ms | 1105 ms | **5.0×** |
| 混合注意力节省的倍数 | 1.9× | **8.7×** | |

**行数变成 4.3 倍，密集注意力的开销变成 22 倍，而窗口只变成 5 倍。** 所以这个
差距不是一个查表可得的常数——你每加一行，它就再拉开一点。

**分辨率带来的收益和时长一样确定。** 一帧是 `(width / 32) × (height / 32)`
行——960 × 544 时是 510 行，1344 × 768 时是 **1008** 行——所以画布翻倍对序列长度
的作用和时长翻倍完全一样，而一个平方增长并不在意是哪一个造成的。因此在同样时长
下从 544p 换到 768p，大约是**密集注意力的 4 倍、混合注意力的 2 倍**，和表里
更长片段所展示的是同一笔交换。而且两者会叠乘，因为它们乘的是同一个数：又长又大
的片段正是密集注意力最糟、而这个设置最值得的地方。上表右侧那一列可以读作 544p
下约 22 秒，**或者 768p 下大约一半的时长**——两种情况下行数相同。

**文本和关键帧可以，参考图不行。** 这一句背后其实是两个不同的问题，答案也
不同。

*模式。* 已发布的两个 stage 都构建在 **FL2VA** 模式的 block 上，也没有可挂接的
Ref2VA 分支，所以本节和 **Ref2VA 模型文件**是二选一，而不是可以搭配的一对。
补上这个缺口的，是上游的**类 Ref2VA** 用法：改为经由 FL2VA 权重输入参考图——
于是这个分支和参考图*可以*同时拥有，在 FL2VA 上。vpipe 会运行它，它是什么、
不是什么见[类 Ref2VA](#ref2va-like--references-on-the-fl2va-weights)，把分支和
参考图放在一张图里的例子见
[`minimax-h3-ref2va-like.vpipeline`](pipelines/minimax-h3-ref2va-like.vpipeline)。
在 Ref2VA *模式*上挂这个分支，仍然是没有任何已发布配置覆盖的组合，
`generate-video` 会在图这样要求时发出警告。

*任务。* 这个分支的训练任务是**输入文本、输出视频和音频**。此后 OpenVDN 在
**同一份检查点**上发布了首帧、末帧以及首末帧条件生成，权重本身没有变化。关键帧
作为条件行打包在生成视频之前，保持在接近干净的噪声水平上，而混合注意力对待它们
的方式和对待提示词、声轨一样：每一个生成行都在窗口之外精确地关注每一个关键帧，
而线性那一半根本看不到它们。

vpipe 对这些行的处理与此相同，所以[不只是文本输入](#more-than-text-in)一节里的
图可以原样挂上分支：一个锚定帧对应开场帧，两个对应首帧和末帧。只给末帧是上游
有、而这个图没有接上的唯一一种模式——`generate-video` 会忽略没有首帧的末帧锚定，
并说明这一点。日志会针对每种几何尺寸记录一次关键帧的数量。

不过，模式不会替你检查。模式在 `model-select` 上选择，分支在
`minimax-h3-model-config` 上选择，是两个不同的 stage；把分支挂到 Ref2VA 模式的
block 上，回来的片段看起来仍然完全正常。发生这种情况时日志会给出警告，但不会
拒绝——所以请确认 `model-select` 选的是 FL2VA。

<a id="get-it-and-run-it"></a>
#### 获取并运行

两个模型目录条目，同一个仓库。**`stage-dmd`** 是 8 步的蒸馏版，也是该取的那
一个；`stage-b` 是它蒸馏自的那个 50 步模型。在 DiT 之外，每一个约 **5 GB**。

```sh
vpipe --launch docs/pipelines/prepare-minimax-h3-vdn.vpipeline
```

然后是 [**`minimax-h3-vdn.vpipeline`**](pipelines/minimax-h3-vdn.vpipeline)，
它就是文生视频那张图，只是在 `minimax-h3-model-config` 上加了两行：

```json
"linear_branch": "OpenVDN/vdn-minimax-h3-stage-dmd",
"lora": "larryvrh/MiniMax-H3-Turbo-Lora-v4-600-ema"
```

它能和 Turbo LoRA 组合——分支共享 DiT 自己的 q/k/v 投影，所以加在这些投影上的
一个适配器会同时喂给混合注意力的两半——也能和
[`i8_gemm`](#the-settings-worth-knowing) 组合，后者从两个方向影响到它：分支的
输出投影是由 DiT 自己的 GEMM 跑的，而它读到的 q/k/v 到那时已经被量化过了。
分支内部不受任何影响。

**它是加载期参数，和 `lora` 一样，原因也一样：** 它改变的是 block *是什么*，
而不是某个东西被施加得有多强。在一个已经构建好的 DiT 下面指定另一个分支会被
拒绝，而不是半生效。而一个被指定了却无法挂接的分支会让**该 stage 失败**，而不是
只发出警告，因为一张悄悄不带它运行的图，会用为另一种注意力训练的权重产出一段
完全说得过去的视频——没有任何人能从输出上看出来。

**它的内存开销：权重，而在 544p 下别无其他。** 这个分支就是上面那约 5 GB 的
额外权重，按与 DiT 自身权重相同的条件逐 block 流式读取。它的*工作*内存比听起来
更大——5 秒的片段超过 1 GB，13 秒的超过 2 GB——但那不是第二块分配。它是从 DiT
已经持有的注意力暂存区里划出来的，而那块暂存区恰好在分支运行的那段时间里闲置，
所以在 960 × 544 及以上，分支的暂存区是免费的。不过两者并不同步增长：那块暂存区
随片段的**行数**增长，而分支的随其**帧数**增长，所以一个由很多小帧组成的片段
——每帧大约不到 250 行，也就是低于 672 × 384——会越过交叉点，分支开始要为这个
差额付出代价。`generate-video` 会在运行开始之前，按二者中实际适用的那个给机器
算内存。

<a id="what-it-saves"></a>
#### 它能省下多少

和[需要多长时间](#how-long-it-takes)里同一段 124 帧的片段，在 **M5 Pro，
24 GB** 上，配 Turbo LoRA 跑 6 步：

| | 124 帧，5.2 秒 |
|---|---|
| 密集注意力 | **5 分 0 秒** |
| VDN 混合注意力 | **4 分 38 秒** |

**在这个长度上是墙上时间的 8%，而这已经是收益偏小的一端。** 5 秒的片段不到
40 个潜变量帧，此时窗口仍然覆盖序列中很大一部分；混合注意力的优势在于窗口
*触及不到*的那部分片段，这也是为什么上面那个数字是一个下限。注意力也不是一整步
的全部——前馈和各个投影无论如何都要花它们该花的——所以即便那一个环节上的 1.9×，
传到墙上时间时也被稀释了两道。片段长度变成四倍时，同样的测量是 8.7×，那才是
这个设置值得拿出来用的地方。

> **这既是速度上的交换，也是质量上的交换，而质量这一面在这里没有实测过。**
> 混合注意力是另一种注意力，虽然分支的权重是为它训练的，但本仓库里没有任何东西
> 在保真度上把一段 VDN 的片段和一段密集注意力的片段比较过。上游报告其质量与
> 密集基线相当；请把那当成他们的说法，而不是一次复现。在把一个长任务交给它之前，
> 先在一个你满意的 seed 上各生成一遍。

<a id="cheaper-attention--sageattentions-int8-qk"></a>
### 更省的注意力——SageAttention 的 int8 QK

另一个改变注意力的设置，也是这里能和其他一切组合的那一个。
[**SageAttention**](https://arxiv.org/abs/2410.02367) 用 **int8 计算 QK^T
乘积**——它不是靠丢弃 key，而是把每一个 key 都算得更便宜。一个开关：

```json
"sage_attn": true
```

**它做了什么。** 一个 flash 注意力的矩阵计算是每个 key 块两次乘积：`QK^T` 和
`P·V`。Sage 把第一对操作数量化为 int8，**每块一个 scale**——Q 是每个 query 块
一个，K 是每个 key 块一个，而这正好是 kernel 本来就在分块的粒度，所以反量化
一个分数块只需一次标量乘法。`P·V` 保留张量原本的 dtype：`P` 是概率，本身条件
就好，量化它虽然能再省同样多，但换来的误差要差得多。

**让它成立的是对 key 的平滑处理，而且这个处理是精确的。** K 是按 token 维度以
`K − mean(K)` 的形式量化的。key 某个通道上的离群值并不是 token 之间的变化，而是
所有 token 共有的一个大偏置，把它减掉就能让 int8 的数值范围花在信号上而不是
偏置上。事后不需要加回任何东西：一个按通道的平移会让一行里的每个分数都移动同样
的 `⟨q, mean⟩`，而 softmax 看不到按行的平移。Q *不*做平滑，而这种不对称不是
疏漏——只有 key 一侧带着那个共有的偏置，而平移 Q 会让每个分数移动
`⟨q_shift, k_j⟩`，这个量在一行之内是变化的。

在一台 M5 上、8 heads × 20036 rows × head_dim 128 的规模下**实测**：

| | 一次注意力 |
|---|---|
| f16 kernel | **156.6 ms** |
| int8 QK，含前导处理 | **130.6 ms** |

**1.20×**，其中量化前导是 2.2 ms——整次调用的 1.7%。上限是 1.33×：int8 在
fragment 流水线上是 2.00×，而 QK 只占一个 flash kernel 矩阵计算的一半，所以
这已经拿到了其中可拿的大部分。精度方面，与一个双精度参考相比余弦相似度为
**0.99992**，而 f16 kernel *也是* 0.99992。

**仅限矩阵核。** int8 的 fragment MMA 是 M5 的指令，而且没有 ALU 回退路径，
所以 M4 会在日志里说明一次，然后照密集注意力运行，而不是拒绝——今天能跑的图，
在它以前能跑的所有地方都还能跑。

**它可以组合。** `sage_attn`、`sol_attn` 和 `i8_gemm` 是三个互相独立的选择，
彼此都不读对方：`i8_gemm` 决定一个 block 的 GEMM 怎么算，`sol_attn` 决定哪些
key 块会被关注，而 `sage_attn` 决定被关注的那些怎么相乘。`sage_dense_layers`
会把开头连续若干个 block 保留在 f16；它默认为 **0**，与 `sol_dense_layers` 的
1 不同，因为 Sage 会计算每一个 key 和每一个 query，也没有任何已发表的配置需要
一段密集的前缀。

`generate-image` 上也有同一个设置，FLUX.2、Krea-2 和 Qwen-Image-Edit 都支持。

<a id="faster-attention--sol-attn-routing"></a>
### 更快的注意力——Sol-Attn 路由

另一种让每一步变便宜的办法，而且它不需要你额外准备任何东西。
[**Sol-Attn**](https://nvlabs.github.io/Sana/Sol-Attn/) 是来自 NVIDIA Sana
项目的免训练稀疏注意力。与上面的 VDN 分支不同，它**不是一个模型文件**——没有
东西要下载，也没有东西要挂载，而且它完全不依赖具体的模式，而 VDN 分支是针对
FL2VA 的块训练出来的。只需在 `generate-video` 上加一个开关：

```json
"sol_attn": true
```

**它做了什么。** 注意力的开销主要花在那些几乎没有贡献的 key 分块上，而*哪些*
分块是这样的，取决于片段、注意力头和层——所以无法提前决定。Sol 在 softmax
运行的过程中做出决定，依据是它本来就要计算的一个代理量。每 64 个 key 组成的
分块保留两个摘要——key 的质心和 value 的均值——以及每个 query 分块一个阈值。
一次针对质心的乘积就能以稠密开销的 1/64 给每个 key 分块打分。高于阈值的分块
**精确**参与注意力；低于阈值的分块则被**折叠进同一个运行中的 softmax**，就好像
它那 64 个 key 全都带着质心的分数一样。所以没有任何东西被丢弃，从不构建路由
表，也没有第二遍计算。

**阈值是一个分布，而不是分块数量。** `sol_tau` 以该代理量自身在各分块间离散
程度的**标准差**为单位，正是这一点让单个数值能适配每一种注意力头、层、分辨率
和片段长度——而「保留最好的 20 个分块」做不到这一点。数值越高，保留越少。

**无论 `sol_tau` 取何值，有两样东西始终保持精确**，而且两者都不是可调参数。
一是 query 自身所在帧两侧的一条分块带：紧邻对角线的位置恰恰是质心最难替代
的地方，因为相邻的 key 正是 query 要去区分的对象。二是提示词和音轨所在的行
——H3 把它们和视频打包进同一个序列，而被质心概括过的提示词等于只读了一半。

<a id="what-it-saves-1"></a>
#### 它能省下多少

124 帧，**832 × 480**，24 fps，6 步，配合
[Turbo LoRA](#fewer-steps--the-turbo-lora) 和 `i8_gemm`，在一台
**MacBook Pro 16 英寸（M5 Pro）、24 GB** 上：

| | 124 帧，5.2 s |
|---|---|
| 稠密注意力 | **3 min 30 s** |
| Sol-Attn，`sol_tau` 1.0 | **2 min 44 s** |

**整体耗时 1.27×，去噪本身 1.40×**——两种方式下模型加载、提示词编码和 VAE
解码的工作量都是一样的。这次运行有 **23%** 的 key 分块保持精确。这个比例是
**数据**的属性，而不只取决于 `sol_tau`，所以它是实测出来的而非预测出来的：
日志会在每次前向计算时报告一次，调参时就该盯着这个数。

**它不消耗内存。** Sol 需要的一切只在一次注意力调用期间存在，所以 vpipe 把
模型在这段时间里不用的缓冲区借给它，而不是另行分配——在上述尺寸下，这个设置
**从不向机器索要**的内存是 **861 MB**。

<a id="against-the-vdn-branch"></a>
#### 与 VDN 分支的对比

两者替换的是同一个注意力，而且**只能开启其中一个**；同时指定两者会让 Sol
在分支所覆盖的块上关闭，并在日志中说明。它们的取舍并不相同：

| | VDN 线性分支 | Sol-Attn |
|---|---|---|
| 额外权重 | 约 5 GB，一个额外的模型文件 | **无** |
| 模式 | 仅 FL2VA——它的权重是针对那些块训练的 | **不依赖具体模式** |
| 改变了什么 | 注意力本身——一个窗口加一个线性递推 | 哪些分块精确参与注意力 |
| 如何随规模变化 | 窗口随片段长度**线性**增长，所以其优势会无上限地扩大 | 大致固定的一个分块比例——在这里实测的各次运行中接近四分之一 |

所以在大尺寸的长片段上该选分支，而 Sol 则是那个试一下毫无代价的选项。这里
不提供在同一几何配置下的正面对比——各节中的数字都是在各自实测时的尺寸下
得到的。

<a id="the-knobs"></a>
#### 各个选项

| 选项 | 出厂值 | 说明 |
|---|---|---|
| `sol_attn` | `false` | 关闭。它是一种近似，而它在哪些片段上是安全的，这是关于模型的判断，而不是关于 kernel 的判断。 |
| `sol_tau` | `1.0` | 以标准差为单位。数值越高保留的分块越少：速度上升、质量下降，两者都是单调的。 |
| `sol_key_block` | `64` | **只能取 32 或 64。** 32 做更多精确计算，更慢也更忠实——32 个 key 上的质心对它们的替代效果更好，而分块减半会让路由和摘要序列都翻倍。更大的分块实测在两方面都更差，因此会被拒绝，并给出警告并回退到 64。 |
| `sol_dense_layers` | `1` | 前若干个块保持稠密。第一个块是残差流冗余度最低的地方，而它只是 50 个中的 1 个。 |
| `sol_dense_steps` | `0` | 前若干个去噪步保持稠密——`sol_dense_layers` 在时间轴上的对应项。最前面的几步决定片段的大致结构。HyperFlow 发布的配方在它的 8 步里取 2。 |
| `sol_local_radius` | `1` | query 自身所在分块两侧的分块，无论路由怎么判定都保持精确。 |

它可以和 [Turbo LoRA](#fewer-steps--the-turbo-lora) 以及 `i8_gemm` 组合使用，
其中第二点值得说清楚，因为两者都是有损的：在本技术栈中实测，它们的误差是
**相互独立的**——两者同时开启时的误差落在各自误差的平方和开根号处，而不是
线性相加处，开启 Sol 并不会放大 `i8_gemm` 的代价。

> **这既是速度上的取舍，也是质量上的取舍。** Sol 丢弃什么是按片段决定的，
> 所以它对每条提示词的影响并不相同，而本仓库中没有任何东西在完整长度上把
> 路由过的片段与稠密片段作保真度对比。上游报告称质量得以保持；请把那当作
> 他们的说法，而不是一次复现。在把一个长任务交给它之前，先用你满意的随机
> 种子把两种结果都生成出来，并且只在你亲眼看过输出之后再提高 `sol_tau`。

<a id="the-neural-engine--for-m4-family-macs"></a>
### 神经网络引擎——面向 M4 系列 Mac

每一台 Apple Silicon Mac 在 GPU 旁边都还有第二个加速器，即 **Apple Neural
Engine**（ANE），而一次生成通常会让它闲置。vpipe 可以把每个块的一部分交给
它：**前馈网络**的行，以及可选的 **q|k|v 投影**的行，会被拆分到两个引擎上
并**同时**计算。这些层中的行彼此独立，所以拆分是精确的；不存在接缝。

```json
"ane_ffn": true,
"ane_qkv": true
```

**请在 M4 系列 Mac 上使用它**（M4、M4 Pro、M4 Max）。在那些机器上，GPU 的
矩阵乘法已经跑到接近其内存带宽所允许的极限，所以 ANE 是芯片上唯一空闲的算力，
把一部分行从 GPU 上拿走就能缩短每一个块的时间。

**在 M5 上它通常没有帮助——请关闭。** M5 GPU 的矩阵核心跑这些层比 M4 GPU 快
好几倍，而 ANE 大致还是原来的速度，所以留给它接手的空间要小得多。两个引擎还
读取同一份内存，所以 ANE 在推理时 GPU 自身的速率会下降。剩下的收益只有百分之
几，而且取决于几何配置：在 24 GB 的 M5 Pro 上，960 × 544、243 帧、开启 Sol
和 Sage、4 步的条件下**实测**，该层级成功启用，把去噪从 **203 s 降到 189 s
（1.07×）**。

在 M5 上它有可能划算的情形，是前馈网络和投影在一个块中占主导的情形——即
**高分辨率的长片段**——但那同时也是这些模块与前向计算争抢内存的情形，而在
24 GB 的机器上它们会败下阵来。在 1344 × 768、328 帧时，资源计划批准了该模块，
随后 stage 却拒绝了它：

```
the 98887-row forward fits only without the ANE modules' ~2711 MB
  -- running the GPU alone for this clip
```

那次运行与它的纯 GPU 对照组相差在一秒之内，这正是回退机制在正常工作。所以在
M5 上，请把这个层级当作需要**在你自己的片段上实测**的东西，而不是一个可以一直
开着的设置。

**无论进程占用看起来如何，它并不节省内存。** 在上面那组 M5 对比中，峰值 RSS
从 16.0 GB 降到 12.1 GB，而其中没有一点是节省下来的：这些模块的权重是**在进程
之外被 wire 住的**（约 1134 MB 和约 374 MB，有日志记录），而 DiT 的应对方式是
只保留 **50 个块中的 9 个**常驻，而不是 21 个。这个层级与让流式加载变便宜的
块常驻机制争抢内存，所以在内存紧张的机器上，它在重复读取权重上付出的代价可能
超过它在前馈网络上赢得的收益。无论哪种方式，激活值的临时空间都不变——它的大小
由序列决定，而不是由哪个引擎计算某一行决定。

<a id="what-it-saves-on-an-m4"></a>
#### 在 M4 上能省下多少

在一台 **MacBook Pro（M4 Pro）、64 GB** 上**实测**，8-bit，960 × 544，124
帧，4 步，稠密注意力：

| | 去噪 | 整体耗时 |
|---|---|---|
| 仅 GPU | 10 min 2 s | 14 min 17 s |
| `ane_ffn` + `ane_qkv` | **7 min 23 s** | **11 min 46 s** |

**去噪 1.36×。** 整体耗时的收益较小（1.21×），因为它还包含了一次性的模块构建，
以及一次 ANE 并不参与的 VAE 解码。按块来看，这两个层级是在一个更大的几何配置
下分别实测的（1344 × 768、328 帧，每个块都走 Sol 路由）：仅前馈网络就把一个块
从 23.6 s 降到 19.3 s，再加上 q|k|v 则降到 17.2 s。

生成的片段与纯 GPU 的结果不是逐位相同的——在同一随机种子下，两者的画面相差约
35 dB PSNR，这正是 fp16 路径的代价。

<a id="how-it-behaves"></a>
#### 它的行为方式

- **它会自我平衡。** `ane_rows` 为 0（默认值）时，会根据两个引擎实测出的速度
  来确定 ANE 分担的比例。该层级还会不时对一个纯 GPU 的块计时，并在拆分测出
  更慢时**退回到仅使用 GPU**。猜错的代价是百分之几，而不是整个片段。
- **它是有损的，和 `i8_gemm` 一样。** ANE 以 fp16 计算，它接手的那些行与 GPU
  的结果会略有差异。在两个真实几何配置的块上**实测**：视频输出的相对 L2 为
  **0.0035**，音频为 **0.00045**。溢出 fp16 的激活值会被检测出来，对应的分块会
  以更小的缩放重新计算，而不是直接传递下去。
- **8-bit 和 4-bit 模型文件都能用**，运行时 LoRA 也一样：每个块的权重会被
  反量化（并合入任何适配器）到 ANE 读取的缓冲区中，一次处理一个块。
- **它会消耗内存，并且在运行前就已规划好。** 无论有多少个块使用，每个层级只
  持有一个共享模块，而资源计划会把两者一起计入：960 × 544 的 124 帧时为
  **1922 MB**，1344 × 768 的 328 帧时为 **2711 MB**。如果前向计算无法与它们
  共存，运行时会说明这一点，并改用**仅 GPU**，而不是拒绝执行。
- **首次运行要付一次性的构建开销。** macOS 会在第一次使用时编译每个模块：在
  M4 Pro 上，前馈网络约 **10 s**，q|k|v 约 **5 s**，全都发生在第一个去噪步
  之前。之后它会被缓存，只需约 50 ms。该缓存是**按程序**保存的（`vpipe` 和
  `vpipe-web-ui` 各自构建一次），并且**按 macOS 版本**保存，所以系统升级会再付
  一次。分辨率和片段长度从不触发重新构建。

| 选项 | 默认值 | 说明 |
|---|---|---|
| `ane_ffn` | `false` | 让前馈网络跑在 ANE 上。 |
| `ane_qkv` | `false` | 让 q\|k\|v 投影也跑在上面，作为第二个模块。**需要 `ane_ffn`**；两者是一起规划、一起批准的，所以在只批准前馈网络本来可以通过的情况下，资源计划可能会拒绝这一对。 |
| `ane_rows` | `0` | ANE 分担的行比例，从 0 到 1。`0` 表示自动平衡；固定比例用于基准测试。 |
| `ane_layers` | `0` | 限制有多少个块使用它；`0` 表示全部。这不是一个内存设置：所有块共享同一个模块。 |

同一个开关在 **`vae-decode`** 上也有，用于 H3 的视频解码器，其前馈网络占了
解码的大部分。那里的 `ane_ffn` 与 `generate-video` 的相互独立，并且遵循同样的
M4 开、M5 不开的规则。

<a id="memory"></a>
## 内存

**16 GB 是下限，而且确实可行**——但这只是因为两个大模型从来不会完整常驻。
vpipe 会针对每次运行做出决定，并在日志中说明它选了什么。

**权重流式加载。** 在 8-bit 下，transformer 约 33 GB，提示词编码器约 27 GB，
而机器只有 16 GB。两者都从磁盘流式加载各层，而不是整体载入，所以内存峰值由
*工作集*决定，而不是由模型文件大小决定。在条件编码器上设置
`unload_when_idle: always` 还意味着提示词编码器在去噪开始之前就已被卸载——两者
从不同时占用机器。

**自适应常驻。** 如果每一步都对所有内容做流式加载，8-bit 下每次前向计算要重新
读取约 8.9 GB。所以 transformer 在用过某些块之后会在空闲内存允许的范围内尽量
*保留*它们，让常驻集长到机器剩余空间的大小，并在压力下再交还出去。能保留多少
取决于机器里还有什么：在一台 16 GB 的 Air 上、两个 VAE 也都加载的情况下，它会
稳定在少数几个块——几个 GB——并在测到自己的页面正在离开内存时开始释放。

**这正是更大内存能带来收益的地方。** 常驻集只受空闲内存约束，别无其他，所以
32 GB 或 64 GB 的机器会在各步之间按比例保留模型的更多部分，并按比例减少重复
读取。在足够大的机器上，它会彻底停止流式加载，直接预加载。这一切都不需要配置
——它在加载时测量，并随着运行的推进自行适应。

**一次只跑一个重负载任务。** Metal 缓冲区是 wire 住的，无法被换出，所以另一个
大模型与它同时运行时，不会让机器优雅地变慢——而是会把机器耗尽。请让一次生成
先完成。

<a id="troubleshooting"></a>
## 疑难排查

**模型不在选择器里。** `model-select` 的 Browse 列表只显示各 stage 确实能运行
的模型家族。如果你准备好的模型没有出现，检查它在模型目录中的 `model_type`
是否为 `minimax-h3-fl2va` 或 `minimax-h3-ref2va`；你也随时可以直接输入注册键
或绝对路径。

**生成了视频，但音频是静音或不对。** 检查 `save-video` 是否设置了
`enable_audio: true`，以及 `audio-vae-decode` 是否连接到 `generate-video` 的
**端口 1**（端口 0 是视频）。

**生成的音轨没有体现参考音乐。** 提示词必须提出这个要求——声音来自与画面相同
的文本，所以要说明正在演奏什么（见[第 2 步](#step-2--text-to-video-and-audio)）。
然后检查编码器的这一行：`N reference(s) -> ... M reference audio rows`，其中
`M` 应该等于 `2 × 40 × 秒数`——5 s 的参考对应 400。远多于这个数，说明音轨是以
错误的采样率送进来的；编码器会就此发出警告并重采样，而修正方法是把上游
`audio-to-pcm` 的 `output_sample_rate` 设为 **32000**。

**`frames` 不是你要求的值。** 这是预期行为——见上面的表格。

**设置了 `ane_ffn`，但 ANE 上什么都没跑。** 有两行日志会说明原因。要么是这些
模块无法与前向计算共存（*"the N-row forward fits only without the ANE
modules"*），这时片段会仅在 GPU 上运行——降低 `frames` 或画面尺寸，或者关闭
该层级；要么是该层级成功启用，但它的控制器测出拆分更慢并退回了 GPU，而这在
M5 上是预期结果。见[神经网络引擎](#the-neural-engine--for-m4-family-macs)。

<a id="under-the-hood"></a>
## 内部机制

- 一个打包好的序列同时承载视频行和音频行；对其进行条件化的逐行 AdaLN 调制占
  33B 中的 13B。
- 两条 sigma 时间表同步推进——视频偏移 12，音频偏移 3（`minimax-h3-model-config`
  上的 `video_shift` / `audio_shift`）。
- 视频 VAE 是 1/16 分辨率下的 24 通道；音频通过一个单独的 VAE 解码为
  **32 kHz 立体声**。
- 在 **M5** 上，GEMM 和注意力跑在 GPU 的矩阵核心上（`matmul2d` / NAX flash
  attention）——包括两种加速注意力设置，它们的精确部分就是同一个 flash kernel，
  只是只遍历它们保留的那些 key 分块。

<a id="references-and-licences"></a>
## 参考文献与许可协议

**MiniMax-H3** 由 MiniMax AI 依据 **MiniMax H3 Community Licence Agreement**
发布，这也是模型文件所附带的许可协议，以及你在下载前需要在其 Hugging Face
页面上接受的那一份：

- 权重（FL2VA + Ref2VA）：<https://huggingface.co/MiniMaxAI/MiniMax-H3>
- 用于交叉验证 qkv 分组的重打包版本：
  <https://huggingface.co/Comfy-Org/MiniMax-H3>

**VideoDeltaNet on MiniMax H3 (VDN-H3)**——[更快的注意力](#faster-attention--the-vdn-linear-branch)
一节中的混合注意力——作者是 Haocheng Xi、Yiming Xie、Hexu Zhao、Yiwen Zhang、
Michael Liu、Thomas Creavin、Kurt Keutzer、Xiuyu Li、Zhaoyang Lv、Chenfeng Xu
和 Haiwen Feng，分别来自 UC Berkeley、Impossible Inc. 和 UT Austin。它的
Hugging Face 仓库声明的许可协议与它所挂载的基础模型相同，即
**MiniMax H3 Community Licence Agreement**，许可文本位于该仓库自己的
`LICENSE` 中。

- 项目主页：<https://openvdn.github.io/#vdn-h3>
- 代码：<https://github.com/OpenVDN/vdn-minimax-h3>
- 权重：<https://huggingface.co/OpenVDN/vdn-minimax-h3>

```bibtex
@misc{xi2026videodeltanet,
  title  = {VideoDeltaNet on MiniMax H3},
  author = {Haocheng Xi and Yiming Xie and Hexu Zhao and Yiwen Zhang and
            Michael Liu and Thomas Creavin and Kurt Keutzer and Xiuyu Li and
            Zhaoyang Lv and Chenfeng Xu and Haiwen Feng},
  year   = {2026},
  url    = {https://openvdn.github.io/}
}
```

**Sol-Attn**——[更快的注意力——Sol-Attn 路由](#faster-attention--sol-attn-routing)
一节中的路由注意力——作者是 Haopeng Li、Yitong Li、Junsong Chen、Tian Ye、
Haozhe Liu、Jincheng Yu、Duomin Wang、Ruihua Zhang、Zeke Xie、Enze Xie 和
Song Han，来自 NVIDIA。它作为 NVIDIA **Sana** 仓库的一部分发布，遵循该仓库的
**Apache-2.0** 许可协议。它本身不带任何权重，所以不需要为它下载任何东西，这里
也不转发任何内容——本代码树中的实现是根据已发表的方法和源码编写的。

- 项目主页：<https://nvlabs.github.io/Sana/Sol-Attn/>
- 论文：<https://arxiv.org/abs/2607.24027>
- 代码：<https://github.com/NVlabs/Sana>

```bibtex
@misc{li2026solattn,
  title  = {Sol-Attn: Accelerating Video Generation Inference via
            On-the-Fly Attention Sparsification},
  author = {Haopeng Li and Yitong Li and Junsong Chen and Tian Ye and
            Haozhe Liu and Jincheng Yu and Duomin Wang and Ruihua Zhang and
            Zeke Xie and Enze Xie and Song Han},
  year   = {2026},
  eprint = {2607.24027},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CV},
  url    = {https://arxiv.org/abs/2607.24027}
}
```

**Turbo LoRA** 是社区蒸馏版本，各自遵循其所在仓库的条款：
[larryvrh/MiniMax-H3-Turbo-Lora](https://huggingface.co/larryvrh/MiniMax-H3-Turbo-Lora)
和 [lightx2v/Minimax-h3-Turbo](https://huggingface.co/lightx2v/Minimax-h3-Turbo)。

这些内容都不在此处转发——本文中的每一个流水线都从发布方下载，你接受的也是他们
的许可协议。在将其输出用于商业用途之前，请逐一核对。
