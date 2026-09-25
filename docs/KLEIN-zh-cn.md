# 使用 FLUX.2-klein-9B 进行参考图编辑

[English](KLEIN.md) | 简体中文

**FLUX.2-klein-9B** 可以接收一张照片和一段提示词，输出编辑后的照片，或者接收**两张**
图片和一段提示词，把它们合成为一张。vpipe 通过自己的 **metal-compute** 后端在本机
运行它。使用 vpipe 自己的 Metal kernel，前向计算中不使用 Python，也不使用第三方张量
库。在vpipe内量化到 8-bit 或 4-bit，最快**4 步**出图。16 GB 内存的 Mac 即可运行。

<a id="two-checkpoints"></a>
## 两个模型版本：klein-9B 与 klein-9b-kv

Black Forest Labs 发布了该模型的两个版本，vpipe 都能运行：

- **`FLUX.2-klein-9B`**——模型本身。参考图与提示词、正在生成的图像一起参与联合
  注意力（joint attention），并在每个去噪步中重新计算。
- **`FLUX.2-klein-9b-kv`**——同一模型的一个**变体**，经过蒸馏，使**参考图 token
  只关注自身**。它们既看不到提示词，也看不到正在生成的图像，因此其 K/V 与时间步
  无关，vpipe 只需在第 0 步计算**一次**，之后各步复用。BFL 实测提速
  **1.21–2.66×**，参考图较多、输出尺寸适中时提速最明显。

速度的代价是：对同一张照片和同一条提示词，-kv 变体**有时效果会略差**于
klein-9B。大多数编辑两者效果相当；遇到不理想的结果时，用 klein-9B 运行同一个
流水线即可。

| | klein-9B | klein-9b-kv |
|---|---|---|
| **参考图 token** | 每步重新计算 | 只计算一次并复用 |
| **带参考图时的速度** | 基准 | 更快——两张参考图时优势更大 |
| **质量** | 基准 | 有时略低 |
| **`klein_kv`** | 不设置——没有 `flux2-model-config` stage | **`true`，必须设置** |
| **Hugging Face 仓库** | [`black-forest-labs/FLUX.2-klein-9B`](https://huggingface.co/black-forest-labs/FLUX.2-klein-9B) | [`black-forest-labs/FLUX.2-klein-9b-kv`](https://huggingface.co/black-forest-labs/FLUX.2-klein-9b-kv) |

本文的每个流水线都提供两个版本。二者只有两处不同：`model-select` 指定的模型，
以及 -kv 版本多出的一个 `flux2-model-config` stage。

> **`klein_kv` 不是优化开关。** 两个模型版本在磁盘上无法区分——相同的
> `config.json`、相同的张量名、相同的形状——所以 vpipe 无法识别你用的是哪一个。
> 两者的 token 顺序和注意力掩码不同，而权重正是*针对*各自的掩码蒸馏出来的，所以
> 用另一个模型版本的前向计算来运行，结果不是变慢，而是**错误**。-kv 模型版本请设置
> `klein_kv: true`，klein-9B 则不要设置。随附的流水线都已设置好。
>
> 该选项设在一个 **`flux2-model-config`** stage 上，这个 stage 连接到
> `generate-image` 的 `model_config` 输入端口（iport），而不是写在
> `generate-image` 自己的配置里。模型专属的设置放在各模型家族自己的 stage 中，
> 这样一个流水线能看出它是为哪个模型版本搭建的——而且当某个配置项不适用于当前加载的
> 模型时，会表现为一个明显的连线错误，而不是一行被悄悄忽略的配置。

<a id="what-you-need"></a>
## 准备条件

| | |
|---|---|
| **机器** | Apple Silicon Mac（M 系列）。 |
| **内存** | 16 GB。 |
| **磁盘** | 直接运行发布的 bf16 版本需**约 32 GB**；准备 4-bit 版本需**约 45 GB**，完成后保留**约 12 GB**——每个模型版本分别计算。 |
| **构建** | Apple Silicon 版 vpipe——在 arm64 macOS 上默认即是。详见主 [README](../README.md)。 |
| **Hugging Face** | 一个账号、已接受许可协议，以及一个**访问令牌（access token）**。两个模型版本都是受限访问（gated）的——见[下文](#models-gated)。 |

发布的模型是 bf16 格式。你可以直接运行，也可以让 vpipe 预先一次性量化为
4-bit——详见[需要量化吗？](#do-you-need-to-quantize)。两个模型版本大小相同：

| | |
|---|---|
| 下载（bf16） | **约 32 GB** |
| 准备期间峰值（下载 + 输出） | **约 45 GB** |
| 4-bit 模型（删除下载文件后） | **约 12 GB** |

每个仓库约 49 GB，但 vpipe 的模型目录条目只锁定 diffusers 子目录，**跳过顶层那份
多余的 transformer 副本**（约 17 GB 的相同权重）和示例图片。实际下载约 32 GB。

你只需要一个模型版本。两个都准备则需要两倍的空间。

<a id="models-gated"></a>
## 模型受限访问（gated）

两个 Hugging Face 仓库都受 **FLUX Non-Commercial License**（FLUX 非商业许可）
约束。未认证的下载不会成功——直接返回错误码 **401**——所以必须完成以下三步，准备
流水线才能下载。（ModelScope 镜像不受限；见[运行 bf16](#running-bf16)。）

1. **接受许可协议**，每个需要的模型版本都要接受。登录 Hugging Face，打开模型
   页面——[klein-9B](https://huggingface.co/black-forest-labs/FLUX.2-klein-9B) 或
   [klein-9b-kv](https://huggingface.co/black-forest-labs/FLUX.2-klein-9b-kv)
   ——接受条款。许可按仓库分别接受。请先做这一步：令牌无法授予你尚未接受的访问
   权限。
2. **创建访问令牌**：在
   [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
   创建。**Read** 令牌即可。如果创建的是细粒度（fine-grained）令牌，它必须带有
   **“Read access to contents of all public gated repos you can access”**
   权限——缺少该权限的细粒度令牌看起来有效，但访问这些仓库仍会返回 401，排查起来
   很费时间。
3. **把令牌填入流水线。** 打开准备流水线，填写 `model-fetch` stage 中空着的
   `hf_token`：

   ```json
   {
     "id": "fetch",
     "type": "model-fetch",
     "config": {
       "model_path": "black-forest-labs/FLUX.2-klein-9B",
       "hf_token": "hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx",
       "base_path": "./models",
       "skip_existing_files": true,
       "overwrite_existing": false
     }
   }
   ```

   也可以在 Web UI 中设置——**流水线**（Pipeline Manager）▸ `fetch` stage ▸
   `hf_token`——这样令牌根本不会写进文件。

> **令牌就是密码。** 准备流水线是一个你可能会提交或分享的文件，粘贴进去的令牌会
> 随之流出。下载完成后请清空该字段；一旦泄露，请到 Hugging Face 上撤销该令牌。

## 流水线

| | klein-9B | klein-9b-kv |
|---|---|---|
| **准备**——下载并量化。只需运行一次。 | [`prepare-klein-9b-4bit`](pipelines/prepare-klein-9b-4bit.vpipeline) | [`prepare-klein-9b-kv-4bit`](pipelines/prepare-klein-9b-kv-4bit.vpipeline) |
| **编辑照片**——输入照片 + 提示词，输出编辑后的照片。附带[已保存的自定义视图](#saved-composer-view)。 | [`klein-ref-edit`](pipelines/klein-ref-edit.vpipeline) | [`klein-kv-ref-edit`](pipelines/klein-kv-ref-edit.vpipeline) |
| **合成两张参考图**——输入两张图片 + 提示词，输出一张图片。 | [`klein-multi-ref`](pipelines/klein-multi-ref.vpipeline) | [`klein-kv-multi-ref`](pipelines/klein-kv-multi-ref.vpipeline) |

所有文件的扩展名均为 `.vpipeline`。双参考图示例读取本仓库自带的两张图片：
[`minimax-h3-reference-subject.jpg`](images/minimax-h3-reference-subject.jpg)
（与 [MiniMax H3](MINIMAX-H3.md) 教程共用）和
[`klein-multi-ref-panda.jpg`](images/klein-multi-ref-panda.jpg)。

点击链接后使用 **Raw ▸ 另存为** 下载，或直接从克隆仓库中的 `docs/pipelines/` 和
`docs/images/` 获取。

<a id="do-you-need-to-quantize"></a>
## 需要量化吗？

**不需要——即使是 16 GB 的 Mac 也不需要。** 准备流水线之所以量化到 4-bit，是因为
约 12 GB 的模型加载快，并能在小内存机器上留出余量；但 vpipe 可以直接运行发布的
**bf16** 模型版本。

当 bf16 模型放不进内存时，vpipe 会对它进行**流式加载**：transformer 的各个 block
和提示词编码器的各层在需要时才从磁盘读取，而不是一次性全部驻留在内存中。是否流式
加载由 vpipe 在每次运行时根据机器内存自动决定，无需任何配置。模型存放在 Mac 的
**内置 SSD** 上时，流式加载往往完全不增加耗时；外置硬盘读取较慢，影响会比较明显。

bf16 带来的是质量：它**更准确地遵循提示词中的细节**，输出图像的**细节通常也更
丰富**。

<a id="running-bf16"></a>
### 运行 bf16：只需一次下载，直接在终端完成

无需量化时，准备模型只需要一个 `model-fetch` stage，因此不必编辑流水线文件：用
`--launch-stage` 直接启动这个 stage 即可。它会下载约 32 GB、登记模型，然后退出。
请在工作目录中运行（见[第 1 步](#work-directory)），并把该目录放在内置 SSD 上。

**`/path/to/vpipe` 指的是什么。** 如果你自己构建了 vpipe，它位于构建目录下的
`build/apps/vpipe/vpipe`。如果你安装的是 **Vpipe Manager** 应用，应用内置了同一个
二进制文件：打开 **Settings（设置）**，找到底部的 **About（关于）** 一节，在
**Command-Line Tools（命令行工具）** 中点击 **Copy vpipe Path（复制 vpipe 路径）**。
复制到的路径中含有空格——因为应用名为 `Vpipe Manager`——所以在终端里**要用双引号把
它括起来**，否则 `Vpipe` 之后的部分会被当成另一个参数：

```sh
"/Applications/Vpipe Manager.app/Contents/Helpers/vpipe" --version
```

同一行还有 **Copy vpipe-web-ui Path（复制 vpipe-web-ui 路径）**，对应第 2 步
用到的 Web UI 程序，同样需要加双引号。

**从 Hugging Face 下载。** 仓库受限访问：先按[模型受限访问](#models-gated)中的
第 1、2 步接受许可协议并创建访问令牌，然后通过 stage 的 `hf_token` 配置项传入
令牌：

```sh
cd ~/vpipe-work                                    # 你的工作目录
/path/to/vpipe --launch-stage model-fetch \
  --stage-cfg model_path=black-forest-labs/FLUX.2-klein-9B \
  --stage-cfg hf_token=hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

**从 ModelScope 下载**（modelscope.cn——适用于无法访问 huggingface.co 的中国大陆
用户）。两个模型版本都以相同名称镜像在 ModelScope 上——
[klein-9B](https://modelscope.cn/models/black-forest-labs/FLUX.2-klein-9B) 和
[klein-9b-kv](https://modelscope.cn/models/black-forest-labs/FLUX.2-klein-9b-kv)
——且镜像**不受限**，无需账号或令牌：

```sh
cd ~/vpipe-work                                    # 你的工作目录
/path/to/vpipe --launch-stage model-fetch \
  --stage-cfg model_path=black-forest-labs/FLUX.2-klein-9B \
  --stage-cfg source=modelscope
```

如需 -kv 变体，在上面任一命令中改用
`model_path=black-forest-labs/FLUX.2-klein-9b-kv`。

即使从 ModelScope 下载，`model_path` 仍然填 **Hugging Face** 上的名称：`source`
只改变文件从哪里下载。无论从哪里下载，模型都登记在同一个键下、存放在同一个目录中，
因此后续流程无需任何改动。（如果某台机器总是从 ModelScope 下载，可以执行一次
`export VPIPE_MODEL_SOURCE=modelscope`，之后就不必再传 `source`。）

`Ctrl-C` 可以干净地停止下载，再次运行同一条命令即可断点续传。

然后**让 `model-select` 指向下载的模型**：把它的 `hf_dir` 设为
`black-forest-labs/FLUX.2-klein-9B` 或 `black-forest-labs/FLUX.2-klein-9b-kv`，
并直接跳到[第 2 步](#step-2)。流水线中的其他设置保持原样——包括 `klein_kv`，它
取决于模型版本，而不是精度。

## 第 1 步——准备模型

<a id="work-directory"></a>
### 首先，选择工作目录

vpipe 把**启动它时所在的目录**当作工作区，并在那里创建状态文件：`models/` 存放
所有下载或量化的模型，`data.mdb`/`lock.mdb` 是模型注册表，另外——仅在
`vpipe-web-ui` 下——还有一个 `sandbox/`，stage 的文件读写被限制在其中。

请在满足[准备条件](#what-you-need)中空间要求的卷上选择目录，并在第 2、3 步中使用
**同一个**目录：你即将准备的模型会登记在该目录的注册表中，在其他目录启动的运行
找不到它。

### 然后运行流水线

可以运行 klein-9B 的、-kv 变体的，或两个都运行——每个都是独立的：

```sh
cd ~/vpipe-work                                    # 你的工作目录
# 如果你有源码克隆：cp ~/src/vpipe/docs/pipelines/prepare-klein-9b-4bit.vpipeline .
curl -O https://raw.githubusercontent.com/tgo-app-dev/vpipe/main/docs/pipelines/prepare-klein-9b-4bit.vpipeline
# ... 填入 hf_token（见上文）...
/path/to/vpipe --launch prepare-klein-9b-4bit.vpipeline
```

如需 -kv 变体，换成 `prepare-klein-9b-kv-4bit.vpipeline`。

这项任务适合用命令行：耗时长、无需值守、主要受磁盘速度限制，启动后无需任何点击。
`Ctrl-C` 可以干净地停止，`skip_existing_files` 则让重新运行时从中断处继续。

依次包含四个 stage：

1. **`model-fetch`**——把仓库下载到 `./models`。令牌问题会在这一步暴露出来，
   表现为授权失败，而不是下载缓慢。
2. **`model-quantize`**（`target: dit`）——把 9B transformer 量化为 4-bit，
   group size 64，使用 **AWQ** 激活感知平滑及配套的裁剪搜索（`awq: true`、
   `awq_clip: true`）。AWQ 为每一层搜索一个缩放系数，使对激活值影响大的权重在
   舍入后得以保留；它只增加准备时间，运行时没有任何开销。-kv 流水线还在这里设置了
   `klein_kv: true`——校准时会以参考图为条件，若在普通联合注意力下校准该模型版本，
   裁剪依据的将是错误的激活值。klein-9B 的流水线则不设置，道理相同，方向相反。
3. **`model-quantize`**（`target: text_encoder`）——把 8B Qwen3 提示词编码器
   量化为 4-bit。其输出——`local/FLUX.2-klein-9B-4bit` 或
   `local/FLUX.2-klein-9b-kv-4bit`——是一个**完整模型**：量化后的部分加上所有
   未改动的部分。
4. **`model-remove`**——删除第 2 步产生的中间结果，它只用于为第 3 步提供输入。

完成后，你只需要这个 4-bit 模型（约 12 GB）；约 32 GB 的下载文件可以删除。

<a id="step-2"></a>
## 第 2 步——编辑图片

```sh
cd ~/vpipe-work                                    # 同一个工作目录
/path/to/vpipe-web-ui
```

把你的照片放进沙盒，命名为 **`sandbox/reference.jpg`**——随附的 `load-image`
stage 读取的就是这个文件名，缺少它流水线会失败。然后打开服务器打印的 URL，加载
`klein-ref-edit.vpipeline`（-kv 变体用 `klein-kv-ref-edit.vpipeline`），编辑
`text-prompt` stage，点击**启动**（Start）。随附的提示词：

> *Paint the reference picture in Claude Monet's impressionist style. Use
> low-saturation shades of blue and yellow colors. Use fine horizontal strokes
> to paint the water. Write "T-Go" as the artist name in handwriting.*
>
> （以克劳德·莫奈的印象派风格绘制参考图。使用低饱和度的蓝色和黄色。用细腻的
> 水平笔触描绘水面。以手写体写上“T-Go”作为画家署名。）

共九个 stage，-kv 变体再多一个：

```
text-prompt ──> diffusion-conditioner ──┐ prompt
                                        ├──> generate-image ──> vae-decode ─┬─> save-image
load-image ─┬─> image-resample ─┬─> vae-encode ──┘ reference latent (port 5) │        ^
            │                   │                                            │        │
            │                   └──> compare-image <─────────────────────────┘        │
            └── metadata (EXIF) ──────────────────────────────────────────────────────┘

model-select ──> diffusion-conditioner, vae-encode, generate-image, vae-decode

-kv only:  flux2-model-config (klein_kv: true) ──> generate-image model_config (port 7)
```

这个流水线有两点值得理解，正是它们让这次运行成为*编辑*而不是重新生成：

**参考图以 latent 而非像素的形式进入 DiT。** `vae-encode` 把重采样后的照片转换
为 latent，由 `generate-image` 在其参考图端口（上图中的 port 5）接收。在 -kv 变体
上，KV 缓存正是由这些 token 构建的。提示词走另一条路径，经过
`diffusion-conditioner`。

**`image-resample` 独自决定输出尺寸。** 它在编码前把图片裁剪为 512 × 512，而由于
`generate-image` **既没有**配置 `width` 也没有配置 `height`，它会根据参考 latent
推断两者。因此这一个 stage 就是全部的尺寸控制：改成 768 × 768，就得到
768 × 768 的编辑结果，无需同步修改其他任何地方。（在 `generate-image` 上设置
`width` 和 `height` 会优先生效——但要**同时**设置两者。推断是全有或全无的：只设置
一个维度，另一个会悄悄回退到 256 而不是被推断出来，结果得到一张比例奇怪的图片，
且不会报错。）

末尾有两个便利之处：`compare-image` 接收*原图*（来自 `image-resample`，因此与
模型看到的裁剪完全一致）和*结果*，并把它们配对供下文的自定义面板使用；
`save-image` 除图片外还接收 `load-image` 的 **metadata** 端口，因此原照片的 EXIF
信息会写入编辑后的文件。

量化到 4-bit 后整个模型约 12 GB——5.4 GB 的 transformer、6 GB 的提示词编码器，
外加一个小型 VAE——这在 16 GB 的 Mac 上所剩不多。vpipe 每次运行时都会据此调整：
此处未设置 `unload_when_idle`，因此它取值为 `auto`，每个 stage 根据实际内存决定
是否在两个 beat 之间释放权重，并在日志中记录所做的选择。在内存更大的机器上，所有
模型都会直接常驻内存。

## 第 3 步——合成两张参考图

同一个模型可以接收**第二张**参考图。这里它把一张图中的猫放到另一张图中的熊猫
旁边，并以第二张图的风格绘制。

把两张图片以流水线读取的文件名复制到沙盒中：

```sh
cd ~/vpipe-work
# 如果你有源码克隆：cp ~/src/vpipe/docs/images/<文件名>.jpg sandbox/
curl -o sandbox/minimax-h3-reference-subject.jpg \
  https://raw.githubusercontent.com/tgo-app-dev/vpipe/main/docs/images/minimax-h3-reference-subject.jpg
curl -o sandbox/klein-multi-ref-panda.jpg \
  https://raw.githubusercontent.com/tgo-app-dev/vpipe/main/docs/images/klein-multi-ref-panda.jpg
```

然后加载 `klein-multi-ref.vpipeline`（或 `klein-kv-multi-ref.vpipeline`），点击
**启动**。随附的提示词：

> *The orange tabby cat wearing the navy velvet jacket from image 1 sits
> beside the giant panda from image 2 on a mossy rock in a misty bamboo
> forest. The panda offers the cat a bamboo stalk. Keep the cat's face, fur
> pattern and jacket exactly as in image 1. Paint the whole scene in the soft
> watercolor style of image 2.*
>
> （图 1 中穿着藏青色天鹅绒夹克的橘色虎斑猫，与图 2 中的大熊猫一起坐在雾气弥漫的
> 竹林里一块长满青苔的石头上。熊猫递给猫一根竹子。猫的脸、毛色花纹和夹克要与
> 图 1 完全一致。整个场景用图 2 那种柔和的水彩风格绘制。）

这个流水线就是第 2 步的流水线，只是参考图链路变成了两条：

```
text-prompt ──> diffusion-conditioner ──────────────────────────┐ prompt
                                                                ├──> generate-image ──> vae-decode ──> save-image
load-image-1 ──> image-resample-1 ──> vae-encode-1 ─────────────┤ image 1 (port 5)
load-image-2 ──> image-resample-2 ──> vae-encode-2 ─────────────┘ image 2 (port 6)

model-select ──> diffusion-conditioner, vae-encode-1, vae-encode-2, generate-image, vae-decode

-kv only:  flux2-model-config (klein_kv: true) ──> generate-image model_config (port 7)
```

需要了解的几点：

**提示词按端口指称图片。** 参考图只以 latent 的形式进入 DiT——klein 的提示词编码
器只处理文本，看不到这些图片——所以模型靠位置区分它们：每张参考图在 DiT 的位置
编码中各占一个区段，按端口顺序排列。*image 1*（图 1）是连接到 `generate-image`
端口 5（`ref_latent0`）的那张，*image 2*（图 2）是端口 6（`ref_latent1`）上的
那张。调换连线，就必须同时调换提示词中的说法。

**`generate-image` 最多接收两张参考图。** 它只有两个参考图端口。

**输出尺寸在 `generate-image` 上设置。** 有两张参考图时，没有单一的源照片可供推断
尺寸，因此这个流水线设置了 `width: 768` 和 `height: 512`——适合并排放置两个主体的
横向画幅——而每张参考图都裁剪为 512 × 512。如果不设置，输出尺寸将只取自图 1。
与第 2 步一样，两个维度要么都设置，要么都不设置。

**这正是 -kv 变体收益最大的场景。** 一张 512 × 512 的参考图是 1,024 个 token，
两张就是 2,048 个——比正在生成的 768 × 512 图片的 1,536 个 token 还多。klein-9B
在每一步的每个 block 中都要处理全部这些 token；-kv 变体只计算一次。

这里没有 `compare-image`：有两个来源，就没有单一的原图可供对比。结果保存在沙盒中
的 `klein-multi-ref.jpeg`（或 `klein-kv-multi-ref.jpeg`）。

## 值得了解的设置

| stage | 配置项 | 随附值 | 说明 |
|---|---|---|---|
| `generate-image` | `steps` | 4 | 两个模型版本都经过引导蒸馏（guidance-distilled）。4 步是既定配方，并非偷工减料。 |
| `flux2-model-config` | `klein_kv` | -kv 流水线中为 `true`；klein-9B 的流水线中没有这个 stage | -kv 模型版本**必须**设置，对 klein-9B 则是错误的。见[上文说明](#two-checkpoints)。连接到 `generate-image` 的 `model_config` 输入端口。 |
| `generate-image` | `i8_gemm` | `true` | DiT 中的大矩阵乘法使用动态 int8 GEMM——速度约为 f16 的 **2 倍**，但**有损**。在没有 NAX 矩阵核心的 GPU 上会被忽略，因此保持开启是安全的。对比画质时可关闭。 |
| `generate-image` | `width`/`height` | 不设置（编辑）；768 × 512（两张参考图） | 不设置时，输出尺寸取自端口 5 上的参考图。要么都设置，要么都不设置。 |
| `image-resample` | `width`/`height` | 512 × 512 | 每张参考图编码时的尺寸——在编辑流水线中也决定**输出**尺寸。`fit: crop` 裁剪填满画面；`pad` 则会在参考图中填充灰色边框。 |
| `model-select` | `hf_dir` | `local/FLUX.2-klein-9B-4bit` / `local/FLUX.2-klein-9b-kv-4bit` | 只需指定一次模型；conditioner、VAE 和 DiT 都会共用它。必须与 `klein_kv` 一致。bf16 则填下载模型的键：`black-forest-labs/FLUX.2-klein-9B` / `black-forest-labs/FLUX.2-klein-9b-kv`。 |

`load-image` 和 `save-image` 中的路径都相对于沙盒——在 Web UI 下，编辑结果保存在
工作目录的 `sandbox/klein-edit.jpeg`（或 `sandbox/klein-kv-edit.jpeg`）。

<a id="watching-it-form--live-previews"></a>
## 看着它成形——实时预览

任何 klein 流水线都可以逐步显示画面的形成。`generate-image` 的端口 **2** 在每一步之后
取出模型对成图的当前估计，用 madebyollin 的 **TAEF2**（为 FLUX.2 潜空间训练的小型自编码器）
解码；接在那里的 `preview` 阶段会依次显示。两处改动即可开启：

1. 在 `flux2-model-config` 上把 `preview_vae` 设为 `madebyollin/taef2`。-kv 流水线已经带有
   这个阶段；klein-9B 的流水线需要添加一个，并接到 `generate-image` 的 `model_config` 输入
   端口。用一个 `model_path` 为 `madebyollin/taef2` 的 `model-fetch` 阶段下载一次（10 MB）。
2. 添加一个 `preview` 阶段，输入接 `generate-image` 的端口 2。

`preview_every` 每 *N* 步渲染一次（最后一步总会渲染），`preview_max_edge`（默认 512）在
发送前缩小画面。`preview_vae` 留空或端口 2 不接，就什么都不加载。

实测（FLUX.2-klein-4B，1024 × 1024，4 步，M4 Pro）：

- **没有可测量的开销：**每一步都预览为 21.2 秒，不预览为 21.0 秒。
- 每个预览在对应步骤之后 1.2–1.8 秒到达，是排在 DiT 后面等待；最后一个在 GPU 空闲时只用了
  0.3 秒。
- 最后一步的预览与真正 VAE 的解码结果一致度为 **28.2 dB**。TAEF2 比完整的 FLUX.2 VAE 粗糙，
  预览就当预览看；端口 0 上的成图仍由真正的 VAE 解码。

TAEF2 读取的是反打块（unpatchify）后的 DiT 潜变量，但*不*做反归一化：真正 VAE 会先做的
批归一化逆变换，恰恰是这个 TAE 蒸馏时没有的。

<a id="saved-composer-view"></a>
## 已保存的自定义（Composer）视图

`klein-ref-edit.vpipeline` 和 `klein-kv-ref-edit.vpipeline` 包含的不只是流水线图。
自定义视图的布局**保存在流水线文件中**——一个顶层的 `aux.composer` 对象，流水线
核心会忽略它，由 Web UI 读取——因此流水线可以连同你想用来观察它的仪表板一起分发。
这两个文件附带一个原图与结果的**滑动对比（wipe）**面板，下方是一个**流水线
编辑器**。

### 加载布局

布局是按需恢复的，而不是自动恢复，并且与流水线绑定——所以要先加载流水线：

1. **流水线**（Pipeline Manager）▸ **加载** ▸ `klein-ref-edit.vpipeline`。
2. 切换到**自定义**（Composer）视图。
3. **加载** ▸ **为流水线加载…** ▸ 选择 **`klein-ref-edit`**。

（-kv 变体在两处都改为 `klein-kv-ref-edit`。）

界面会提示*布局已加载*，面板随即出现。在启动流水线之前，对比面板显示 *waiting*，
启动后会自动连接——它所指定的对象（`klein-ref-edit` / `compare-image`）随布局一起
保存，所以无需手动指定，它就知道要显示哪个 stage 的输出。

在图片上拖动，即可在原图与编辑结果之间滑动对比。两个视图的缩放和平移保持同步，
正是这一点让细微的变化——一个签名、一处色偏——得以被看见。面板的 ⋯ 菜单可在滑动
对比、并排显示和仅 A/B 之间切换。

### 保存自己的布局

与上述过程相反：排好你想要的面板，然后 **保存** ▸ **随流水线保存…** ▸ 选择
流水线 ▸ 确认路径。这会把你的布局写入 `.vpipeline` 文件。同一菜单中的
**从文件加载… / 保存为文件…** 则处理独立的布局 JSON，适合想在不同流水线之间复用
的布局。

## 故障排除

**`model-fetch` 报告授权失败。** [模型受限访问](#models-gated)中的三步缺一不可，
最常被遗漏的是细粒度令牌的受限仓库权限。许可协议按仓库分别接受，所以接受了
klein-9B 的并不能访问 klein-9b-kv。另外，在已登录的浏览器中接受协议，对属于另一个
账号的令牌没有帮助。

**编辑结果忽略了参考图。** 检查 `vae-encode` 是否连接到 `generate-image` 的参考图
端口，以及 `image-resample` 是否位于 `load-image` 和 `vae-encode` 之间。

**切换模型版本后结果严重错误。** `klein_kv` 必须与模型一致：-kv 模型版本需要带
`klein_kv: true` 的 `flux2-model-config` stage，而 klein-9B 不能有它。只修改
`model-select` 的 `hf_dir`，会让一个模型版本在另一个模型版本的注意力方式下运行。

**-kv 的结果比 klein-9B 略差。** 这是该变体的取舍，不是故障。对这张图片改用同一
流水线的 klein-9B 版本即可。

**两个模型版本的输出质量都下降。** 先试试 `i8_gemm: false`——它是有损的提速模式。
如果不是这个原因，另一个可调的是 4-bit 量化：以 `bits: 8` 重新准备，得到更大、
更接近原模型的版本。

**似乎只有一张参考图起作用。** 检查两个 `vae-encode` stage 是否都连到了
`generate-image`——一个在端口 5，一个在端口 6——并且提示词以 *image 1* 和
*image 2* 同时提到了两张图。

**图片比例奇怪。** `generate-image` 上只设置了 `width` 或 `height` 中的一个。要么
都设置，要么都不设置。

**自定义视图提示“没有已保存的布局”。** 流水线已加载，但没有 `aux.composer`——你选
了别的流水线，或者选的流水线保存时没有带布局。双参考图流水线本身就不带布局。

## 实现细节

- **klein-9B**：token 顺序为 `[text, image, refs]`——参考图 token 追加在生成图像
  token 之后——所有 token 相互关注。第 *i* 张参考图位于位置编码的时间坐标
  10·(*i*+1)，生成的图像位于 0。
- **klein-9b-kv**：token 顺序为 `[text, refs, image]`——参考图 token 位于生成图像
  token **之前**。
- 其参考图 token 只关注自身，并以固定的时间步 0 进行调制。正因如此，它们的 K/V 在
  整个去噪过程中保持不变，可以缓存。
- 其参数形状与 klein-9B 完全相同，因此加载器、量化器和 LoRA 融合都无需新增任何
  代码——只有前向计算和掩码不同。
- 在 **M5** 上，GEMM 和注意力运行在 GPU 的矩阵核心上（`matmul2d` / NAX flash
  attention）；`i8_gemm` 也使用同一硬件。
