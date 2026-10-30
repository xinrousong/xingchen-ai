# fastbpe · 星辰分词器的 C 加速内核

把 tokenizer 里最耗 CPU 的那一步——「每轮扫描相邻对、找 rank 最小的 pair、整体合并」——
用纯 C 重写，编译成一个 48 KB 的 DLL，由 `xingchen/fastbpe.py` 通过 ctypes 调用。
其余部分（正则预切分、特殊 token、词表读写）仍在 Python，**语义逐 token 相同**。

## 为什么只做了这一层

| 想加速的部分 | 现状 | 说明 |
|---|---|---|
| 分词器合并循环 | ✅ 已换成 C | 纯 CPU 热点，语义可逐 token 验证 |
| MoE 分派 / MLA 注意力 | ✅ 已换成纯 PyTorch 快速路径 | 不需要新编译器，见 `xingchen/model.py` 的 `XINGCHEN_MOE_DISPATCH` / `XINGCHEN_MLA_ATTN` |
| 训练前向/反向的融合 kernel | ⏳ 工具链已就绪 | 2026-10-29 装好 MSVC 14.44 + CUDA 13.4 + ninja，冒烟测试通过（C++ 12.8s、CUDA 74.9s） |

## 编译环境（2026-10-29 已装好并验证）

- VS Build Tools 2022 17.14.42 / MSVC 14.44.35207：`C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools`
- CUDA 13.4（nvcc V13.4.59）：`C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.4`
- ninja 1.13.2（装在 python_embeded 的 Scripts 下）
- 一键进入编译环境：`scripts\msvc_env.cmd <script.py>`

踩过的三个坑，都写在 `msvc_env.cmd` 里：

1. `ninja` 必须**在 PATH 上**（torch 用 `shutil.which("ninja")` 查，装了 pip 包不够）
2. python_embeded 是 embeddable 版，**没有 include/ 和 libs/**，要借用同版本系统 CPython 3.13.14 的头文件与 import lib（`extra_include_paths` + `/LIBPATH`）
3. CUDA 13.4 的 CCCL 头要求 MSVC 用标准预处理器，nvcc 侧要传 `-Xcompiler /Zc:preprocessor`；另外 `cl` 的中文横幅会让 torch 在解码时崩（`UnicodeDecodeError`），用 `VSLANG=1033` + `chcp 437` 压成英文

验证脚本：`fastbpe\smoke_ext.py`（C++ 与 CUDA 各一个最小扩展）。

## 怎么用

无需任何开关：`ByteBPETokenizer.load()` 会自动尝试加载 `xingchen/fastbpe.dll`。
DLL 不存在 / 加载失败 → 自动退回原来的纯 Python 实现，行为不变。

```bash
# 重新编译（改了 bpe.c 之后）
fastbpe\build.cmd

# 校验一致性 + 测速（结果写到 logs/fastbpe_check.txt）
python fastbpe\verify.py
```

## 实测（2026-10-29，RTX 5060 笔记本 / CPU 同机）

- 一致性：308/308 段、44,205 个 token **完全一致**
- 编码 400 万字符（3,116,590 token）：纯 Python 9.99 s → C 2.39 s，**4.2×**
- 瓶颈已经转移到 Python 的正则预切分（`re.findall`），再往上提就把预切分也搬进 C

## 文件

- `bpe.c` — C 实现（哈希表存 (a,b)→rank，1<<17 个桶）
- `build.cmd` — MinGW 一条命令编译
- `verify.py` — 等价性校验 + 基准测试
- `../xingchen/fastbpe.py` — ctypes 包装（可选加速，失败即降级）
