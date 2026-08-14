# Learning-CUDA

这是我的 GPU 算子实现与学习记录。目前已完成 NVIDIA 和 MetaX 平台下的
RMSNorm 与 Flash Attention，并在两个平台上通过全部正确性测试。

## 完成内容

### RMSNorm

- 支持 `float` 和 `half`；
- 使用 `float` 完成中间累加，降低半精度输入的累加误差；
- 一个线程块处理一行输入；
- 线程并行计算平方和，并通过共享内存完成归约；
- 归一化和权重缩放由线程并行写回。

### Flash Attention

- 支持 `float` 和 `half`；
- 支持 causal masking；
- 支持 GQA（Grouped Query Attention）；
- 一个线程块处理一个 Query 行；
- 并行计算不同 Source 位置的 QK 分数；
- 使用共享内存保存 QK 分数，避免重复计算；
- Softmax 采用减去最大值的稳定计算方式；
- 输出维度由线程并行完成加权求和；
- 当所需共享内存超过当前 GPU 单块上限时，自动使用串行兜底 Kernel。

主要实现位于：

```text
src/kernels.cu
src/kernels.maca
```

## 本地环境

| 项目 | 配置 |
| --- | --- |
| 操作系统 | Ubuntu 22.04（WSL2） |
| GPU | NVIDIA GeForce RTX 4060 Laptop GPU |
| 显存 | 8 GB |
| NVIDIA 驱动 | 595.71 |
| CUDA Toolkit | 13.2 |
| NVCC | 13.2.51 |
| C++ 标准 | C++17 |
| GNU Make | 4.3 |

## 编译与测试

在项目根目录运行：

```bash
make clean
make VERBOSE=true
```

只测试 RMSNorm：

```bash
SKIP_ATTENTION=1 make VERBOSE=true
```

只测试 Attention：

```bash
SKIP_RMS_NORM=1 make VERBOSE=true
```

## 验证结果

- NVIDIA RTX 4060 Laptop GPU：54 项验证全部通过；
- NVIDIA RTX 4090 D：54 项验证全部通过；
- MetaX C500：54 项验证全部通过；
- RMSNorm 共 13 组测试，`float` 和 `half` 均通过；
- Attention 共 14 组测试，`float` 和 `half` 均通过；
- causal masking 和 GQA 测试通过；
- 完整测试进程退出码为 `0`。

## MetaX C500

适配环境为 MetaX C500、MACA 3.5.3.20，使用 `mxcc` 编译器。MACA 环境配置
完成后，在项目根目录运行：

```bash
make clean PLATFORM=metax
make PLATFORM=metax VERBOSE=true
```

MetaX 实现使用 `mcMalloc`、`mcMemcpy` 等 MACA Runtime API，并保持与 NVIDIA
版本相同的并行归约、稳定 Softmax、GQA 和 causal masking 逻辑。为兼容不同
设备的单块共享内存限制，Attention 在共享内存需求超过 48 KiB 时使用串行
兜底 Kernel。

## RTX 4090 D

在训练平台的 NVIDIA GeForce RTX 4090 D 上完成了独立验证：

| 项目 | 配置 |
| --- | --- |
| GPU | NVIDIA GeForce RTX 4090 D |
| 显存 | 24 GB |
| NVIDIA 驱动 | 570.124.06 |
| CUDA Toolkit | 12.8.61 |
| 最新验证提交 | `8eb40cb` |

平台的 CUDA 路径未加入默认环境变量，使用以下命令编译并运行：

```bash
PATH=/usr/local/cuda-12.8/bin:$PATH \
LD_LIBRARY_PATH=/usr/local/cuda-12.8/lib64:$LD_LIBRARY_PATH \
make clean

PATH=/usr/local/cuda-12.8/bin:$PATH \
LD_LIBRARY_PATH=/usr/local/cuda-12.8/lib64:$LD_LIBRARY_PATH \
make VERBOSE=true
```

完整测试共 54 项，全部通过，失败数为 0，进程退出码为 0。Attention 第 14
组测试中，`float` 平均耗时约 32.09 ms，`half` 平均耗时约 22.82 ms。

## CUDA 13.2 兼容说明

本地 CUDA 13.2 运行库与预编译的 `tester/tester_nv.o` 存在
`cudaGetDeviceProperties_v2` 链接符号差异。本地验证时使用了仅位于 `/tmp`
的兼容链接对象，该对象未加入源码，也未提交到仓库。

在 CUDA 12.8 环境中可以直接使用项目 Makefile：

```bash
export PATH=/usr/local/cuda-12.8/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.8/lib64:$LD_LIBRARY_PATH

make clean
make VERBOSE=true
```

## 其他 NVIDIA GPU

RTX 4090（Ada，`sm_89`）：

```bash
make clean
make CFLAGS="-std=c++17 -O0 -arch=sm_89" VERBOSE=true
```

RTX 5090（Blackwell，`sm_120`，需要 CUDA 12.8 或更新版本）：

```bash
make clean
make CFLAGS="-std=c++17 -O0 -arch=sm_120" VERBOSE=true
```

如果 5090 使用预编译测试器时出现以下错误，则测试器对象本身可能未包含
Blackwell 目标代码：

```text
no kernel image is available for execution on the device
```

## 性能记录

以下结果来自 RTX 4060 Laptop GPU，用于记录并行优化前后的本机变化：

| 测试用例 | 基础实现 | 并行实现 | 加速比 |
| --- | ---: | ---: | ---: |
| Attention 13，float | 61 ms | 19 ms | 3.2x |
| Attention 13，half | 59 ms | 11 ms | 5.4x |
| Attention 14，float | 239 ms | 89 ms | 2.7x |
| Attention 14，half | 149 ms | 79 ms | 1.9x |

不同 GPU、驱动版本和运行状态下的耗时会有所变化。
