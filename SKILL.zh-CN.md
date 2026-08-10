# SGLang PD 并行策略自动 Sweep（中文版）

> 本文件是 `SKILL.md`（英文）的中文对照版，内容严格对应；skill 运行时加载的是英文版，
> 两边修改需同步。

目标：给定任意 SLA（TTFT / TPOT）+ 负载画像（ISL / OSL）+ 模型，全自动扫出 SLA 内
吞吐最优的 Prefill 配置和 Decode 配置，再做 **QPS matching** 定 PD 配比，输出
**每 1000 卡 QPS**（= 1000 × 系统QPS ÷ 总卡数）。

## 0. 输入参数（开跑前全部确认）

| 参数 | 含义 | 默认 |
|---|---|---|
| `TTFT_SLO` | prefill SLA，P50 TTFT 上限 (ms) | 500 |
| `TPOT_SLO` | decode SLA，mean TPOT/ITL 上限 (ms) | 15 |
| `ISL` / `OSL` | 输入/输出长度画像 | 10000 / 700 |
| `MODEL` | 容器内模型路径 | 见 references/site.md |
| `IMG` | 服务镜像；**默认拉 latest**，把解析出的 tag+digest 记入结果 | latest |
| `PREFIX_RATIO` | 负载声明的共享前缀比例（0 = 完全独立 prompt） | 0 |
| `ENGINE` | 服务引擎（旋钮名与启动模板随引擎不同） | sglang |

用户只给 SLA 时其余用默认并明确告知。执行通道（SSH/MCP 工具名）、宿主机与容器路径、
容器命名、模型位置**一律见 `references/site.md`（私有文件，不随方法论公开）**。

## 0.5 Sweep Plan（每次执行前必须先给出，用户确认后才开跑）

任何 sweep 动作（包括追加补测）之前，先产出一份 **Sweep Plan**：

1. **环境快照**：GPU 是否空闲（`nvidia-smi`）、将使用的镜像 tag+digest、模型路径、磁盘余量。
2. **参数回显**：SLA（TTFT/TPOT）、ISL/OSL、判据（P50 还是 P90）。
3. **候选配置表**：P/D 各自的 config 列表——label、并行组合、chunk/MTP、并发梯队、
   **每条的推导依据**（§3.5 规则或硬件推论）。
4. **执行顺序与决策点**：哪些无条件跑、哪些取决于前序结果；剪枝规则写明。
5. **预算**：单 config 耗时估计（冷启动 1-6min + 每并发档 2-10min），总上限；超预算先砍谁。
6. **产出物**：results.tsv 路径、报告文件名（带时间戳）。

计划以简洁表格呈现，用户确认（或修改）后才执行；执行中偏离计划要说明原因；计划假设
中途破裂（镜像行为变化、OOM 边界移动）就停下修订，不闷头硬推。

## 1. 硬件先验（裁剪搜索空间；详见 references/hardware.md）

- **纯 PCIe 机型、无 NVLink**：GPU 间只有 PCIe（同 NUMA 内 PXB/NODE，跨 NUMA SYS）
  + RDMA NIC（卡数/显存/拓扑见 site.md 和 hardware.md）。
- **大 TP 是毒药**：TP 的逐层 all-reduce 走 PCIe。decode 访存密集、逐 token
  all-reduce——TP 越大通信占比越高，大 TP 会灾难性拖垮单卡吞吐。TP 不超过单 NUMA
  的 GPU 数。
- 跨 NUMA 的并行只能用 PP（只传激活）或 DPA/EP（通信可走 RDMA/A2A）。

## 2. P / D 可用旋钮（对照 server_args.py 核实）

`tp_size × pp_size = 单元卡数`。硬约束（源码 assert）：
- DPA：`--enable-dp-attention --dp-size N`，要求 **tp % dp == 0**；
  chunked_prefill_size 内部会除以 dp_size。
- EP：`--ep-size N`，要求 **ep × moe_dp_size == tp**（moe_dp 默认 1，通常 ep == tp，
  或配 `--moe-dp-size`）。
- FlashInfer A2A：`--moe-a2a-backend flashinfer` 仅在 **dp == tp 且开 DPA** 时合法。
- A2A 后端可选：`none, deepep, mooncake, nixl, mori, flashinfer, megamoe`。
- Decode 侧 radix cache（PD 模式 `--disaggregation-decode-enable-radix-cache`）
  **与投机解码互斥**——开 MTP 就不要开它。
- PP 与 context parallelism / elastic EP 互斥。
- CP（注意力上下文并行，`attn_cp_size`；约束 tp % (dp × cp) == 0）是**负载条件维度**：
  中等 ISL（~10k）下 attention 只占 prefill 计算的小头，CP 收益有限，且 CP 排斥 PP、
  等于放弃本平台最优的 P 形状族。仅在 ISL ≥ ~32k 时把 TP×CP 形状纳入 G2/G3 探针。
  **混合线性注意力模型（如 GDN+GQA）进一步降级**：CP 的收益——分摊 O(L²) 的 softmax
  attention——只作用于少数 GQA 层；线性注意力层为 O(L) 且其递归 state 沿序列存在
  串行依赖（CP 退化为状态传递链），框架对混合栈的 CP 支持通常缺失。且此类模型每请求
  state 为 O(1)，长上下文的显存压力论证同样不成立。
- SP（Megatron 式序列并行）在 SGLang 中不是独立旋钮——它内化于 TP 实现，无从扫描。

| 旋钮 | flag | P | D |
|---|---|---|---|
| TP / PP | `--tp-size N --pp-size M` | ✅ PP 是 prefill 主力 | PP 伤 TPOT，D 少用 |
| chunk | `--chunked-prefill-size N` | ✅ 核心旋钮 | 无关 |
| DPA | `--enable-dp-attention --dp-size N` | 长 ISL 下收益小 | ✅ decode 吞吐关键 |
| EP | `--ep-size N` | **权重驻留 regime 下** prefill 无收益：EP 依附 TP（ep×moe_dp==tp），开 EP 必先交纯 PP 避开的 TP 税，且其收益针对 decode 瓶颈——实测 EP1 vs EP8 prefill 无差。**offload regime 下判定翻转**（§3.4-1）：EP 把每卡专家足迹除以 N、直接削减 H2D 流量 | ✅ 摊薄 MoE 专家 |
| MTP | `--speculative-algorithm NEXTN --speculative-num-steps a --speculative-eagle-topk b --speculative-num-draft-tokens c` | 无关 | ✅ 头号杠杆，深度有甜点 |
| A2A | `--moe-a2a-backend flashinfer` | – | EP>1 时对比一次 |

MTP 记法 a/b/c = steps/topk/draft-tokens，如 3/1/4。新镜像若 `NEXTN` 报
"Unknown speculative algorithm"（它走插件注册），回退试 `EAGLE`。注意：0.5.16 上
NEXTN 的 `--speculative-eagle-topk > 1` 起服即崩（维度封死）；`--mem-fraction-static`
是 D 阶段的正式坐标维度（默认→0.95 一档；KV 池加大可推高饱和点；过高会加载期
OOM——SRVFAIL 即边界）。

## 3. 单压方法（P、D 分开测；QPS matching 的输入）

用 `sglang.bench_serving --dataset-name random --random-range-ratio 1` 对单实例压测：

- **P 单压**：`isl=ISL, osl=1`（纯 prefill）。radix 处理跟随负载声明：
  `PREFIX_RATIO=0` 时必须 `--disable-radix-cache`（等长 random prompt 并发会命中
  prefix cache，产出假的超低 TTFT——那是测量伪影，不是负载）；`PREFIX_RATIO>0` 时
  缓存命中**就是负载的一部分**：保持 radix 开启，用受控前缀数据集
  （generated-shared-prefix，按声明比例配置分组）使实测命中率与声明值一致。
  指标：**P50 TTFT 判 SLA，Input TPS 计吞吐**。P 单压天然高保真（prefill 就是全部工作）。
- **D 单压：一律用官方 pure-D fake KV 后端（完全高保真；0.5.15+ 已验证全链路）**：
  - D server 在正常 decode flags 之上加 `--disaggregation-mode decode
    --disaggregation-transfer-backend fake`；
  - bench_serving 用真实负载长度 `--random-input-len ISL --random-output-len OSL`，并加
    `--extra-request-body '{"bootstrap_host":"2.2.2.2","bootstrap_room":0}'`
    （"2.2.2.2" = 源码 `FAKE_BOOTSTRAP_HOST`，FakeKVReceiver 的 poll 直接返回 Success）。
  - 效果：完全跳过 prefill 计算，但每个请求**按真实 ISL 独立分配 KV**（内容是垃圾、
    性能语义正确）——每步 decode 读真实长度 KV、显存足额占用、并发上限真实。上游 main
    的 `sglang.benchmark.one_batch_server --fake-prefill` 就是同一机制的封装。
  - 注意：此模式 TTFT 无意义（只看 TPOT / Output TPS）；KV 内容为垃圾 → **MTP 接受率
    可能与真实文本不同**——用同模式 MTP off 基线的增益倍数做 sanity check，最终选型
    在真 PD 环境复核；server 对 fake 请求自动跳过 radix insert，无需关 radix。
  - 备用（仅 fake 路径异常时）：`--dataset-name generated-shared-prefix`
    （gsp-system-prompt-len=ISL + 开 radix）。缺点：共享前缀只存一份，显存占用/
    并发上限读数偏乐观。
  - **禁用短输入代理**（isl=128 之类）：每步 decode 只读一小截 KV，严重高估 D 容量，
    连相对排序都可能失真（DPA 这类访存型收益被低估）——已从流程移除。
- 指标：**mean TPOT/ITL 判 SLA，Output TPS 计吞吐**。并发梯队两个停止条件：
  ① FAIL 即停（单调假设）；② **吞吐平台期即停**——TPOT 仍达标但 outTPS 增益 <5%
  说明实例已饱和、增量并发全变排队（饱和实例的 TPOT 依然"达标"，排队全部堆进 TTFT）。
  **毛刺防护**：吞吐离散 >5% 的形状可能出现假回落误触平台期——顶部候选在 G4 阶段
  强制延梯一档再定操作点。另外起服 SRVFAIL ≠ 形状不可行：自动推导的 mem-fraction
  对权重贴边的形状过保守——按报错信息带显式 `--mem-fraction-static` 重试一次再判生死。
  PASS 中吞吐最高的档就是该配置的"操作点"，直接进 QPS matching。


### 可直接复制的命令（P 一档 / D 一档）

**P 单压**——server（示例形状 TP1/PP2、chunk 2048）+ c=2 一档：

```bash
python3 -m sglang.launch_server \
  --model-path $MODEL --tokenizer-path $MODEL $QUANT_ARGS \
  --tp-size 1 --pp-size 2 --chunked-prefill-size 2048 \
  --disable-radix-cache \
  --trust-remote-code --context-length 16384 --host 127.0.0.1 --port 30000

python3 -m sglang.bench_serving \
  --backend sglang --host 127.0.0.1 --port 30000 --model $MODEL \
  --dataset-name random --random-input-len 10000 --random-output-len 1 \
  --random-range-ratio 1 --num-prompts 8 --max-concurrency 2 --request-rate inf
# 用 "Median TTFT (ms)" 判 TTFT_SLO；用 "Input token throughput (tok/s)" 计吞吐
```

**D 单压（官方 pure-D）**——server（示例 TP1 + MTP 3/1/4 + mem 0.95）+ c=48 一档：

```bash
python3 -m sglang.launch_server \
  --model-path $MODEL --tokenizer-path $MODEL $QUANT_ARGS \
  --tp-size 1 --mem-fraction-static 0.95 \
  --speculative-algorithm NEXTN --speculative-num-steps 3 \
  --speculative-eagle-topk 1 --speculative-num-draft-tokens 4 \
  --disaggregation-mode decode --disaggregation-transfer-backend fake \
  --trust-remote-code --context-length 16384 --host 127.0.0.1 --port 30000

python3 -m sglang.bench_serving \
  --backend sglang --host 127.0.0.1 --port 30000 --model $MODEL \
  --dataset-name random --random-input-len 10000 --random-output-len 700 \
  --random-range-ratio 1 --num-prompts 96 --max-concurrency 48 --request-rate inf \
  --extra-request-body '{"bootstrap_host":"2.2.2.2","bootstrap_room":0}'
# 用 "Median ITL (ms)" 判 TPOT_SLO；用 "Output token throughput (tok/s)" 计吞吐；此模式 TTFT 无意义
```

换候选时只改并行/chunk/MTP 那几个 flag，其余保持不变。runone.sh 自动化的正是这套
序列（杀干净-等显存-起服-爬梯-判定）。

执行器 `scripts/runone.sh`（权威副本在 skill 里；开跑前按 site.md 写到
`$WORK_DIR/runone.sh` 并 `chmod +x`；WORK_DIR / MODELS_DIR 从 site.md 导出）：

```
env: IMG MODEL WORK_DIR MODELS_DIR TTFT_SLO TPOT_SLO [PORT CTX_LEN QUANT_ARGS]
用法: runone.sh <P|D> <label> <tp> <pp> <dpa> <ep> <chunk> <mtp> <isl> <osl> <conc_csv> -- <sglang extra args>
```

它负责：容器生命周期、杀干净残留 sglang、等显存归零、起 server（fatal 日志秒失败）、
逐档压测、PASS/FAIL 判定、追加 `auto/results.tsv`。tp/pp/dpa/ep/chunk/mtp 参数只是
记录用标签——真正生效的是 `--` 后的 sglang flags，两边必须一致。

## 3.4 Regime 扩展（触发时在种子推导前先套用）

§3.5 的基础规则隐含四个假设：权重装得进显存、中等 ISL（~10k）、独立 prompt、SGLang。
四种已知的 regime 变化会改写特定规则——推种子前逐项检查触发条件：

1. **权重 offload regime**——触发：W > 0.8 × GPU_MEM × 单机卡数。
   最小单元公式失效；选择变为 {切得更宽} vs {MoE 感知的 host offload}（routed expert
   的 w13/w2 及 scale 卸载到 NUMA 本地 pinned host 内存、独立 H2D copy stream、
   group/prefetch 调优）。**EP 对 prefill 的判定在此翻转**：权重驻留时 EP 对 prefill
   无益（见 §2）；offload 下 EP-N 把每卡专家足迹除以 N、直接削减 H2D 流量——
   DP×EP（如 TP1/DP8/EP8 + DeepEP 类 A2A）成为一等 P 形状族。坐标列表新增旋钮：
   offload group/num/prefetch、copy stream 数量/优先级、按 PCIe root 交错的 GPU
   顺序、NUMA 绑定。
2. **长上下文 regime**——触发：ISL ≳ 32k 且为 softmax 族注意力（GQA/MLA）。
   CP/PCP 重新成为 P 形状族，主要服务单请求 TTFT（吞吐通常仍属 DP/PP 族）；
   §2 的混合线性注意力降级仅适用于 GDN 类结构。
3. **前缀密集负载**——触发：PREFIX_RATIO > 0。测量协议按 §3（radix 开 + 受控前缀
   数据集）；操作点从此依赖命中率，报告需连带声明。
4. **引擎适配**——§2 旋钮表与 runone.sh 启动模板是 SGLang 专属。换引擎（如 vLLM）
   需映射：tp/pp/dp/ep 尺寸；MoE 后端（`--moe-backend deep_gemm`）与 A2A
   （`--all2all-backend deepep_v2`）；offload/环境旋钮为引擎特有（VLLM_* 变量）。
   方法论本体（单压隔离、坐标下降、G1-G4、matching）与引擎无关，只需移植执行器的
   启动模板与旋钮映射。

## 3.5 新模型种子推导（结论是 per-model 的；每次换模型重推）

调优结论是 **per-model** 的（依赖权重大小、层数、MoE 结构、MTP head 质量）。
新模型按第一性原理推种子：

1. **显存可行域**：W = 权重字节数（nvfp4 ≈ 参数量×0.55）；单卡显存 GPU_MEM（见
   site.md）。最小单元 = ceil(W / (0.8×GPU_MEM)) 卡（留 KV+激活）；
   **KV 余量 = GPU_MEM×卡数 − W** 决定 decode 并发上限；余量 <20GB/卡 的形状
   decode 必饱和，直接剪。
2. **P 种子**：纯 PP 优先（平台定律）。PP 深度候选 = 可行的 {浅,中,深} 三档
   （单卡放得下就从 TP1/PP1 起，靠副本而非 PP 深度扩展）。chunk 起点由
   **流水线填充启发式**给出，推导如下：

   *模型。* 一条 ISL token 的 prompt 按 chunk 切块得到每请求 M = ISL/chunk 个微批；
   操作点并发为 c 时，在场微批总数 N ≈ c×M。pp_size 段流水线处理 N 个微批、每段每批
   耗时 t，完成时间为 (N + pp_size − 1)·t，理想满载为 N·t，故流水效率 =
   N/(N+pp_size−1)，空泡占比 = (pp_size−1)/(N+pp_size−1)。与此独立地，每个块边界要支付一次固定的调度/启动开销
   t_o（本级别机器实测约 10 ms/块），总开销随 M 线性增长。

   *权衡。* 块切小 → N 增大、空泡占比下降，但每多一块多付一次 t_o；块切大则相反。
   取 N ≈ 2×pp_size 时效率达 2·pp_size/(3·pp_size−1) ≈ 2/3 以上——这是效率曲线的
   拐点：再增加块数，每块换回的空泡缩减不足 5%，而 t_o 照付。解
   c×(ISL/chunk) = 2×pp_size 即得种子：

   **chunk_seed ≈ c_est × ISL / (2 × pp_size)**

   其中 c_est 为 P 单元的预期操作点并发（天然较低，通常 1-3；未知时取 2——
   后续爬山会以 1-2 个测点的代价纠正错误的种子）。

   *极限行为（自检）。* pp_size=1（无流水线）：空泡项消失，公式退化为 chunk ≈ c·ISL/2——
   即单卡 prefill 在 c=1 时最优趋向"完全不切块"，与实测一致（没有流水线可填、
   没有并发请求可轮转时，切块是纯开销）。深 pp_size 低 c：公式给出小块，以每块开销
   换取流水线填充——同样与实测一致。该公式已在本硬件的两种 regime 上验证；
   深流水场景下公式预测的种子与实测甜点重合，未消耗额外爬山步。

   *注意。* 甜点随操作点并发漂移（公式中的 c 是该操作点自身的并发）：在 c=2 下
   爬出的 chunk 对 c=3 不是最优——这正是 G1 阶段 chunk×并发对角探针对 prefill
   为必选项的原因。
3. **D 种子**：单元卡数取"权重装下 + KV 余量 ≥30GB/卡"的最小值；MoE → EP=TP 加
   DPA=2 对照；dense → DPA 是唯一杠杆。模型带草稿头就从 MTP 3/1/4 起，没有就 off
   （NGRAM 作替补）。
4. 之后走同一套搜索算法（§4）——**方法论永远不换，只换种子**。同硬件的跨模型规律
   （P 侧 PP>TP、D 侧低TP+DPA+EP、大TP毒药）可以继承，具体数值不行。

## 4. 搜索算法：坐标下降 + 全局加固（LLM 在环执行，规则显式化保证可复现）

不跑固定候选表；逐点推进，每测完一个点读 results.tsv 再定下一个：

```
1. 种子:   每 phase 2-4 个, 由 §3.5 从模型属性+硬件第一性原理推导
2. 初筛:   短爬梯快测(D 同样走 fake pure-D, 只爬 1-2 档); 淘汰明显落后者
           (单卡吞吐 < 头名 60%); 留 top-K (K≈2)
3. 下降:   对冠军按敏感度降序逐维爬山:
           P: pp深度 → chunk → 并发    D: mtp深度 → dpa → ep → mem-fraction → 并发
           沿一维 ±1 档; 改进(单卡吞吐↑且PASS) → 同方向继续; 退步 → 换下一维;
           整轮无改进 → 局部最优, 停
4. 边界:   冠军落在任何扫描边界上 → 沿该维再扩一档(防漏扫);
           通用收敛准则: 连续改进但单步增益 <5% → 收敛, 停止外扩(记入推断剪枝)
5. 决赛:   top-2 完整爬梯定操作点 + 临界点(|指标-SLO|<5%) 3 次取中位, 且相邻并发档
           一起测——操作点可能移到邻档
6. 预算闸: 短爬梯 ~5-10min/点, 完整爬梯 ~15-25min/点; 预算过半后缩步长、只推进冠军分支
```

这是 successive halving + 坐标下降：初筛用短爬梯省时，决赛完整爬梯定值，测量口径
全程一致（fake pure-D）；坐标下降只测 O(维度×步数) 个点而非笛卡尔积；边界扩展防
先验剪过头。

### 全局性加固（坐标下降收敛后必做；预算 ≈ 主搜索的 50%）

坐标下降只给"单维局部最优"证书；全局最优可能藏在维度交互、未扫旋钮、噪声、窄峰里。
四步收紧（按性价比排序）：

```
G1 交互探针:   在冠军周围测 2-3 个"两维同时动"的点(单维爬山走不到的对角线)。
              P 侧 chunk×并发为必选——chunk 甜点随操作点并发漂移(实测 c=2 与 c=3
              的最优差一整档), 单并发爬 chunk 必漏高并发最优。任一探针胜出 →
              以它为新起点重启下降。
G2 未扫旋钮点名: §2 表里冠军没动过的旋钮每个试一档(MTP topk>1、attention backend、
              mem-fraction↑、A2A 后端…)。每个一枪; SRVFAIL 也是信息。
G3 远点 ε-探索: 1-2 个与冠军结构完全不同的形状(不同单元卡数/不同并行家族)——
              防种子锁死在错误的形状家族。
G4 细网格审计:  冠军邻域半步加密(如并发 ±8、chunk 中点), 确认不是窄峰边缘;
              前两名相差 <5%(噪声量级)时, 双方各再复测一次才定名次。
```

排程说明：G1 和 G4 的探针坐标依赖**定稿后的冠军**——必须等下降收敛后再发（挑战移动靶
浪费探针；冠军一移位这些点就作废）。G3 远点和不依赖冠军坐标的 G2 探针（全新形状族、
版本/后端点名）在机器窗口紧张时可以提前并入更早的批次，省墙钟。

报告新增**置信度一节**：① 局部最优证书（每维内点/收敛证据）；② G1-G4 结果；
③ **roofline 上界对比**——按模型 activated FLOPs 对硬件峰值估算 utilization
（标明假设）。utilization 高 → 全局残余空间被数学压住；低 → 大头在 flag 空间之外
（kernel/版本）。诚实声明全局最优不可证——报告已知最优 + 剩余空间可能藏身的清单。

路线图（v3，尚未实现）：用一轮 campaign 的实测点标定一个小型性能模型
（roofline + 流水线空泡项 + 实测每块开销 + MTP 接受率），此后**新 SLA 用解析求解 +
2-3 个 spot check 验证**替代整轮 sweep——一次标定、多 SLA 复用。第一性原理已负责
选形状族；模型负责外推操作点；实测只保留阈值验证。

### Phase P — Prefill（判 TTFT_SLO，按 Input TPS/卡 排名）
1. 种子按 §3.5：纯 PP 优先、{浅,中,深} 可行深度；单卡模型从 TP1/PP1 起（副本扩展）。
   补 1-2 个含 TP 的形状做对照。
2. chunk 爬山：从中档（~2048 或 SLO 推算值）起双向各一步；甜点随 PP 深度变化，
   高并发下大 chunk 甚至可能降 TTFT——让数据说话，不预设方向。
3. 并发用低档（1,2,3,4），找 TTFT 拐点。
4. 剪枝：不跑跨 NUMA 的大 TP；EP 对 prefill 基本无益，预算紧时后置。
5. **KV 余量候选**：额外保留一个 TTFT 余量 ≥（KV 传输残差 + 抖动）的 P 操作点，
   供保守方案用。

### Phase D — Decode（判 TPOT_SLO，按 Output TPS/卡 排名）
1. 种子按 §3.5：取"权重装下 + KV 余量 ≥30GB/卡"的最小单元；MoE → EP=TP 加 DPA=2
   对照；dense → 只有 DPA。
2. MTP 深度扫（模型带草稿头时）：从 3/1/4 双向爬（1/1/2 与 4/1/5）；TPOT 反升
   （接受率崩）即停；没有草稿头 → off，NGRAM 作替补。
3. 并发爬梯 32,48,64,96,128…；FAIL 或平台期（+<5%）即停。
4. EP>1 的形状胜出时补一枪 FlashInfer A2A（需 dp==tp + DPA）。DP-MLP 保持默认 skip，
   绝不开强同步。
5. ⚠️ colocated（非 disagg）模式下，DPA + MTP + chunked-prefill 联合路径在部分版本
   触发 kernel 级崩溃（Triton illegal memory access）；fake pure-D 模式无此路径。
   colocated 扫该组合前先单点验证。

### 防漏扫机制（每个 phase 收尾必做）

先验剪枝可能剪掉真最优。两个兜底：

1. **边界检测**：冠军若落在**扫描边界**上（最深 PP、最大 EP、最深 MTP、最高 PASS
   档之上没有数据），真最优可能在界外——沿该维扩一步，直到冠军是内点。
2. **临界复测**：|指标 − SLO| < 5%×SLO 的行单发不可信（如 15ms SLO 下 TPOT 在
   14.25-15.75ms）——复测 3 次取中位；两个候选吞吐相差 <5% 时同样处理。

接受不扫、但必须写进报告覆盖声明的次级维度：受支持版本上的 MTP topk>1、
deepep/megamoe A2A、`enable_dynamic_chunking`、moe_dp_size 组合、chunk×PP 全交互
（只有冠军做 chunk 精调）、正式坐标之外的 mem-fraction/调度参数。剪枝分两级——
**数据剪枝**（有实测 FAIL，安全）与**推断剪枝**（外推——报告里写明依据）。

### Phase R — QPS matching → PD 配比 → 每 1k 卡 QPS（`scripts/analyze.py`）
```
python3 analyze.py auto/results.tsv --isl ISL --osl OSL
```
- 每个 PASS 配置取操作点：P 单元 QPS = InputTPS/ISL，D 单元 QPS = OutputTPS/OSL。
- 枚举 xP:yD（既约比，x,y ≤ 8）：系统 QPS = min(x·qps_P, y·qps_D)；
  每 1k 卡 QPS = 1000 × QPS ÷ (x·gpus_P + y·gpus_D)。
- 输出 top-3 组合及瓶颈侧（P-bound / D-bound）。这是稳态近似，未计 KV 传输。
- **可部署性约束**：配比的好坏还要看它能不能铺到真实节点上。优先选组卡数**等于或
  整除节点卡数**（如 8）的配比——节点整装组（如 5P:3D = 8 卡）就是一台自包含 PD
  机器：KV 传输不出机、每机一个 router、同构运维。铺不平的配比（如 13 卡/组）在
  机群级用**同构节点池**部署（整机 P 副本节点 : 整机 D 副本节点按目标比例），
  绝不把一个组劈到两台机器上。
- **副本数注意**：单机跑很多单卡副本会放大 host 侧开销（每个 server 一套
  tokenizer/scheduler 进程、共享 PCIe/内存带宽）。每个副本的 CPU 绑到其 GPU 所在
  NUMA、NIC 用 NUMA 本地的。单压数字默认"副本间无干扰"——**报产线容量前必须做
  满副本同压验证**（预期损失几个百分点）。
- **报告必须给三个方案**：① 稳态最优；② **节点对齐运维解**——组卡数恰好铺满节点的
  最优配比（差距 ~5% 内优先推荐）；③ **KV 余量保守解**——换用有 TTFT 余量的 P
  操作点（真 PD 会给 TTFT 加几十毫秒的 KV 传输残差——要实测；贴线 P 配置到生产
  必超 SLA）。

### Phase V —（可选）真 PD 分离终验
单机 8 卡可以容纳如 1P(4)+1D(4) 来实测 KV 传输开销：
- P server：`--disaggregation-mode prefill --disaggregation-transfer-backend nixl`
  （或 mooncake/mooncake_tcp）+ `--disaggregation-ib-device <同 NUMA 的 NIC>`；
  D server：`--disaggregation-mode decode ...`。
- 路由：`python -m sglang_router.launch_router --pd-disaggregation --prefill
  http://127.0.0.1:P1 [bootstrap_port] --decode http://127.0.0.1:P2`，对 router 跑
  Poisson 负载（`--request-rate` ≈ analyze.py 给出系统 QPS 的 90%）。
- 终验以 **P90** 论：closed-loop → Poisson 通常显著放大尾延迟；产线容量按 P90 规划。

## 5. 报告

- results.tsv → HTML 报告（**白色背景**），文件名带时间戳
  `<name>-sweep-YYYYMMDD-HHMM.html`。
- 头条：最优 P 配置、最优 D 配置、PD 配比、每 1k 卡 QPS（三方案）；完整数据表
  （PASS/FAIL 标色）、镜像 tag+digest、SLA 参数、置信度一节、覆盖声明。
- 只差 <3ms 的 FAIL 属临界——Poisson P90 会放大；选型留余量。

## 6. 运维红线（每一条都交过学费）

1. **换配置前杀干净**：容器内 `pkill -9 -f sglang`，然后轮询 `nvidia-smi` 等显存
   <2GB 再起新 server（PP 模式 scheduler 回收慢；残留 = 假 OOM）。
2. **server 日志必须用容器内路径**；重定向到宿主机路径会静默失败，server 永远起不来。
3. **绝不干等**：轮询 `/health` 的同时 grep server 日志抓
   `Traceback|OOM|NCCL error`——命中立即判 SRVFAIL；任何等待超 1 分钟先查日志。
4. PP 冷启动慢（35s+）；health 检查给足 600s。
4b. **CUDA graph 覆盖核查**（decode 配置）：server UP 后 grep 日志中的捕获列表
    （`Capture ... bs=[...]`）。捕获范围随剩余显存自适应——KV 越紧的形状捕获上限越低。
    若某配置的操作点并发（按 DP rank 折算）达到或超过捕获上限，decode 会静默回退
    eager，实测 TPOT 不再代表该配置的真实能力：要么加一枪 `--cuda-graph-max-bs`
    的 G2 探针，要么在结果行注记。另：当前版本引擎自动禁用 prefill CUDA graph
    （且与 DP attention 不兼容）——P 侧全程 eager，视为版本封闭维度。
5. 日志/结果文件名带时间戳；输出行加 `[HH:MM:SS]` 前缀，防读旧数据。
6. 开跑时 `docker pull $IMG` 并记录解析出的 digest；换镜像后先用已知配置 smoke
   一次，再信跨 campaign 对比。
7. 共享机器：先用 `nvidia-smi` 确认没有别人的任务。他人的自动化 pipeline 可能
   **链式排队**（一个结束下一个立刻起）——campaign 间的空档 ≠ 空闲；等其调度/驱动
   进程全部退出再开跑，否则双方数据都被污染。
8. `IMG=latest` 是滚动 tag：开跑记录
   `python3 -c "import sglang; print(sglang.__version__)"` + digest；跨 campaign
   对比先核版本。
9. 经 MCP SSH 通道起后台任务必须用子 shell 包裹：`( setsid nohup ... & )`——否则
   通道挂 120s 后报假失败（任务实际在跑；用新连接验证）。
10. 不要用 `pgrep -f <批名>` 做"等前一批完成"的 watcher——watcher 自己的命令行就含
    批名，会自匹配。改轮询 driver.log 里的 BATCH_DONE 标记。
11. 方法论泛化已验证：同一套种子推导+搜索规则，在"多卡切分 regime"（大模型，PP/EP）
    和"单卡副本 regime"（小模型，TP1/PP1）都收敛到正确形状家族——换模型只重推种子，
    永远不改规则。
