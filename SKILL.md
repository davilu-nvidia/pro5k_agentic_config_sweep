---
name: pro5k-sgl-sweep
description: Fully-automatic SGLang inference performance sweep on the pro5k server (8x RTX Pro Blackwell 73GB, PCIe-only, no NVLink). Given ANY SLA (TTFT/TPOT) and workload (ISL/OSL), sweeps prefill/decode parallel configs (TP/PP/DPA/EP/MTP/chunk), picks the best P and D configs, does QPS matching for the PD ratio, and reports max throughput per 1000 GPUs (千卡QPS). Use when asked to tune, sweep, or optimize SGLang serving performance on pro5k, or to find the best config under an SLA.
---

# pro5k SGLang PD 并行策略自动 Sweep

目标：给定任意 SLA（TTFT / TPOT）+ 负载画像（ISL / OSL）+ 模型，全自动扫出 SLA 内吞吐最优的
Prefill 配置、Decode 配置，再做 **QPS matching** 定 PD ratio，输出 **千卡 QPS**
（= 1000 × 系统QPS ÷ 总GPU数）。

## 0. 输入参数（开跑前必须确认齐全）

| 参数 | 说明 | 示例默认 |
|---|---|---|
| `TTFT_SLO` | Prefill SLA，P50 TTFT 上限 (ms) | 500 |
| `TPOT_SLO` | Decode SLA，mean TPOT/ITL 上限 (ms) | 15 |
| `ISL` / `OSL` | 输入/输出长度画像 | 10000 / 700 |
| `MODEL` | 容器内模型路径 | 见 references/site.md |
| `IMG` | SGLang 镜像，**默认拉最新** `lmsysorg/sglang:latest`；开跑时 `docker pull` 并把实际 tag+digest 记入结果 | latest |

用户只给 SLA 时其余用默认并明确告知。执行通道（SSH/MCP 工具名）、宿主机与容器路径、
容器命名、模型位置等**站点信息一律见 `references/site.md`（私有文件，不随方法论公开）**。

## 0.5 Sweep Plan（每次执行前必须先给出，用户确认后才开跑）

任何 sweep 动作（包括追加补测）之前，先产出一份 **Sweep Plan** 给用户确认，包含：

1. **环境快照**：GPU 是否空闲（`nvidia-smi`）、将使用的镜像 tag+digest、模型路径、宿主机磁盘余量。
2. **参数回显**：SLA（TTFT/TPOT）、ISL/OSL、判据（P50 还是 P90）。
3. **候选配置表**：P 阶段和 D 阶段各自的 config 列表——label、并行组合、chunk/MTP、
   并发梯队、**每条的推导依据**（§3.5 规则或硬件推论）。
4. **执行顺序与决策点**：哪些 config 无条件跑，哪些取决于前序结果（如"TP1/PP4 PASS 且
   余量>30% 才追加 chunk1536"），剪枝规则写明。
5. **预算**：单 config 耗时估计（冷启动 1-6min + 每并发档 2-10min），总时长上限；
   超预算时先砍哪些低优先级 config。
6. **产出物**：results.tsv 路径、报告文件名（带时间戳）。

计划以简洁表格呈现。用户确认（或修改）后才执行；执行中偏离计划（新增/跳过 config）要说明原因。
中途发现计划假设不成立（如镜像行为变化、OOM 边界移动），停下来修订计划再继续，不闷头改。

## 1. 硬件先验（裁剪搜索空间，详见 references/hardware.md）

- **无 NVLink 的 PCIe 机型**：GPU 间只有 PCIe（同 NUMA 内 PXB/NODE，跨 NUMA SYS）+ RDMA NIC（具体卡数/显存/拓扑见 site.md 与 hardware.md）。
- **大 TP 是毒药**：TP 逐层 all-reduce 走 PCIe。decode 访存密集、逐 token all-reduce，TP 越大通信占比越高，大 TP 会灾难性拖垮单卡吞吐。TP 不超过单 NUMA 的 GPU 数，且 TP 组必须落在同一 NUMA 的 GPU 集合内。
- 跨 4 卡边界的并行只用 PP（只传激活）或 DPA/EP（通信可走 RDMA/A2A）。

## 2. P / D 各自可用的并行旋钮（源码 server_args.py 已核实）

`tp_size × pp_size = 单元卡数`。硬约束（源码 assert）：
- DPA：`--enable-dp-attention --dp-size N`，要求 **tp % dp == 0**；chunked_prefill_size 内部会除以 dp_size。
- EP：`--ep-size N`，要求 **ep × moe_dp_size == tp**（moe_dp 默认 1 → 通常 ep == tp 或用 `--moe-dp-size` 配）。
- FlashInfer A2A：`--moe-a2a-backend flashinfer` 仅在 **dp == tp 且开 DPA** 时合法。
- A2A 后端可选：`none, deepep, mooncake, nixl, mori, flashinfer, megamoe`。
- Decode radix cache（PD 模式 `--disaggregation-decode-enable-radix-cache`）**与投机解码互斥**——开 MTP 就别开它。
- PP 与 context parallelism / elastic EP 互斥。

| 旋钮 | flag | P 适用 | D 适用 |
|---|---|---|---|
| TP / PP | `--tp-size N --pp-size M` | ✅ PP 是 prefill 主力 | PP 伤 TPOT，D 少用 |
| chunk | `--chunked-prefill-size N` | ✅ 核心旋钮 | 无关 |
| DPA | `--enable-dp-attention --dp-size N` | 长 ISL 下无益 | ✅ decode 吞吐关键 |
| EP | `--ep-size N` | prefill 通常无收益 | ✅ 摊薄 MoE 专家 |
| MTP | `--speculative-algorithm NEXTN --speculative-num-steps a --speculative-eagle-topk b --speculative-num-draft-tokens c` | 无关 | ✅ 头号杠杆，深度有甜点 |
| A2A | `--moe-a2a-backend flashinfer` | – | EP>1 时对比一次 |

MTP 记法 a/b/c = steps/topk/draft-tokens，如 3/1/4。新镜像若 `NEXTN` 报
"Unknown speculative algorithm"（它走插件注册），回退试 `EAGLE`。注意：0.5.16 上 NEXTN 的
`--speculative-eagle-topk >1` 起服即崩（维度封死）；`--mem-fraction-static` 是 D 阶段正式
坐标维度（默认→0.95 一档，KV 池扩大可推高饱和点；过高会加载期 OOM，SRVFAIL 即边界）。

## 3. 单压方法（P、D 分开测，这是 QPS matching 的输入）

用 `sglang.bench_serving --dataset-name random --random-range-ratio 1` 对单实例压测：

- **P 单压**：`isl=ISL, osl=1`（纯 prefill）。必须 `--disable-radix-cache`（等长 random prompt 并发会命中
  prefix cache，产出假的超低 TTFT）。指标：**P50 TTFT 判 SLA，Input TPS 计吞吐**。
  P 单压天然高保真（prefill 就是全部工作）。
- **D 单压：一律用官方 pure-D fake KV 后端（完全高保真；0.5.15+ 已验证全链路支持）**：
  - D server 在正常 decode flags 基础上加 `--disaggregation-mode decode
    --disaggregation-transfer-backend fake`；
  - 压测端 bench_serving 用真实负载长度 `--random-input-len ISL --random-output-len OSL`，并加
    `--extra-request-body '{"bootstrap_host":"2.2.2.2","bootstrap_room":0}'`（"2.2.2.2" =
    源码 `FAKE_BOOTSTRAP_HOST`，FakeKVReceiver 的 poll 直接返回 Success）。
  - 效果：完全跳过 prefill 计算，但每个请求按真实 ISL **独立分配 KV**（内容是垃圾、性能语义正确）——
    每步 decode 读真实长度 KV、KV 显存足额占用、并发上限真实。上游 main 的
    `sglang.benchmark.one_batch_server --fake-prefill` 就是同一机制的封装。
  - 注意：此模式 TTFT 无意义（只看 TPOT/Output TPS）；KV 内容为垃圾 → **MTP 接受率可能与真实
    文本不同**，sanity check 用同模式下 MTP off 基线的增益倍数是否合理，最终选型建议真 PD 复核；
    server 对 fake 请求自动跳过 radix insert，无需关 radix。
  - 备用（仅 fake 路径异常时）：`--dataset-name generated-shared-prefix`（gsp-system-prompt-len=ISL
    + 开 radix）。缺点是共享 KV 只存一份 → 显存占用/并发上限偏乐观。
  - **禁用短输入代理**（isl=128 之类）：每步 attention 只读零头长度的 KV，严重高估 D 容量，
    连相对排序都可能失真（DPA 等访存型收益被低估）——已从流程移除。
- 指标：**mean TPOT/ITL 判 SLA，Output TPS 计吞吐**。并发梯队逐档爬，两个停止条件：
  ① FAIL 即停（单调假设）；② **吞吐平台期即停**——TPOT 仍 PASS 但 outTPS 增益 <5% 说明
  实例已饱和、增量并发全变成排队（饱和实例 TPOT 依然达标，排队全部堆进 TTFT），再爬只浪费时间。取 PASS 中吞吐最高档为"操作点"，其数字直接进 QPS matching。

执行器 `scripts/runone.sh`（权威副本在 skill 里，开跑前写入 site.md 指定的 `$WORK_DIR/runone.sh`
并 `chmod +x`；env 需带上 site.md 里的 WORK_DIR/MODELS_DIR）：

```
env: IMG MODEL TTFT_SLO TPOT_SLO [PORT]
用法: runone.sh <P|D> <label> <tp> <pp> <dpa> <ep> <chunk> <mtp> <isl> <osl> <conc_csv> -- <sglang extra args>
```

它负责：起容器、杀干净旧 sglang、等显存清零、起 server（扫日志抓 fatal 秒失败）、逐档压测、
按 SLA 打 PASS/FAIL、追加 `auto/results.tsv`。tp/pp/dpa/ep/chunk/mtp 参数只是记录用的标签，
真正生效的是 `--` 后你构造的 sglang flags——两边必须一致。

## 3.5 新模型种子推导（先验只对同模型有效，换模型从这里重新推）

调优结论是 **per-model 的**（依赖权重大小、层数、MoE 结构、MTP head 质量），
换模型一律按第一性原理重推种子：

1. **显存可行域**：W = 权重字节数（nvfp4 ≈ 参数量×0.55），单卡显存 GPU_MEM（见 site.md）。
   最小单元 = ceil(W / (0.8×GPU_MEM)) 卡（留 KV+激活）；**KV 余量 = GPU_MEM×卡数 − W** 决定 decode 并发上限，
   余量 <20GB/卡 的形状 decode 必饱和，直接剪。
2. **P 种子**：纯 PP 优先（平台铁律）。PP 深度候选 = 显存可行的 {最小,中,大} 三档
   （小模型单卡放得下就从 TP1/PP1 起，此时 P 扩展靠加副本而非加深 PP）；
   chunk 起点 ≈ SLO_ms × 预估单卡 prefill TPS / 1000 / PP深度，按爬山调。
3. **D 种子**：单元卡数取"权重装下 + KV 余量 ≥30GB/卡"的最小值；MoE 模型 EP=TP、
   补 DPA=2 对照；dense 模型无 EP，DPA 是唯一杠杆。MTP 有 head 就从 3/1/4 起扫，
   没有就 off（NGRAM 可作替补）。
4. 之后走同一套搜索算法（§4）——**方法论不换，只换种子**。同硬件的跨模型规律
   （PP>TP for P、低TP+DPA+EP for D、大TP毒药）可以继承，具体数值不能。

## 4. 搜索算法：坐标下降 + 全局加固（LLM 在环执行，规则显式化保证可复现）

不用固定候选表全跑，按下面的启发式推进；每跑完一个点读 results.tsv 再定下一个点：

```
1. 种子:   每 phase 取 2-4 个种子, 由 §3.5 从模型属性+硬件第一性原理推导
2. 初筛:   短爬梯快测(D 同样走 fake pure-D, 只爬 1-2 档), 淘汰明显落后者(吞吐/卡 < 冠军 60%), 留 top-K(K≈2)
3. 坐标下降: 对 champion 沿敏感度降序的维度做爬山:
             P: pp深度 → chunk → 并发    D: mtp深度 → dpa → ep → mem-fraction → 并发
             沿一维 ±1 档, 改进(吞吐/卡 ↑ 且 PASS)→ 同方向继续; 退步 → 换下一维;
             一轮全维度无改进 → 局部最优, 停
4. 边界扩展: champion 任一维在已扫边界上 → 沿该维再扩一档(防漏扫);
             通用收敛准则: 沿一维连续改进但单步增益 <5% → 视为收敛, 停止扩档(记入推断剪枝)
5. 决赛:   top-2 完整爬梯定操作点 + 临界点(|指标-SLO|<5%) 3 次取中位;
             临界复测时把相邻并发档一起测——操作点可能移位(贴线档的中位值确认后, 邻档反而更优)
6. 预算闸: 短爬梯 ~5-10min/点, 完整爬梯 ~15-25min/点; 预算过半时砍步长、只推进 champion 分支
```

这是 successive halving + coordinate descent：初筛用短爬梯省时，决赛完整爬梯定值，测量口径全程一致（fake pure-D）；坐标下降只测 O(维度×步数) 个点而非笛卡尔积；边界扩展防先验剪过头。

### 全局性增强（坐标下降收敛后必做，预算 ≈ 主搜索的 50%）

坐标下降只给"单维局部最优"证书，全局最优可能藏在维度交互、未扫维度、噪声和窄峰里。
四步收紧（按性价比排序）：

```
G1 交互探针: 在 champion 周围测 2-3 个"两维同时动"的点(单维爬山摸不到的对角方向),
            如 chunk↑×conc↑ 联动、MTP深度×conc 联动。任一探针胜出 → 以它为新起点重启下降。
G2 未扫维度点名: 把 §2 表里 champion 没动过的旋钮逐个试一档(MTP topk>1、attention
            backend、mem-fraction↑、A2A 后端…)。每个旋钮一枪, SRVFAIL 也是信息。
G3 远点 ε-探索: 1-2 个与 champion 结构完全不同的形状(不同单元卡数/完全不同并行族),
            专防"起点锁死在错误形状族"。
G4 细网格审计: champion 邻域半步长加密(如 conc ±8、chunk 中点), 确认不是窄峰边缘;
            冠军与亚军差距 <5%(噪声量级)时两者各复测一次再定名次。
```

报告新增 **置信度一节**：① 局部最优证书（每维内点/收敛证据）；② G1-G4 结果；
③ **roofline 上界对比**——按模型 activated FLOPs 和硬件峰值粗算 utilization（标明假设），
utilization 高则全局残余空间被数学压住，低则说明大头在 flag 空间之外（kernel/版本），
诚实声明"全局最优"不可证，只报告已知最优 + 剩余空间清单。

### Phase P — Prefill（判 TTFT_SLO，比 Input TPS/GPU）
1. 种子由 §3.5 从模型属性推导：纯 PP 优先，深度取显存可行的 {最小,中,深} 三档；
   单卡放得下的模型从 TP1/PP1 起（扩展靠副本）。补 1-2 个含 TP 的形状做对照。
2. chunk 爬山：从中档（~2048 或 SLO 推算值）起双向各试一档；甜点随 PP 深度变化，
   高并发下大 chunk 可能反而降 TTFT——用数据说话，不预设方向。
3. 并发用低档（1,2,3,4），找 TTFT 拐点。
4. 剪枝：跨 NUMA 的大 TP 不跑；EP 对 prefill 通常无收益，预算紧时后置。
5. **KV 余量候选**：额外保留一个 TTFT 距 SLO 有充分余量（≥ KV 残差 + 抖动）的
   P 操作点，供保守方案用。

### Phase D — Decode（判 TPOT_SLO，比 Output TPS/GPU）
1. 种子由 §3.5 推导：单元卡数取"权重装下 + KV 余量 ≥30GB/卡"的最小值；
   MoE → EP=TP 起步、补 DPA=2 对照；dense → DPA 是唯一杠杆。
2. MTP 深度扫（模型带 draft head 时）：从 3/1/4 起双向爬（1/1/2 与 4/1/5），
   TPOT 反升（接受率崩）即停；无 head 则 off，NGRAM 作替补。
3. 并发爬梯 32,48,64,96,128…，FAIL 或吞吐平台期（+<5%）即停。
4. EP>1 的胜者补一枪 FlashInfer A2A（需 dp==tp+DPA）。DP-MLP 保持默认 skip，绝不开强同步。
5. ⚠️ colocated（非 disagg）模式下 DPA+MTP+chunked-prefill 联合路径在部分版本触发
   kernel 级崩溃（Triton illegal memory access）；高保真 fake pure-D 模式无此问题。
   colocated 扫该组合前先单点验证。

### 防漏扫机制（每个 Phase 收尾时必做）

先验剪枝可能剪掉真最优，两个兜底机制：

1. **边界检测**：胜者若落在已扫范围的**边界**上（最深 PP、最大 EP、最深 MTP、最高并发档
   PASS 后没有更高档数据），说明真最优可能在界外——沿该维度扩一档再扫，直到胜者是内点。
2. **临界复测**：|指标−SLO| < 5%×SLO 的行（如 TPOT 在 14.25~15.75ms）单发测量不可信，
   复测 3 次取中位数再判 PASS/FAIL；两个候选吞吐差 <5% 时同样复测再排名。

预算内接受不扫、但要在报告"覆盖声明"一节列出的次级维度：MTP topk>1、deepep/megamoe A2A、
`enable_dynamic_chunking`、moe_dp_size 组合、chunk×PP 交互全扫（只做胜者微调）、
mem-fraction/调度参数。剪枝分两级——**数据剪枝**（有历史 FAIL 数据支撑，如 TP8/大 chunk）
可以放心；**推断剪枝**（如"TP1/PP2 按缩放规律 ~880ms 必 FAIL"）要在报告里写明推断依据。

### Phase R — QPS matching 定 PD ratio + 千卡QPS（scripts/analyze.py）
```
python3 analyze.py auto/results.tsv --isl ISL --osl OSL --ttft-slo X --tpot-slo Y
```
- 每个 PASS 配置在其操作点：P 单元 QPS = InputTPS/ISL，D 单元 QPS = OutputTPS/OSL。
- 枚举 xP:yD（x,y≤8）：系统 QPS = min(x·qps_P, y·qps_D)，千卡QPS = 1000×QPS÷(x·gpus_P+y·gpus_D)。
- 输出 top-3 组合及各自的瓶颈侧（P-bound / D-bound）。这是稳态近似，未计 KV 传输开销。
- **报告必须给三个方案**：① 稳态最优；② 小组粒度运维简化解（差距 <5% 时优先推荐部署）；
  ③ **KV 余量保守解**——P 换用 TTFT 余量大的操作点（真 PD 的 KV 传输会给 TTFT 加数十 ms
  量级的残差，需实测；贴线的 P 配置真实部署会超 SLO）。

### Phase V —（可选）真 PD 分离终验
单机 8 卡可容纳如 1P(4卡)+1D(4卡) 验证 KV 传输开销：
- P server 加 `--disaggregation-mode prefill --disaggregation-transfer-backend nixl`（可选 mooncake/mooncake_tcp）+ `--disaggregation-ib-device mlx5_X`；D server 加 `--disaggregation-mode decode ...`。
- 路由：`python -m sglang_router.launch_router --pd-disaggregation --prefill http://127.0.0.1:P1 [bootstrap_port] --decode http://127.0.0.1:P2`，再对 router 端口跑 Poisson 负载（`--request-rate` 取 analyze.py 给的系统 QPS 的 90%）。
- 终验以 **P90** 论：Closed-loop P90 455ms 的配置换 Poisson 后 P90 飙到 824ms 是常态，容量按 P90 规划。

## 5. 报告

- results.tsv → HTML 报告（**白色背景**），文件名带时间戳 `pro5k-sweep-YYYYMMDD-HHMM.html`。
- 头条：最优 P 配置、最优 D 配置、PD ratio、千卡QPS；附完整数据表（PASS/FAIL 标色）、镜像 tag/digest、SLA 参数。
- FAIL 只超标 <3ms 属临界态，Poisson 下 P90 会放大，选型留余量。

## 6. 运维红线（违反必翻车，全部来自本机实测教训)

1. **换配置前杀干净**：`docker exec $CTR pkill -9 -f sglang` 后轮询 `nvidia-smi` 等显存 <2GB 再起新 server（PP 模式 8 个 scheduler 回收慢，残留 = 假 OOM）。
2. **server.log 用容器内路径**（`/sweep/...`），写宿主机路径重定向失败 server 起不来。
3. **不干等**：health 轮询同时扫 server.log 抓 `Traceback|OOM|NCCL error`，命中即判 SRVFAIL；任何等待超 1 分钟先看日志。
4. PP 冷启动 35s+，health 超时给足 600s。
5. 日志/结果文件名带时间戳，输出行带 `[HH:MM:SS]`，防读旧数据。
6. 开跑前 `docker pull $IMG` 并记录 `docker images --digests` 实际版本；换镜像后先跑一个已知配置做 smoke 对齐历史数据。
7. 机器共享：开跑前 `nvidia-smi` 确认无他人任务。他人的自动化 pipeline 可能**链式排队**占机（campaign 结束自动拉起下一个），
   campaign 间的空档≠空闲——必须等其调度/驱动进程全部退出再开跑，抢空档会双向污染数据。
8. `IMG=latest` 是滚动 tag（2026-08 已从 0.5.15.post1 滚到 0.5.16）：开跑时记录
   `python3 -c "import sglang; print(sglang.__version__)"` + digest，跨 campaign 对比先核版本。
9. 经 MCP ssh_exec 起后台任务必须用子 shell 包裹：`( setsid nohup ... & )`，否则通道被子进程
   拖住挂 120s 后报假 failed（任务实际在跑，用独立连接验证）。
10. 等待前序批完成再接续的 watcher 不要用 `pgrep -f <批名>` 轮询——watcher 自己的命令行就含
    该批名，会自匹配（永不退出或行为错乱）。改用 driver.log 里的 BATCH_DONE 标记轮询。
11. 方法论泛化已验证：同一套种子推导+搜索规则在"多卡切分 regime"（大模型 PP/EP）与
    "单卡副本 regime"（小模型 TP1/PP1）都能收敛到正确形状族——换模型只重推种子，不改规则。
