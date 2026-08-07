# pro5k-sgl-sweep 优化路径示意图

给定任意 SLA + 负载 + 模型，skill 的完整寻优 pipeline（SKILL.md 是权威定义，本图为导览）：

```mermaid
flowchart TB
    A["输入: SLA (TTFT/TPOT) + ISL/OSL + 模型"] --> PLAN["§0.5 Sweep Plan<br/>环境快照/候选表/预算/决策点<br/>用户确认后开跑"]
    PLAN --> SEED["§3.5 种子推导 (零数值先验)<br/>显存可行域: 最小单元=ceil(W/60GB), KV余量≥30GB/卡<br/>P: 纯PP {浅,中,深} / 单卡模型走副本<br/>D: 最小可行单元, MoE→EP=TP+DPA对照, MTP从3/1/4"]

    SEED --> SCREEN["初筛 (fake pure-D 短爬梯)<br/>淘汰 吞吐/卡 < 冠军60%, 留 top-K"]

    SCREEN --> CD["坐标下降 (LLM在环, 每点读 results.tsv)<br/>P: pp深度→chunk→并发<br/>D: mtp深度→dpa→ep→mem-fraction→并发"]
    CD --> CDRULE{"单维 ±1 档"}
    CDRULE -->|"改进且PASS"| CD2["同方向继续"] --> CDRULE
    CDRULE -->|"退步"| CD3["换下一维"] --> CDRULE
    CDRULE -->|"全维度无改进"| LOCAL["单维局部最优"]

    CD -.停止条件.-> STOPS["① SLA FAIL 即停爬<br/>② 吞吐平台期 (+<5%) 即停<br/>③ 边界扩展: 冠军在扫描边界→扩一档<br/>④ 连续改进但单步 <5% → 收敛"]

    LOCAL --> FINAL["决赛 (高保真)<br/>D 用官方 pure-D: fake KV后端+FAKE_BOOTSTRAP_HOST<br/>真实 ISL 上下文/足额 KV 分配<br/>临界点 (|指标-SLO|<5%) 3次取中位+邻档一起测"]

    FINAL --> G["G1-G4 全局性加固 (预算≈主搜索50%)<br/>G1 交互探针: 两维对角联动<br/>G2 未扫旋钮点名: 每个一枪, SRVFAIL也是信息<br/>G3 远点ε-探索: 不同形状族<br/>G4 细网格审计: 半步长+噪声runoff"]
    G -->|"探针胜出 > 噪声(5%)"| CD
    G -->|"全部不胜"| CERT["局部最优证书 (G加固版)"]

    CERT --> MATCH["QPS Matching<br/>qps_P=inpTPS/ISL, qps_D=outTPS/OSL<br/>枚举 xP:yD, 系统QPS=min(x·qps_P, y·qps_D)<br/>千卡QPS = 1000×QPS/总卡数"]

    MATCH --> REPORT["报告 (白底HTML, 时间戳)<br/>三方案: 稳态最优/小组运维解/KV余量保守解<br/>置信度: 证书+G结果+roofline上界<br/>覆盖声明: 数据剪枝 vs 推断剪枝 vs 未扫维度"]

    style SEED fill:#ede9fe
    style CD fill:#dbeafe
    style G fill:#fef3c7
    style MATCH fill:#d1fae5
    style REPORT fill:#f3f4f6
    style STOPS fill:#fee2e2
```

## 测量方法速览

```mermaid
flowchart LR
    subgraph P["P 单压 (天然高保真)"]
      P1["isl=ISL, osl=1 纯prefill<br/>+ --disable-radix-cache<br/>判 P50 TTFT, 计 Input TPS"]
    end
    subgraph D["D 单压 (一律高保真 fake pure-D)"]
      D2["--disaggregation-transfer-backend fake<br/>+ bootstrap_host=2.2.2.2<br/>零prefill, 每请求足额 ISL KV<br/>初筛短爬梯 / 决赛完整爬梯<br/>判 mean TPOT, 计 Output TPS"]
    end
```

两轮实测验证：多卡切分 regime（大 MoE，PP/EP 形状族）与单卡副本 regime（小 MoE，TP1/PP1 形状族）均由同一套规则收敛到正确形状族；G 阶段曾抓到坐标下降遗漏的旋钮改进。
