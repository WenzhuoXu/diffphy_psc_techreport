# E242 对 PhyReAct 的设计启发

## 使用边界

这份材料来自《硅谷101》E242 的访谈观点，适合作为 PhyReAct 的
设计动机、工程检查表和未来研究方向，不应被当作方法有效性的实验证据。
如果论文正式讨论自验证、多智能体评审或 reward hacking，仍需补充对应的
同行评审文献。

- 官方节目页：https://sv101.fireside.fm/255
- 用户提供的音频：
  https://media24.fireside.fm/file/fireside-audio-2024/podcasts/audio/f/f0f20376-8faf-4940-b920-84af6c734e2d/episodes/3/308f9278-dd48-44d3-a4ce-ee436968fc28/308f9278-dd48-44d3-a4ce-ee436968fc28.mp3
- 主播：泓君
- 嘉宾：Simon Shaolei Du（杜少雷）、Beibin Li（李辈滨）
- 主题：递归自我提升、自我验证、多 Agent 协作、Harness 与递归漂移

## 节目观点与 PhyReAct 的对应关系

### 1. 角色分离比“让一个大模型多想几遍”更重要

**节目观点（约 16:21）：** 解题 Agent 与验证 Agent 应分开运行，
使用不同的任务指令和上下文；同一问题还可以由多个子 Agent 独立处理，
再由全局 Agent 比较结果。

**对 PhyReAct 的指导：**

- 将系统明确分成 planner、perception specialists、reasoner 和
  deterministic verifier。
- 负责提出检查计划的模型不能同时直接决定最终 verdict。
- 多次调用的价值来自角色隔离和证据覆盖，而不是调用次数本身。
- 不应把同一个 VLM 的多次采样描述成独立验证；共享模型、提示词和
  query framing 会带来高度相关的错误。

这与 `sec:contract` 中的 evidence-only contract，以及 `sec:semantics`
中的 deterministic roll-up 直接对应。

### 2. 冗余应来自异质证据，而不只是多数投票

**节目观点（约 16:21、29:19）：** 多个 Agent 独立处理同一问题，
通常优于单个 Agent 承担全部工作。

**对 PhyReAct 的指导：**

- 优先组合机制不同的证据：检测、计数、轨迹、深度、OCR、音频和
  dense-frame reasoning。
- 多数投票最多用于触发复核，不能替代局部测量。
- 应记录不同证据是否真正独立；共享 backbone 或共享定位结果时，
  不能假设误差相互抵消。
- 冲突证据应保留在 trace 中，并触发 `unknown`、追加测量或人工复核，
  而不是被平均成一个分数。

### 3. 闭环最大的风险是递归漂移

**节目观点（约 14:29）：** 当模型生成自己的训练数据或验证信号时，
推理中的小偏差可能逐轮积累。即使最终答案偶尔正确，错误的推理过程也可能
被下一轮学习。

**对 PhyReAct 的指导：**

- 当前冻结的评价流水线不是 self-evolution；风险主要出现在 critic
  的输出被用于训练 generator、planner 或 critic 自身之后。
- 每条反馈必须保留 claim、measurement plan、模型与工具版本、
  阈值、证据哈希、拒答和汇总规则。
- 每轮更新都必须在人工金标和未参与优化的隐藏数据上重新评估。
- 只看视频级最终分数不足以发现漂移；还应分别监控 coverage、
  false accusation、miss attribution、abstention 和各工具错误。

### 4. 正式评价器与可进化评价器应采用双轨制

**节目观点（约 18:41）：** 训练过程中裁判也可以学习，以减少模型针对
固定裁判进行 reward hacking。

**PhyReAct 不应直接照搬这一做法。** 生成器和裁判同时更新也可能形成
共适应，甚至一起偏离人工目标。更安全的结构是：

1. `reference critic`：冻结、版本化，负责正式报告和回归测试；
2. `shadow critic`：允许更新 planner、routing 或模型，在影子模式中运行；
3. 使用隐藏集、人工标签和对抗性 reward-hacking 案例比较两者；
4. 候选版本只有通过预先定义的门槛后才能晋升；
5. 每次晋升都保留旧版本，并支持回滚。

### 5. `unknown` 和 `abstain` 不能被当作正奖励

未来 refinement loop 中，`plausible` 不能自动解释为“视频完全符合物理
规律”。它只表示已经执行的检查没有发现矛盾。

- `supported`、`unknown` 和 `unchecked` 必须在 reward 中保持不同含义。
- 如果把 unknown 或未检查主张折叠为正奖励，generator 会学习制造
  evaluator 看不清、工具无法覆盖或 planner 不会提问的失败模式。
- claim-level coverage 和 evidence sufficiency 应随 reward 一起记录。
- 任何缺少证据的义务都不应获得与 supported claim 相同的奖励。

这条原则应在 `The clip answer` 和 `sec:loop` 中明确写出。

### 6. 模型和 Harness 必须作为一个整体版本化

**节目观点（约 19:11、23:56、33:46）：** 自我改进可以发生在预训练、
后训练或 Harness；模型与脚手架会共同影响最终能力。

**对 PhyReAct 的指导：**

一个可复现版本必须同时记录：

- planner/reasoner 模型及权重；
- 提示词与 claim taxonomy；
- plan language 与 schema 版本；
- routing 规则；
- perception specialist 版本；
- deterministic interpreter；
- 阈值、容差和预算；
- generator 与 critic snapshot 的对应关系。

实验比较的对象应是完整的 `model × harness` 组合，而不是只报告基础模型名。

### 7. “科学品味”用于决定检查什么，而不是决定真假

**节目观点（约 36:18--51:33）：** 提出有价值的问题和假设需要科学品味；
节目还讨论了避免模型迎合大众偏好，以及用专家数据稳定判断标准。

**对 PhyReAct 的指导：**

- 品味可以用于 claim prioritization、implicit-physics hypothesis generation、
  value-of-information estimation 和追加工具调用。
- 品味不能直接产生最终 verdict。
- 最终真假判断仍应来自局部证据、明确的物理义务和确定性组合规则。
- 物理学家和视觉专家的价值主要体现在整理高价值主张、困难负例、
  失败分类和工具调用优先级，而不是训练一个不透明的整体评分器。

节目中的“AI 宪法”在论文中更适合技术化为：
`versioned evaluation specification`、`fixed evidence contract` 和
`human-calibrated flaw taxonomy`。

## 建议的 PhyReAct 更新协议

未来真正实现 refinement 时，每一轮应采用以下边界：

1. 冻结正式评价器快照 \(E_k\)；
2. 用 claim-level evidence 和三值结果更新 generator \(G_k\)；
3. 得到候选 \(G_{k+1}\)，在隐藏人工标注集上评价；
4. 检查 distribution shift、reward gaming、coverage 和 abstention；
5. planner 或 critic 的改动先形成 shadow candidate \(E'_k\)；
6. \(E'_k\) 通过独立回归测试和人工抽检后，才可晋升为 \(E_{k+1}\)；
7. 保存每个 generator/critic/harness 组合，支持回放与回滚。

## 对当前论文的写作约束

- 可以说 PhyReAct 为 verification-guided refinement 提供评价基础。
- 可以把 closed loop 写成明确的下一阶段实验。
- 不能说当前系统已经实现 recursive self-improvement。
- 不能把静态评价器实验解释成 generator improvement。
- 不能把多模型调用等同于独立互验或正确性保证。
- 不能把 deterministic roll-up 描述为已经解决 evaluator drift。
- 不能把 critic verdict 当作 ground truth；当前 ground truth 仍是人工标签。
- 不能把 `plausible` 描述成完整物理正确性的证书。

## 一句话总结

E242 对 PhyReAct 最重要的启发，不是让评价器自由地“自我进化”，而是把
自我改进限制在一个角色分离、证据可追踪、正式评价通道冻结、能够回归测试
并且可以回滚的 verification harness 中。
