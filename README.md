# OOP

中国科学院大学《面向对象程序设计》课程作业仓库。

## Assignment 0 — 调研报告：服务化后的软件发展新特征

调研软件形态从**独立化 → 产品化 → 结构化 → 对象化 → 网络化 → 服务化**的演化历程，分析总结"服务化"之后软件发展的新特征（如 SOA / SaaS / PaaS、云计算与 API 经济等）。以 Word 形式提交，篇幅 1000–10000 字，列出重要参考文献。

交付文件：`Assignment0/调研报告_服务化后的软件发展新特征.docx`

## Assignment 1 — UML 类图设计

学习用 UML 类图表示"类"，完成两道设计题：

1. **图书馆借书、还书场景**：设计 `Student`（学生）、`Book`（书籍）、`Librarian`（管理员）三个类，定义属性与方法，并表达借阅关联（1 对 0..*）与封装职责。
2. **三视角下的"汽车"**：分别从厂家（产品）、车主（交通工具）、交通管理机构（监管对象）三个角度设计 3 个"汽车"类，体现"视角决定抽象边界"。

交付文件（`Assignment1/`）：

- `题目1-图书馆借还书类图.png` / `题目2-三视角汽车类图.png` — 类图（PlantUML 导出）
- `assignment1_library.puml` / `assignment2_car.puml` — 类图源文件
- `类设计文档.md` — 详细类设计（属性、方法、关系与分析）

本地重新渲染类图：

```powershell
C:\Users\NO.9\.local\bin\jre21\jdk-21.0.12.1+1-jre\bin\java.exe -jar C:\Users\NO.9\.local\bin\plantuml\plantuml.jar -tpng -charset UTF-8 .\Assignment1\assignment1_library.puml
```

## 课程大作业 — Mandol 融合索引增强

基于 [AgentCombo/Mandol](https://github.com/AgentCombo/Mandol)（面向 LLM Agent 的记忆组织、存储与检索框架），源码位于 `Mandol/`。

Mandol 原生支持 dense / BM25 / SPLADE 三路并行召回与分数融合，但属性、时间、关系约束只能"先检索、后线性过滤"，无法与多路召回协同下推。本项目围绕"语义相似 + 属性约束 + 关系约束"的组合查询，实现了**约束下推式组合检索**：约束先被解析为候选集并按选择率估计，再由调度器决定 pre-filter 下推或 post-filter 兜底。

### 优化内容

| 模块 | 位置 | 说明 |
|---|---|---|
| 约束值对象与集合代数 | `Mandol/src/mandol/constraints/` | `QueryConstraints` 声明式约束；`CandidateSet` 提供求交/并/差与选择率 |
| 约束解析策略族 | 同上 | 元数据、时间范围、关系、空间四类 `ConstraintResolver`，统一 `resolve() -> CandidateSet` 契约 |
| 元数据索引 | `Mandol/src/mandol/indexes/` | `MetadataIndex` 注册路由 + `HashIndex` 等值倒排 / `SortedIndex` 时间二分 + `TimestampNormalizer` 多格式时间归一化 |
| 关系约束 | `Mandol/src/mandol/constraints/resolvers.py` | 直接在 rustworkx 图上做限深 BFS，绕开上游断链 API |
| 关系类型倒排 | `Mandol/src/mandol/core/relation_type_index.py` | `relation_type → 边` 倒排，维护覆盖增边、删边与删点 |
| 自适应调度 | `Mandol/src/mandol/constraints/planner.py` | 按选择率阈值切换 pre/post-filter；可配置回退策略与可审计执行计划 |

设计要点：

- **零侵入衔接**：候选集经现有 `candidate_uids` 参数透传，dense / BM25 / SPLADE 三路检索器零修改。
- **组合优于继承**：索引以组合方式注入 `SemanticMap`，避免加深已有上帝类耦合。
- **默认行为不变**：未注册字段、不可服务条件回退线性扫描；退化候选默认返回空，放宽须显式 opt-in。
- **冷热层一致**：分层换出保留索引条目，重启后按需补建冷层索引。

### 实测结果

离线合成语料（2000 记忆单元 / 2000 边，确定性 dummy encoder），脚本 `Mandol/examples/bench_constrained_retrieval.py`。

约束解析层对比（同一进程内基线）：

| 场景 | 加速比 |
|---|---|
| 元数据等值查询（1/40） | 77.4x |
| 时间区间查询 | 69.9x |
| 关系一跳 BFS | 67.4x |
| 按关系类型取边 | 6.2x |

端到端（cosine 后端，top_k=10）：

| 约束画像 | 仅后置过滤 recall@k | 约束下推 recall@k |
|---|---|---|
| 高选择（平均 6.5 候选） | 16.7% | **100%** |
| 宽约束（平均 580.6 候选） | 51.0% | **100%** |

维护成本（2000 单元）：全量重建 13.14 ms，增量插入 7.6 μs/单元，删除 2.4 μs/单元，常驻内存约 248.5 B/单元。

测试：`cd Mandol && .venv/Scripts/python -m pytest tests/ -q`，当前 **103 passed**。

> 口径说明：以上数据来自合成语料与 dummy encoder，衡量的是执行策略与索引开销，不代表真实语义质量；LoCoMo / LongMemEval 真实数据评测尚未进行。组合查询当前为四类叶子谓词的扁平 AND，OR / NOT 留作扩展点。

设计文档见 `Mandol/个人增加文档/` 下的《优化方向.md》《实现总结.md》《系统分析与设计报告-第二版.md》。
