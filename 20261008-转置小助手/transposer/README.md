# 转置小助手（作业 MVP）

把一封客户入离职邮件（正文 + 附件）一键转置成公司标准报岗表，自动标出需要人工确认的字段。
对应东浩兰生"AI来赛"竞赛项目「转置小助手」（序号 14）的核心需求，全部数据为**合成样本**。

> 技术方案见 `../docs/转置小助手-作业MVP实施方案.md`；需求依据见 `../docs/转置小助手-陪跑需求梳理与规划.md`

## 运行

```bash
cd transposer
C:/Users/xnito/.workbuddy/binaries/python/envs/default/Scripts/python.exe run.py
# 浏览器打开 http://127.0.0.1:8000
```

首次运行前生成样本（仓库内已生成，可跳过）：

```bash
python scripts/gen_samples.py
```

## API key 在哪填

**文件：`transposer/.env`**（本目录下），在等号后填入：

```
TRANSPOSER_API_KEY=sk-xxxxxxxxxxxxxxxx
```

可选两行：`TRANSPOSER_API_BASE`（任何 OpenAI 兼容接口地址）、`TRANSPOSER_MODEL`（模型名）。
填好后运行一次 `python scripts/refresh_cache.py` 生成 LLM 抽取缓存，之后演示走缓存回放，不依赖网络。

**不填 key 也能完整运行**：自动降级为离线规则模式（界面右上角会显示当前抽取来源）。

## 三种抽取模式

| 模式 | 触发 | 说明 |
|---|---|---|
| 实时 LLM | 有 key 且未命中缓存 | 调用 OpenAI 兼容接口 |
| 缓存回放 | 命中 `cache/<样本>.json` | 此前 live 调用的结果，演示默认走这条，100% 稳 |
| 离线规则 | 无 key / 调用失败 | 只跑模板规则抽取，置信度封顶 0.75，界面显式标注 |

抽取策略遵循申报书的"**规则优先、模型补充**"：模板规则先行（确定性），LLM 补规则漏掉的字段；两者不一致时保留规则值并记"模型不一致"冲突，进复核清单。

## 目录结构

```
transposer/
├─ app.py               FastAPI 服务（4 个接口）
├─ run.py               启动入口
├─ .env / .env.example  LLM key 配置
├─ core/
│  ├─ schema.py         标准报岗表字段字典
│  ├─ parse.py          xlsx / docx / pdf / 扫描件解析
│  ├─ rules_extract.py  规则抽取 + 按人聚合 + 冲突检测
│  ├─ llm.py            LLM 调用（OpenAI 兼容）
│  ├─ extract.py        live / cache / offline 三层降级
│  ├─ mapping.py        客户模板加载与字段映射引擎
│  ├─ confidence.py     置信度评分（确定性规则加权）
│  ├─ validate.py       校验规则库（纯函数）
│  ├─ pipeline.py       转置流水线编排
│  └─ export.py         导出带标色的报岗表 xlsx
├─ templates/           3 份客户模板 JSON（换客户只改这里）
├─ samples/             7 份合成邮件样本
├─ cache/               LLM 抽取缓存
├─ static/index.html    单页前端
├─ scripts/             gen_samples / smoke / refresh_cache
└─ tests/               pytest 单元测试（25 项）
```

## 演示脚本（3 分钟）

1. **客户 A · A-01**：xlsx 表格、3 条正常记录 → 一键转置，全绿，导出报岗表
2. **客户 B · B-01**：换一家客户，正文是自然语言叙述、日期格式完全不同 → 仍然出表，证明"换客户只改配置"
3. **客户 A · A-02**：故意埋错（证件号校验位、2 月 30 日、必填缺失、枚举非法）→ 标色 + 复核清单 → 点击问题定位单元格 → 人工修正 → 重新校验全绿 → 导出（修正字段标记"人工确认"）
4. **客户 C · C-02**：社保基数在附件与正文不一致 → "多来源取值不一致"冲突；实习生未满 16 周岁 → 年龄预警
5. **S-01 扫描件**：无文字层 → 降级路径提示，正文线索部分抽取，其余进人工

## 校验规则（全部纯函数 + 单测）

证件号校验位（GB 11643）· 日期可解析 · 枚举合法 · 必填非空 · 基数区间 · 入职缺基数 · 生效日期窗口 · 用工年龄 · 跨源/规则-模型一致性冲突

## 已知边界（作业 MVP 范围外）

- 未接真实邮箱（样本为导出的正文文本 + 附件文件）；RPA、通用抽取平台、多语言、20+ 模板均不做
- 记录聚合按"姓名（其次证件号）"匹配，同名不同人需人工处理
- 扫描件 OCR 留了钩子（`core/parse.py: try_ocr`），默认未接入，走降级路径
- 口径假设（基数区间、日期窗口等）写在 `config/assumptions.json`，真实对接前须由业务方替换

## 测试

```bash
python -m pytest tests/ -q     # 25 项
python scripts/smoke.py        # 7 份样本端到端，核对记录数与预期问题项
```
