# AI 行业简报 Agent

**用一句话要一份「可被证伪」的行业研判简报，不是资讯汇总。**

你说「做一份最近一周的 AI 行业简报」，它会：定档 → 六类事件双语检索 → 证据入库去重 → 判定位移 → 写成简报 → 自检 → 落盘 → 记账本。

课程路径：**A（技术栈较完整的智能体应用）**。

---

## 它解决什么问题

追 AI 新闻的人不缺信息，缺的是「哪些变化值得记，凭什么这么判断，什么情况下我该改主意」。

所以这份简报的每一条主线都带四件东西：**判断、证据链、依赖假设、推翻条件**。读者读完应该更清楚自己哪里不确定，而不是更确信。

为此有三处与常见做法不同：

1. **证据先入库，再判断。** 候选事件进 SQLite 证据库，去重、窗口校验、一手认定都由代码做，判定结果写回库。判断的依据是一张能查的表，不是模型记忆。
2. **交付前跑机械自检。** 字数、条数、术语锚点、簇数一致性、正文链接是否都能在库里找到——12 条规则，有客观答案，不靠模型自觉。
3. **报错了就报错。** 英文检索失败不会拿中文结果顶替，检索没有一手就写「本期无一手信源」。检索缺口会写进账本。

## 技术栈

| 层 | 选型 |
|---|---|
| 模型 | 三个写死的选项：`qwen3.7-plus`、`qwen3.8-max`（阿里百炼）、`deepseek-flash`（DeepSeek），在「设置」页切换。实测见下方「模型选择建议」 |
| Agent 框架 | LangChain **Deep Agents** 0.7.x（内含 LangGraph） |
| 界面 | Streamlit 1.66：顶栏五页（对话 / 证据 / 账本 / 简报 / 设置），侧栏档案与会话 |
| 检索 | DeepSeek 自研搜索 API（主）+ 博查（中文降级） |
| 工具 | 5 个进程内工具 + 1 个 MCP 服务器（5 个工具） |
| 技能 | `ai-brief`（Agent Skills 规范，SKILL.md + 8 份 references） |
| 持久化 | SQLite：检查点 / 会话 / 证据库三个库，**每个档案各一套** |

## 环境要求

- Windows 10/11（命令按 PowerShell 或 Git Bash 写）
- Python **3.12 或 3.13**
- 需要两个密钥：阿里百炼 `DASHSCOPE_API_KEY`、DeepSeek `DEEPSEEK_API_KEY`

## 安装与启动

**虚拟环境不入仓库**（`.venv/` 已忽略），克隆下来先跑一次：

```bash
python scripts/bootstrap.py
```

它会建 `.venv`、按 `requirements.lock.txt` 装依赖、把 `.env.example` 复制成 `.env`。
之后打开 `.env` 填两个必填项（其余可留空）：

```
DASHSCOPE_API_KEY=sk-...
LLM_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
DEEPSEEK_API_KEY=sk-...
```

其他用法：

```bash
python scripts/bootstrap.py --check   # 只体检：Python 版本 / 依赖是否对得上 / 密钥填了没
python scripts/bootstrap.py --loose   # 按 requirements.txt 装（换 Python 大版本或换系统时用）
python scripts/bootstrap.py --index https://mirrors.aliyun.com/pypi/simple/
```

> **为什么不传 `.venv`**：它是 468 MB、25 257 个文件，而 `.venv/pyvenv.cfg` 里写死了
> 本机绝对路径（`home = C:\Users\...\python\versions\3.13.12`），换台机器或换个目录都跑不起来。
> 虚拟环境是生成物，仓库该存的是生成它的配方。
>
> **为什么锁版本**：`streamlit` 的 `width="stretch"`、`expander(on_change="rerun")` 都是较新的
> API；`deepagents` 0.7.24 → 0.7.32 之间 `create_deep_agent` 的可用参数变过。不锁的话，
> 半年后 `pip install -r requirements.txt` 装出来的东西可能起不来，而报错会出现在很难定位的地方。
> `requirements.txt` 是 8 个直接依赖（人读），`requirements.lock.txt` 是完整 96 个包（机器读）。

启动：

```bash
.venv\Scripts\python.exe -m streamlit run app.py
```

浏览器打开 http://localhost:8501 。

命令行快速验证（不开界面，能看见工具调用）：

```bash
.venv\Scripts\python.exe scripts/smoke_chat.py -q "今天几号？算一下 (3.5-2)/3.5*100"
```

## 推荐演示话术

1. `做一份最近一周的 AI 行业简报` （缺省档：学习者 × 标准 × 7 天，约 10 分钟）
2. `按从业者视角，深度梳理最近 3 天 AI 行业的变化` （短窗 + 深度，主线数下调）
3. `给我妈看的，最近一个月 AI 有什么大事，简单说说` （非专业 × 精简）
4. `上次那份里说「开源许可在收紧」，现在有反例吗？` （多轮追问，不重启流程）
5. `把这一期简报里第二条主线再展开讲讲` （读已落盘文件，不再检索）

第 1 条是主路径。第 2 条能同时演示档位推断与短窗规则。第 4 条演示多轮与账本。

## 目录

```
app.py                 Streamlit 界面：顶栏五页（对话/证据/账本/简报/设置），对话页输入框固定在窗口底部；执行过程折叠栏有状态，侧栏可一键全收
agent/
  builder.py           装配：模型 / 工具 / 权限 / 检查点 / 技能发现（按档案）
  runtime.py           Agent 运行时：起事件循环、切档案时关闭重建
  runner.py            一次运行的生命周期：与页面解耦，自己写会话记录
  events.py            流式事件归一化（token / 旁白 / 工具 / 正文）
  prompts.py           系统提示词（行为约束，不写人格）
  subagents.py         critic 审稿代理
core/
  config.py            路径与密钥的唯一入口（所有运行态路径都带档案 id）
  settings.py          模型清单（含写死的接口地址）+ .env 读写与脱敏 + 连通性探针
  archive.py           档案：新建 / 改名 / 删除、当前档案、上限 5 个
  sources.py           信源规则：域名分级、URL 规范化、标题指纹、发布页判定
  store.py             证据库（period / evidence / cluster）
  lint.py              12 条机械校验规则（L1–L12）
  history.py           会话库（给人看的记录）
tools/
  search.py            web_search / fetch_page，双语 + 降级 + 配额
  timebox.py           resolve_window / get_current_time
  calc.py              calculator
mcp/brief_store.py     MCP 服务器：证据库与交付工具
skills/ai-brief/       技能（SKILL.md + references + 报告骨架）
memory/AGENTS.md       全局长期记忆：路径、口径、禁止项
memory/archives/<id>.md 档案补充记忆（读者偏好等），与全局口径分开
scripts/               探针、迁移与验收脚本（见下）；bootstrap.py 建环境
requirements.txt       8 个直接依赖，版本锁定（人读）
requirements.lock.txt  完整依赖树 96 个包（机器读，用于精确复现）
tests/                 73 条离线测试
outputs/briefs/<档案>/ 简报产出
docs/ledger/<档案>/    判断账本（本地，不进 Git）
runtime/archives/<档案>/ 证据库、会话、检查点（不进 Git）
```

## 档案（多套互不干扰的运行态）

侧栏第一行是当前档案。一个档案 = 一套独立的证据库、会话、检查点、账本、补充记忆与简报目录，最多 5 个。

- **新建会话时可以选档案，也可以顺手新建一个档案**——给不同读者（自己 / 同事 / 家人）各留一套，账本不会互相污染。
- 档案之间**互不可见**：Agent 的写权限按档案收窄，它读不到也写不了别的档案。
- **切档案会重建 Agent**（旧的就地关闭）。初始化约 5–8 秒，这是唯一能感到的代价。
- **删档案会删会话、证据、检查点、账本，但保留已生成的简报**——简报是交付物，不是运行态。往期简报页会显示它的档案名（档案已删的显示 id）。

第一次从单档案版本升级上来，跑一次迁移（可重复跑，已迁的会跳过）：

```bash
.venv\Scripts\python.exe scripts/migrate_archives.py
```

## 模型选择建议（实测）

同一句提问、同一窗口、三个模型各跑一期：

| 模型 | 用时 | 结果 |
|---|---|---|
| `deepseek-flash` | 4:32 | 一次交付，正文 1997 字，簇合并最干净（7 簇中 6 个多信源）。**演示首选** |
| `qwen3.7-plus` | 13:02 | 第一遍写完 3784 字却**没调交付工具**，自动补救后通过（1952 字） |
| `qwen3.8-max` | >38 分钟中止 | 判完 52 条证据、标了 32 条入选（配额 6–8），随后卡在一次模型调用上，无产出 |

结论不是"谁更聪明"，是**交付纪律与上下文控制**：flash 档最守规矩，max 档采集过猛把上下文撑大。演示建议用 `deepseek-flash`；需要展示"模型可切换"时切到 `qwen3.7-plus`，正好能演一次自动补救。

两处兜底：写完长报告却没交付会自动追问一次（只一次）；单次运行有 25 分钟墙钟上限，超时会中止并把原因写进会话记录。

## 设置（模型与密钥）

顶栏第五页「设置」。三件事都在这里：

- **选模型**：三个写死的选项。清单外的名字会被拒绝——写错一个模型名，报错会出现在第 40 次工具调用上，那时十分钟已经花掉了。
- **填密钥**：`DASHSCOPE_API_KEY`（Qwen 两个）、`DEEPSEEK_API_KEY`（flash 与检索共用）、`BOCHA_API_KEY`（中文降级）。输入框留空表示不改，已配置的只显示 `sk-…a1b2`，页面不回显完整值。
- **测试连通**：真发一次最小请求。只查"键在不在"查不出"键填错了"——百炼那项被填成 DeepSeek 的 key 时，长度和前缀都像，只有真发请求才 401。

保存后会写回 `.env`（保留注释与其它键）并**重建 Agent**。有任务在跑时这一页整体禁用：重建会关掉正在跑的那个运行时。

## 自检与验收

```bash
# 离线单元测试（不联网）
.venv\Scripts\python.exe -m unittest discover -s tests

# 证据库规则验收：去重、窗口、一手认定
.venv\Scripts\python.exe scripts/probe_store.py

# 四种检索源横评（日期完整率、一手候选率、窗口外条数）
.venv\Scripts\python.exe scripts/probe_search.py

# 测试状态隔离：备份 / 清空 / 恢复（只作用于当前档案，-a 指定别的档案）
.venv\Scripts\python.exe scripts/snapshot.py save before-test
.venv\Scripts\python.exe scripts/snapshot.py clear

# 指定档案跑一轮（不开界面）
.venv\Scripts\python.exe scripts/smoke_chat.py -a default -q "最近 3 天是几号到几号？"

# 从检查点恢复丢失的会话记录（运行中途切页/关浏览器后仍可救）
.venv\Scripts\python.exe scripts/recover_transcript.py --list --all
.venv\Scripts\python.exe scripts/recover_transcript.py --archive <档案> --thread <前缀>
```

**为什么要 `snapshot.py`**：证据库按「期」累积，账本按日期追加。同一个窗口重跑而不清状态，你测的是「上一轮 + 这一轮」。真实续期（换窗口）则必须保留账本，那是回检功能的依据。

## 相对课堂 demo 的改动（浓缩版）

课堂 demo（demo_5）是一个课程助教 Agent：装配骨架、流式事件、双层会话持久化、虚拟文件系统与权限表，这些机制直接继承。业务层全部替换，另外补了 demo 没有的四件事：

**一、检索。** demo 没有联网能力——`fetch` MCP 依赖未安装的 `uvx`，名为 `researcher` 的子代理拿到的是课程数据库查询工具。本项目用 DeepSeek 自研搜索（2026-10-17 上线）做主要检索源，博查做中文降级，并按语言分流：英文查询失败不降级，因为博查的英文结果实测全是中文媒体，降级会让「双语检索」变成假的。检索还带 24 次/期的硬配额，防止模型反复重检。

**二、证据库。** 技能要求「同一信源只算一票、单信源最多贡献 2 簇」，这是跨条目统计，模型在几十条候选里数不过来。所以做了 SQLite 证据库与三个工具，去重键是 `url_norm` + `title_norm`（URL 去重拦不住跨站转载），窗口与一手认定都在入库时由代码判定，一手只认「官方域名 + 发布页 + 真的抓过正文」——不认搜索引擎给的权威度评分，那个分数会把二手大媒体判成一手。

**三、可执行的交付。** `quality.md` 的 24 条自检有客观答案的做成了 12 条规则。更关键的是交付动作绑成了一个工具 `publish_brief`：自检不过就不写文件。这一步是被打出来的——此前模型会写完分析直接在对话框里出报告，既没落盘也没自检，结尾还声称「文件路径：/outputs/briefs/...」。**没有返回值的步骤，模型会以为自己做过了。**

**四、多档案。** 一个档案是一套独立的证据库、会话、账本与简报目录，切档案时整台 Agent 重建。这不是加分项，是测试需求：同一窗口重跑会污染证据库与账本（上一轮的入选标记和簇号还在库里），M5 之前只能靠手动清空目录来隔离。

技能的适配改动集中在「工具接入与落盘协议」，判定规则一条未动（详见 `docs/ai-brief改动记录.md`）。

## 常见问题

| 现象 | 原因与处理 |
|---|---|
| `403 Workspace endpoint access denied`，但 `/models` 能列出 180 个模型 | `.env` 里的 `LLM_BASE_URL` 还留着课程专用 workspace 端点。换成 `https://dashscope.aliyuncs.com/compatible-mode/v1` |
| 克隆下来没有 `.venv`，起不来 | 正常，`.venv` 是生成物、不入仓库。跑 `python scripts/bootstrap.py` 重建 |
| 安装依赖时某个包 403 | 镜像同步问题。`python scripts/bootstrap.py --index https://mirrors.aliyun.com/pypi/simple/` |
| Qwen 两个模型报 401 `Incorrect API key` | `DASHSCOPE_API_KEY` 不是百炼的 key（常见于把 DeepSeek 的 key 填到了这一项）。到「设置」页重新填，再点「测试连通」 |
| 改了密钥/模型但没生效 | 保存后会自动重建 Agent。若是手动改了 `.env`，需要重启 `streamlit run`，或到「设置」页点一次保存 |
| `fixtures/mock.json 还是空的` | `SEARCH_PROVIDER=mock` 但没填快照。改回 `deepseek`，或按文件内说明存一份真实结果 |
| 改了 `agent/` 或技能但界面没变化 | Agent 被 `st.cache_resource` 缓存，Ctrl+C 重启 `streamlit run` |
| 一期跑十几分钟没动静 | 正常。标准档约 10–15 分钟（检索 + 抓取 + 判定 + 自检）。看右栏轨迹与阶段条 |
| 报告没落盘，只在对话里出现 | 交付工具没被调用。检查 `skills/ai-brief/SKILL.md` 的 Step 7；这一步只能靠工具返回值约束，提示词约束不住 |
| 跑完没有简报，正文停在半路 | 看正文末尾的「**运行异常中止：…**」。最常见是步数用尽（递归上限 300 步，约 120 次工具调用）；证据与判定都还在库里，下期可接着用。若连续 12 次交付都没过自检，最后一版会存成 `outputs/briefs/<档案>/_未交付-<时间>.md` |
| 删除档案时提示「仍被占用」 | 该档案的 `checkpoints.db` 还被打开着。界面在删除当前档案前会先关闭运行时；若是脚本里自己建过 `AgentRuntime`，先 `close()`。重启应用后一定成功 |
| 提示「档案已达上限 5 个」 | 删掉不用的档案。若怀疑有幽灵档案：看 `runtime/archives/` 下除了 `default` 还有哪些目录 |
| 切换档案后等了几秒 | 正常。重建 Agent 要重新连接 MCP 工具清单并打开新档案的检查点库 |
| 运行中途切了页面/关了标签，回来只剩提问 | 已修：运行与记录都不再依赖页面。万一遇到（比如进程被杀），跑 `scripts/recover_transcript.py` 从检查点恢复，正文与工具步骤都能取回 |
| 侧栏按钮变灰、输入框不能输入 | 有任务在跑。切档案会关掉当前 Agent 的事件循环、腰斩这次运行，所以运行期间锁住了这些操作 |

## 安全与边界

- `.env` 不进 Git，仓库里只有 `.env.example`。
- Agent 的写权限只到**当前档案**的 `/outputs/briefs/<档案>/**` 与 `/docs/ledger/<档案>/**`；`docs/` 其余部分、`runtime/`、`.env` 均不可读写。
- 政治与监管内容只做事实转述，不做立场评论。
- 简报基于公开信息，重要事实以官方发布为准；不含投资建议。

## 许可与致谢

技能 `ai-brief` 来自课程外的一个独立项目（v0.6.1），本仓库内为应用适配版 `0.7.0-app`。
Agent 骨架参考课堂 demo_5（LangChain Deep Agents 实验）。框架文档：<https://docs.langchain.com/oss/python/deepagents/overview>
