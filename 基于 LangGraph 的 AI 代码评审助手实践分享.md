# 基于 LangGraph 的 AI 代码评审助手实践分享

## 1. 背景：LLM 与 Agent 框架概述

近年来，大规模语言模型（LLM）如 GPT-3/4、ChatGPT、Claude 以及开源的 LLaMA 等不断涌现，它们拥有强大的自然语言理解和生成能力。各大公司纷纷推出自家模型（如 OpenAI 的 GPT 系列、Anthropic 的 Claude、Meta 的 LLaMA 等），并通过微调和指令优化（如 RLHF）提升模型在特定任务中的表现。此外，**多轮对话**和**链式思维（Chain of Thought）**等技术让模型具备更强的推理能力。

在此基础上出现了各种**智能体框架（Agent Framework）**，帮助开发者将 LLM 组合、管理并接入外部工具。2022年推出的 LangChain 采用「链式（Chain）」结构，将多个处理模块线性串联，用于简单的输入-输出任务。其他框架如微软的语义内核（Semantic Kernel）、Meta 的 ReACT 架构等，也各具特色。对于需要**检索外部知识**的应用，还出现了**检索增强生成（Retrieval-Augmented Generation，RAG）**的技术方案。RAG 是一种通过在生成前增加检索步骤，将知识库中的相关信息加入回答过程的模式[learn.microsoft.com](https://learn.microsoft.com/en-us/azure/search/retrieval-augmented-generation-overview#:~:text=Retrieval Augmented Generation ,it's possible to fully constrain)。关键点在于选择合适的检索系统（如向量数据库、全文搜索），对数据进行高效索引和查询，并将检索结果作为上下文输入给 LLM[learn.microsoft.com](https://learn.microsoft.com/en-us/azure/search/retrieval-augmented-generation-overview#:~:text=Retrieval Augmented Generation ,it's possible to fully constrain)。这种模式使得模型可以“阅读”企业文档或代码规范等专业内容，从而生成更准确可靠的回复。

## 2. 为什么需要 LangGraph

随着 LLM 能力提升，实际应用往往不再局限于简单问答，而是复杂的多步骤、多角色、多工具协作任务流程。在构建多智能体系统时，会面临诸如**流程编排**、**状态管理**、**条件控制**等挑战。传统的链式结构虽然直观，但只能线性执行，不适合复杂的多分支、循环或并行场景。因此，LangChain 团队在 2024 年初推出了 LangGraph[blog.langchain.com](https://blog.langchain.com/top-5-langgraph-agents-in-production-2024/#:~:text=We launched LangGraph in early,launch%2C we saw LangGraph become)。LangGraph 的核心理念是用**有向图**来描述任务流程，通过共享的**状态（State）**在节点间传递数据。它将每个子任务封装成图中的“节点（Node）”，通过“边（Edge）”连接起来，节点之间可以并行或条件分支执行，使得流程更加灵活[blog.langchain.com](https://blog.langchain.com/top-5-langgraph-agents-in-production-2024/#:~:text=We launched LangGraph in early,launch%2C we saw LangGraph become)。LangChain 官方指出，LangGraph 是一个低层次、可控的编排框架，将 LangChain 的经验融合其中[blog.langchain.com](https://blog.langchain.com/top-5-langgraph-agents-in-production-2024/#:~:text=We launched LangGraph in early,launch%2C we saw LangGraph become)。换言之，图结构更通用，链可以看作图的一种特殊情形，使得 LangGraph 在复杂场景下具有更强的扩展性和可复用性。

在设计上，LangGraph 依然调用底层的 LLM（如 ChatGPT、Claude 等）以及各种工具，与 LangChain 生态兼容。它的定位是用来构建**长期运行、状态化的智能体系统**[langchain-ai.github.io](https://langchain-ai.github.io/langgraph/#:~:text=Trusted by companies shaping the,running%2C stateful agents)，提供比简单链式调用更细粒度的控制和更清晰的状态管理。对于需要流水线式输出的代码评审任务，LangGraph 正好能将不同检查步骤拆分为并行或有序执行的节点，并在节点间维护并更新全局状态。

## 3. LangGraph 快速入门

要使用 LangGraph，只需通过 pip 安装：

```
pip install langgraph
```

基本流程如下：

- **定义状态（State）**：用 Python 类或 `TypedDict` 定义一个数据结构，用来存放流程中需要共享的数据。例如：

  ```
  class MyState(TypedDict):
      messages: Annotated[list, add_messages]  # 存储对话消息列表
  ```

  这个状态将在流程中传递和更新。

- **定义节点（Node）**：编写函数作为图的节点，接收当前 `state`，执行计算或调用模型/工具，并返回新的部分状态。例如：

  ```
  def agent(state: MyState) -> MyState:
      response = model.invoke(state["messages"])
      return {"messages": [response]}
  ```

- **构建图（Graph）**：先用 `StateGraph` 定义一个图构建器，再添加节点和边：

  ```
  workflow = StateGraph(MyState)
  workflow.add_node("agent", agent)
  workflow.add_edge(START, "agent")
  workflow.add_edge("agent", END)
  graph = workflow.compile()
  ```

- **运行图（Invoke）**：调用 `graph.invoke(initial_state)` 来执行流程，如：

  ```
  state = graph.invoke({"messages": [HumanMessage("你好")]})
  print(state["messages"][-1].content)
  ```

  这样就完成了一个最简单的链式调用。

上述流程与 LangChain 使用方式类似，但 LangGraph 支持更多复杂结构。后续我们会在具体代码评审案例中，详细说明条件分支、并行、记忆等功能的用法。

## 4. LangGraph 核心语法要点

LangGraph 的核心组件包括 **State（状态）**、**Node（节点）**、**Edge（边）**、**StateGraph（图）** 等：

- **State**：表示全局状态数据结构，通常通过 `TypedDict` 定义。所有节点读取和更新的是这个全局状态的副本。比如：

  ```
  class MyState(TypedDict):
      x: int
  ```

- **Node**：图中的执行单元。一个节点是一个函数，接受 `state` 并返回新的 `state`（可以是部分字段）。注意节点运行时拿到的是当前状态的快照，返回值会被合并回全局状态。例如：

  ```
  def increment(state: MyState) -> MyState:
      return {"x": state["x"] + 1}
  ```

- **Edge**：表示节点之间的控制流。无条件边用 `workflow.add_edge(src, dst)` 表示始终执行；**条件边**可根据状态判断选择后续节点，使用 `add_conditional_edges(src, router, {...})`，其中 `router(state)` 返回一个值，匹配字典里键对应的下一节点。

- **Command**：如果节点函数返回一个 `Command` 对象，则可以同时更新状态并指定下一个节点，适用于动态路由。示例：

  ```
  from langgraph.types import Command
  def decision_node(state: MyState) -> Command:
      if state["flag"]:
          return Command(goto="A", update={"x": 100})
      else:
          return Command(goto="B", update={"x": 0})
  ```

  这能让节点在运行后直接跳转到指定的下一步。

- **循环与分支**：可以通过条件边实现循环。比如下面示例：

  ```
  workflow.add_node("increment", increment)
  def is_done(state: MyState) -> bool:
      return state.x > 10
  workflow.add_conditional_edges("increment", is_done, {
      True: "print_state",
      False: "increment"
  })
  ```

  当 `state.x` 未超过 10 时，会不断执行 `increment`；一旦 `is_done` 返回真，转向终止节点。

以上机制让我们能够用图灵般的方式组合推理流程。例如，在代码审查场景中，我们可以根据分析结果选择继续深入检查或结束审查，通过条件边和 Command 实现分支逻辑。

## 5. LangGraph 并行执行机制

LangGraph 支持在图中并发执行多个分支，当一个节点连接了多个后继时，符合条件的分支会同时激活。并行执行中需要考虑**状态合并**：如果多个分支节点同时对状态同一字段进行更新，就需要合并策略（Reducer）。可以通过 `Annotated` 给状态字段指定合并函数，例如：

```
class MyState(TypedDict):
    aggregate: Annotated[list, operator.add]  # 使用加法合并列表
```

这样，如果并行分支返回了各自的列表，最终会将它们累加合并，而不是覆盖。没有指定合并器的字段则按照最后一个写入的值覆盖之前的值。

**示例**：假设有如下工作流：起始状态 `{"aggregate": []}`。节点 A 先执行，状态变成 `["A"]`。然后 A 分为 B 和 C 两个分支并行运行，各自将自身的字符加入 `aggregate`。如果使用 `operator.add` 合并器，最后 D 节点得到的是 `["A", "B", "C"]`（不确定顺序，但都被包含）。如果分支上的节点深度不一致，需要用特殊的多父依赖写法来保证同步。例如：

```
workflow.add_edge("B2", "D")
workflow.add_edge("C", "D")
```

表示 D 等待 B2 和 C 两个父节点都完成后才运行。通过这种方式，我们可以控制并行流程的同步点。

在代码评审助手中，可以并行执行多个子分析任务：如同时运行语法检查、性能分析、安全扫描等，最后再汇总结果。这能大幅加速审查流程。

## 6. AI 代码评审助手设计与实现

### 6.1 需求与方案

代码评审涉及多个方面：**语法/规范检查**、**性能/复杂度评估**、**安全漏洞检测**、**代码风格建议**等，有时还需结合文档和最佳实践。因此，我们可以将这些子任务拆分到不同节点，让它们协同工作。具体方案：用 LangGraph 构建一个多智能体流程，每个节点做一项分析，最后再由聚合节点输出综合报告。我们还可以通过 RAG 检索代码规范文档，或调用静态分析工具辅助判断。

### 6.2 状态设计

先定义一个全局状态结构，记录输入的代码和分析过程中的数据。例如：

```
from typing_extensions import TypedDict
from typing import Annotated, List

class ReviewState(TypedDict):
    code: str                             # 待审查的代码文本
    results: Annotated[List[str], list.append]  # 各分析结论的列表
    report: str                           # 最终生成的评审报告
```

- `code`：由用户提供，需要审查的源代码内容。
- `results`：用于收集各个分析节点的输出，这里用 `list.append` 作为合并器，将多个节点返回的结果依次加入列表。
- `report`：最终的合成评审报告，在最后由聚合节点生成。

### 6.3 节点设计

每个智能体节点负责一种分析任务，输出结果追加到 `results`，最终由聚合节点根据这些内容撰写报告。示例节点包括：

- **lint_node**：代码规范检查。可以让模型扮演检查者，或调用静态分析工具（如 Pylint、ESLint、`flake8` 等），找出语法错误、未使用变量、代码味道等。比如调用外部工具，可以用 LangGraph 的 `ToolNode` 机制封装命令行工具。返回值形如 `{"results": ["[Lint] xxx 问题"]}`。
- **security_node**：安全扫描。让模型检查代码中的潜在安全隐患（如 SQL 注入、权限误用），或调用安全检测库。输出示例：`"[Security] 发现可能的安全漏洞：xxx"`.
- **perf_node**：性能评估。分析算法复杂度或资源使用，指出可能的性能瓶颈。如：`"[Perf] 该算法时间复杂度为 O(n^2)，输入规模大时可能效率低"`。
- **style_node**：风格建议。检查变量命名、注释情况、代码结构规范等。例如 `"[Style] 建议将变量名改为更具描述性的名称..."`。
- **aggregate_node**：整合输出。这个节点负责读取 `results` 中收集到的信息，由 LLM 根据不同视角和优先级，生成结构化的代码评审报告，填写到 `report` 字段。例如，可以提示模型按重要性排序问题，并给出修复建议。

节点函数示例（伪代码）：

```
def lint_node(state: ReviewState) -> ReviewState:
    issues = run_linter_on_code(state["code"])
    return {"results": [f"[Lint] {issue}" for issue in issues]}

def security_node(state: ReviewState) -> ReviewState:
    vulns = scan_security(state["code"])
    return {"results": [f"[Security] {v}" for v in vulns]}

def perf_node(state: ReviewState) -> ReviewState:
    perf = analyze_performance(state["code"])
    return {"results": [f"[Performance] {perf}"]}

def style_node(state: ReviewState) -> ReviewState:
    style_warnings = check_style(state["code"])
    return {"results": [f"[Style] {w}" for w in style_warnings]}

def aggregate_node(state: ReviewState) -> ReviewState:
    # 由 LLM 汇总所有 results 成 report
    prompt = SystemMessage(content="请根据收集到的审查要点，输出结构清晰的评审报告...")
    response = model.invoke([*state["results"], prompt])
    return {"report": response.content}
```

这里 `run_linter_on_code`、`scan_security` 等可视为封装好的工具函数，也可以通过 `ToolNode` 来调用外部程序。使用 `ToolNode` 时，LangGraph 会自动将工具调用结果作为对话消息返回给模型，模型决定如何使用它们。

### 6.4 图结构设计

假设我们希望并行运行 lint、security、perf、style 四个节点，再将它们的输出汇总。可以这样定义流程：

```
from langgraph.graph import StateGraph, START, END

workflow = StateGraph(ReviewState)
workflow.add_node("lint_node", lint_node)
workflow.add_node("security_node", security_node)
workflow.add_node("perf_node", perf_node)
workflow.add_node("style_node", style_node)
workflow.add_node("aggregate_node", aggregate_node)

# 所有检查节点并行地从 START 开始
workflow.add_edge(START, "lint_node")
workflow.add_edge(START, "security_node")
workflow.add_edge(START, "perf_node")
workflow.add_edge(START, "style_node")

# 所有检查节点完成后汇总
workflow.add_edge("lint_node", "aggregate_node")
workflow.add_edge("security_node", "aggregate_node")
workflow.add_edge("perf_node", "aggregate_node")
workflow.add_edge("style_node", "aggregate_node")

workflow.add_edge("aggregate_node", END)
graph = workflow.compile()
```

这里，`aggregate_node` 有四个前驱节点，LangGraph 会等待它们全部执行完后再调用聚合节点。执行时，只需调用：

```
initial_state = {"code": user_submitted_code, "results": [], "report": ""}
final_state = graph.invoke(initial_state)
print(final_state["report"])
```

模型依次并行运行四个分析任务，将各自输出追加到 `results`，最后 `aggregate_node` 基于这些内容生成整体评审意见。最终报告会总结发现的问题、风险点和优化建议，就像人类评审总结的报告一样。

### 6.5 人机交互与记忆（可选）

如需在审查过程中引入人工判断，可使用 `interrupt`/`Command` 机制。例如，当模型输出某关键建议时，可暂停流程等待开发者确认是否采纳，然后继续相应路径。结合 **MemorySaver** 检查点可以保存和恢复流程状态，确保中断点的数据不会丢失。在此案例中，如若多轮审查（例如代码修改后再次评审），记忆机制还能记录历史状态，为持续审查提供上下文。

很多实际的业务场景是十分复杂而不确定的，会超出自动化能力的覆盖范围，例如审批、异常处理等场景，都需要在中间进行人为判断，因此我们需要在工作流中引入**人机交互节点**，让系统在关键环节能够暂停执行，等待人工参与，进而确保流程的准确性。

  在 LangGraph 中，人机交互可以使用**动态中断（interrupt）** 和 **Command** 实现，interrupt 能够将工作流在当前节点暂停，而 Command 则可以让工作流从中断的节点继续执行。此外，由于需要保存暂停前的工作流状态，这里还需要使用到 3.4 节提到的**状态记忆**机制来保存状态。示例如下：

```python
python 体验AI代码助手 代码解读复制代码from typing import Literal, TypedDict
from langgraph.constants import START, END
from langgraph.graph import StateGraph
from langgraph.types import interrupt, Command
from langgraph.checkpoint.memory import MemorySaver

class State(TypedDict):
    llm_output: str
    decision: str

# 模拟 LLM 节点
def llm_node(state: State) -> State:
    return {"llm_output": "This is the generated output."}

# 人机交互节点
def human_node(state: State) -> Command[Literal["approved_path", "rejected_path"]]:
  	# 发生中断，decision 为恢复运行时 Command 中 resume 传入的值
    decision = interrupt({
        "question": "Do you approve the following output?",
        "llm_output": state["llm_output"]
    })
    if decision == "approve":
        return Command(goto="approved_path", update={"decision": "approved"})
    else:
        return Command(goto="rejected_path", update={"decision": "rejected"})

def approved_node(state: State) -> State:
    print("✅ Approved path taken.")
    return state

def rejected_node(state: State) -> State:
    print("❌ Rejected path taken.")
    return state

builder = StateGraph(State)
builder.add_node("llm_node", llm_node)
builder.add_node("human_node", human_node)
builder.add_node("approved_path", approved_node)
builder.add_node("rejected_path", rejected_node)

builder.add_edge(START, "llm_node")
builder.add_edge("llm_node", "human_node")
builder.add_edge("approved_path", END)
builder.add_edge("rejected_path", END)

checkpointer = MemorySaver()
config = {"configurable": {"thread_id": 1}}

graph = builder.compile(checkpointer=checkpointer)

# 初次运行，执行到 human_node 会暂停
result = graph.invoke({}, config=config)
print(result["__interrupt__"])
# 恢复运行，输入 approve，resume 输入的值会传入中断节点，这里也可以修改成 reject 走向另一分支
final_result = graph.invoke(Command(resume="approve"), config=config)
print(final_result)
```

  运行结果：

![img](https://p6-xtjj-sign.byteimg.com/tos-cn-i-73owjymdk6/a0e407f979624f18a99d296b73cfab60~tplv-73owjymdk6-jj-mark-v1:0:0:0:0:5o6Y6YeR5oqA5pyv56S-5Yy6IEAg5bCP55KQ5Lmx5pKe:q75.awebp?rk3s=f64ab15b&x-expires=1756307668&x-signature=WdPTjEVpXwfxYob9j0mHktU5csg%3D)

**中断标志**：当图运行时发生中断，会返回 **interrupt** 键，通过这个键我们能判断出一个图是否发生了中断，这也是后续案例判断是否产生中断的一个关键。

## 7. 可视化监控与调试

LangGraph 提供了可视化监控工具（LangGraph 平台或 LangSmith），用于实时追踪工作流执行。在本地开发时，可以运行 `langgraph dev` 启动调试服务器，借助图形化界面输入代码样本并执行。界面会展示每个节点的执行情况、输入输出状态、并行分支的触发顺序等。这对调试复杂流程尤为有用，可以直观看到各智能体的运行结果和状态变化。例如，当同时触发多个分析节点时，开发者可以检查它们对 `results` 列表做了哪些添加，确认最终聚合结果是否符合预期。

LangGraph 提供了实时可视化监控的方法，搭建流程如下（Python 版本 >= 3.11）：

1. **基于模板创建一个 LangGraph 应用**

```bash
bash 体验AI代码助手 代码解读复制代码# path 为自定义的路径
langgraph new [path] --template new-langgraph-project-python
```

1. **安装依赖**

```bash
bash 体验AI代码助手 代码解读复制代码cd path
pip install -e .
```

1. **创建 `.env` 文件**

  找到 path 目录中的 `.env.example` 文件，并在该目录下新创建一个 `.env` 文件，将 `.env.example `文件的内容复制到里面

1. **开发工作流**

  在 `path/src/agent/graph.py` 中编写程序

1. **启动 LangGraph 服务器**

```bash
bash

 体验AI代码助手
 代码解读
复制代码langgraph dev
```

  控制台中会出现如下提示信息![img](https://p6-xtjj-sign.byteimg.com/tos-cn-i-73owjymdk6/e6254e8e812d43cf92d1a4a141a7292c~tplv-73owjymdk6-jj-mark-v1:0:0:0:0:5o6Y6YeR5oqA5pyv56S-5Yy6IEAg5bCP55KQ5Lmx5pKe:q75.awebp?rk3s=f64ab15b&x-expires=1756307668&x-signature=%2BJs2X43t%2FqyfYgHKtiL9oK2Gh3s%3D)

  进入 Stdio UI 后的链接即可进入监控平台。

  这里以之前编写的多智能体案例为示例，将程序复制进 `graph.py` 中，但是需要将图的运行注释掉，之后可以直接在平台测试。并且，编译图时需要去掉持久化检查点，即状态记忆检查点，原因是持久化由平台自动处理，无需在程序中提供。改动点如下：

```python
python 体验AI代码助手 代码解读复制代码# 编译图
graph = workflow.compile()

# 循环运行图
# question = input("请输入你的问题：")
# inputs = {"input": question}
# result = graph.invoke(inputs, config=config)
# while "__interrupt__" in result:
#     human_input = input(f"下一个运行的节点是{result["direction"]}，是否同意该决策（approve/direct）：")
#     result = graph.invoke(Command(resume=human_input), config=config)
#
# # 输出最终结果
# print(result["result"])
```

  之后，在平台上就可以看到如下图所示的监控：

![img](https://p6-xtjj-sign.byteimg.com/tos-cn-i-73owjymdk6/4a099f15b5eb468b864af48181d1db99~tplv-73owjymdk6-jj-mark-v1:0:0:0:0:5o6Y6YeR5oqA5pyv56S-5Yy6IEAg5bCP55KQ5Lmx5pKe:q75.awebp?rk3s=f64ab15b&x-expires=1756307668&x-signature=ku1CqUM1Xjn4rY%2FAPGCZb%2FFbTKM%3D)

  在 input 中输入问题，提交后就能够看到每个节点的运行流程，也能够直观地看到每个节点的输出，进而能更精细化地监控每个节点。

![img](https://p6-xtjj-sign.byteimg.com/tos-cn-i-73owjymdk6/e373bdfcaca6471486b1c2a2e6061bd8~tplv-73owjymdk6-jj-mark-v1:0:0:0:0:5o6Y6YeR5oqA5pyv56S-5Yy6IEAg5bCP55KQ5Lmx5pKe:q75.awebp?rk3s=f64ab15b&x-expires=1756307668&x-signature=NAeBOeWmCUjcY9aZXJ1fidr1MNM%3D)

![img](https://p6-xtjj-sign.byteimg.com/tos-cn-i-73owjymdk6/c2af39097f7946928ff5f8f30f2345d1~tplv-73owjymdk6-jj-mark-v1:0:0:0:0:5o6Y6YeR5oqA5pyv56S-5Yy6IEAg5bCP55KQ5Lmx5pKe:q75.awebp?rk3s=f64ab15b&x-expires=1756307668&x-signature=irCYMTIMsPjtJhB95amrnAzKuW0%3D)

## 8. 总结与展望

通过本案例可以看到，LangGraph 利用图结构和共享状态，为多步骤的代码审查任务提供了清晰可控的方案。我们将不同类型的分析拆解成独立节点，并行执行后再聚合输出，这种模式既符合审查流程也便于扩展（可以随时新增检查节点）。Future，随着更多开源模型和代码分析工具出现，基于 LangGraph 的代码评审助手还能进一步加强：比如结合代码搜索与知识库检索（RAG）获取最新的最佳实践，或引入自动修复工具自动生成代码修复建议。LangGraph 的状态管理和并行机制使其适用于此类复杂场景，有望推动智能化代码工具的发展，让开发者在日常工作中更高效地获得自动化、结构化的审查反馈。