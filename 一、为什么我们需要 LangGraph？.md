## 一、为什么我们需要 LangGraph？

  随着 LLM 能力的不断增强，我们构建 Agent 系统的复杂度也在不断提升，许多场景已经不再是「一个输入 & 一个输出」的简单问答，而是包含**多角色、多阶段、多工具**协作执行的复杂任务流。在这些复杂的场景下，我们会面临着以下的一些挑战：

- 流程编排：多个 Agent 如何串联执行、分工处理、并行加速？
- 状态管理：每个阶段产生的数据如何共享与更新？
- 控制流程：条件判断、异常处理、循环机制如何实现？

  实际上早在 2022 年，一个基于 LLM 的应用开发框架——**LangChain** 就已经诞生，它能够让开发者很方便地将 LLM 接入到现实任务中，正如它的名字，"Chain" 是指“链”，其核心理念是搭建**链式结构**，也就是将多个模块像流水线一样串联起来，构成一个完整的语言任务处理流程。这种结构的优点很明显，就是结构清晰，能够灵活编排各个模块，每个 Chain 都可以自由组合或替换。但其缺点也很明显，这种流程是线性串联的，不太适合于复杂的多分支、多循环、并行执行等场景。为此，**LangGraph** 应运而生。

  LangGraph 是由 LangChain 团队在 2024 年推出的“下一代流程编排框架”，其核心理念是用**图**来描述 LLM 驱动的复杂流程，用状态来驱动智能体系统的协作。了解数据结构的也都知道，图的结构肯定比链更为复杂，而链可以看作一种特殊的图，因此这也就使得 LangGraph 的扩展性更强。本质上 LangGraph 也是使用 LangChain 的底层 API 来接入各类大模型，且 LangGraph 也是完全兼容 LangChain 内置的一系列工具，但由于引入了“**图**”和“**状态**”两个核心概念，它能够更灵活地编排任务，并拥有更清晰的状态管理机制。

![img](https://p6-xtjj-sign.byteimg.com/tos-cn-i-73owjymdk6/bec6748a11e24fd7beb0433378928e9d~tplv-73owjymdk6-jj-mark-v1:0:0:0:0:5o6Y6YeR5oqA5pyv56S-5Yy6IEAg5bCP55KQ5Lmx5pKe:q75.awebp?rk3s=f64ab15b&x-expires=1756307668&x-signature=Y7hT%2FZReG018ZMwhN1Sa2U33g2s%3D)

## 二、LangGraph 快速入门

  环境搭建：

```bash
bash 体验AI代码助手 代码解读复制代码pip install langgraph
pip install langchain-openai
```

  这里先用 LangGraph 创建一个最简单的链式结构，直接调用 LLM 生成回答，如下图所示。

![img](https://p6-xtjj-sign.byteimg.com/tos-cn-i-73owjymdk6/df1d124d890944398dc7c61fc143f692~tplv-73owjymdk6-jj-mark-v1:0:0:0:0:5o6Y6YeR5oqA5pyv56S-5Yy6IEAg5bCP55KQ5Lmx5pKe:q75.awebp?rk3s=f64ab15b&x-expires=1756307668&x-signature=LGLIjASrGAifqap9r2%2BCnO20RY8%3D)

```python
python 体验AI代码助手 代码解读复制代码from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from typing import Annotated
from typing_extensions import TypedDict

# 定义状态
class MyState(TypedDict):
    messages: Annotated[list, add_messages]

# 定义 agent 节点
def agent(state: MyState) -> MyState:
    return {"messages": [model.invoke(state["messages"])]}

# 声明模型
model = ChatOpenAI(
    model_name="DeepSeek-V3",
    openai_api_key="", # 填写 api-key
    openai_api_base="" # 填写 base-url
)

# 声明图
workflow = StateGraph(MyState)
# 在图中添加节点
workflow.add_node("agent", agent)
# 在图中添加边
workflow.add_edge(START, "agent")
workflow.add_edge("agent", END)

# 构建图
graph = workflow.compile()
# 调用图
state = graph.invoke({"messages": [HumanMessage("你好")]})
print(state['messages'][-1].content)
```

## 三、LangGraph 核心语法

### 3.1 StateGraph、State、Node、Edge

  LangGraph 的核心组件包括 **State**（状态）、**Node**（节点）、**Edge**（边）、**StateGraph**（图构建器）。

- **State**：状态，是图在执行过程中传递的数据结构，每个节点会读取和更新该状态，一般为 Python 中的类。

```python
python 体验AI代码助手 代码解读复制代码# 定义状态
class MyState(TypedDict):
    messages: Annotated[list, add_messages]
```

- **Node**：节点，是图中的基本处理单元，用于接收 State 做计算，并返回更新后的状态，一般为 Python 函数。需要注意的是，节点接受的状态是一个**全局状态的快照**，而不是某个节点单独维护的局部状态。

```python
python 体验AI代码助手 代码解读复制代码# 定义 agent 节点
def agent(state: MyState) -> MyState:
    return {"messages": [model.invoke(state["messages"])]}
# 在图中添加节点
workflow.add_node("agent", agent)
```

- **Edge**：边，表示节点之间的流转控制，可以是无条件边，也可以是带条件判定的边，决定了下一个要执行的节点。

```python
python 体验AI代码助手 代码解读复制代码# 在图中添加边
workflow.add_edge(START, "agent")
workflow.add_edge("agent", END)
```

- **StateGraph**：用于构建图，可以添加节点、边等，最终被编译成一个可运行的图对象。

```python
python 体验AI代码助手 代码解读复制代码# 声明图
workflow = StateGraph(MyState)
# 添加节点和边...
# 构建图
graph = workflow.compile()
```

### 3.2 条件分支 & 循环

  在 LangGraph 中，条件分支和循环可以通过带条件判定的边来实现的，需要提供**路由函数**和**路由规则**，执行过程中会根据路由函数返回的值匹配到路由规则中的下一个结点。这里构建下图所示的工作流作为案例。

![img](https://p6-xtjj-sign.byteimg.com/tos-cn-i-73owjymdk6/1df4599f195a405bb452cc69c70285dd~tplv-73owjymdk6-jj-mark-v1:0:0:0:0:5o6Y6YeR5oqA5pyv56S-5Yy6IEAg5bCP55KQ5Lmx5pKe:q75.awebp?rk3s=f64ab15b&x-expires=1756307668&x-signature=%2Biz2hNErVcyHymjZLp04VG%2B0z7Y%3D)

```python
python 体验AI代码助手 代码解读复制代码from pydantic import BaseModel
from langgraph.graph import StateGraph
from langgraph.graph import START, END

# 声明状态
class MyState(BaseModel):
    x: int

# 声明节点
def increment(state: MyState) -> MyState:
    print(f"[increment] 当前 x = {state.x}")
    return MyState(x=state.x + 1)

def print_state(state: MyState) -> MyState:
    print(f"[print_state] 最终 x = {state.x}")

# 判别函数
def is_done(state: MyState) -> bool:
    return state.x > 10

workflow = StateGraph(MyState)
# 添加节点和边
workflow.add_node("increment", increment)
workflow.add_node("print_state", print_state)
workflow.add_edge(START, "increment")
# 带条件的边，第一个参数为起始节点，第二个参数为路由函数，第三个参数为路由规则
workflow.add_conditional_edges("increment", is_done, {
    True: "print_state",
    False: "increment"
})
workflow.add_edge("print_state", END)
```

### 3.3 Command

  如果我们希望节点不仅能更新状态，还能决定下一步要走向哪个节点，就需要使用 **Command** 对象来实现，示例如下：

```python
python 体验AI代码助手 代码解读复制代码from langgraph.types import Command

def my_node(state: MyState) -> Command[Literal["other_node"]]:
    return Command(
        # 更新状态
        update={"foo": "bar"},
        # 控制工作流走向哪个节点
        goto="other_node"
    )
```

  使用 Command 时，函数的返回值表示下一步可能走向的节点，如果存在可能的后继节点，可以配合 if-else 来动态控制，此时就和条件边非常类似了，示例如下：

```python
python 体验AI代码助手 代码解读复制代码def my_node(state: MyState) -> Command[Literal["other_node1", "other_node2"]]:
    if state["foo"] == "bar":
        return Command(
            goto="other_node1",
            update={"foo": "baz"}
        )
    else:
        return Command(
            goto="other_node2",
            update={"foo": "bau"}
        )
```

**什么时候使用命令，什么时候使用条件边**：当需要同时更新图的状态和控制工作动态流路由到其他结点时，推荐使用 Command，如果只是在节点之间有条件地路由，使用条件边则更加合适。

### 3.4 状态记忆

  在构建基于 LangGraph 的智能体流程，我们通常需要保存对话历史（状态），此时需要使用 LangGraph 的 **Memory** 机制，Memory 是 LangGraph 中用于记录状态历史的机制，每当一个节点执行完，LangGraph 会自动将当前状态保存为一个检查点，并**保存到线程中**，不同线程之间是相互隔离的，因此我们需要指定 thread_id。这里以基于内存的检查点记录器 `MemorySaver` 作为示例：

```python
python 体验AI代码助手 代码解读复制代码from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.checkpoint.memory import MemorySaver

# 定义状态
class MyState(TypedDict):
    messages: Annotated[list, add_messages]

# 定义 agent 节点
def agent(state: MyState) -> MyState:
    return {"messages": [model.invoke(state["messages"])]}

# 声明模型
model = ChatOpenAI(
    model_name="DeepSeek-V3",
    openai_api_key="", # 填写 api-key
    openai_api_base="" # 填写 base-url
)

workflow = StateGraph(MyState)
# 添加节点和边
workflow.add_node("agent", agent)
workflow.add_edge(START, "agent")
workflow.add_edge("agent", END)

# 短期记忆管理
memory = MemorySaver()
# 配置 thread_id
thread_config = {"configurable": {"thread_id": "session_10"}}

# 构建图，携带检查点
graph = workflow.compile(checkpointer=memory)
# 调用图，需要携带线程参数
state1 = graph.invoke({"messages": [HumanMessage("你好，我是小璐乱撞")]}, config=thread_config)
print(state1['messages'][-1].content)
state2 = graph.invoke({"messages": [HumanMessage("你还记得我是谁吗")]}, config=thread_config)
print(state2['messages'][-1].content)
```

  运行结果：

![img](https://p6-xtjj-sign.byteimg.com/tos-cn-i-73owjymdk6/79a24a5b069f4ca6a97251ac3134395e~tplv-73owjymdk6-jj-mark-v1:0:0:0:0:5o6Y6YeR5oqA5pyv56S-5Yy6IEAg5bCP55KQ5Lmx5pKe:q75.awebp?rk3s=f64ab15b&x-expires=1756307668&x-signature=HxUz51VRNv5xB8Xo99W0AMs%2Bgb8%3D)

### 3.5 工具调用（MCP）

  LangGraph 原生支持将工具作为图中节点进行封装执行，核心机制是 ToolNode，它是封装了所有工具的节点，LangGraph 支持将多个工具打包给一个 ToolNode，并由 LLM 来自动选择使用哪一个工具。

  这里以调用 MCP Server 工具作为示例，MCP Server 的搭建就不过多赘述，这次使用之前的文章（[从原理到落地：MCP在Spring AI中的工程实践](https://juejin.cn/post/7512902892585992228)）中使用 Spring AI 搭建的 MCP Server，包含获取时间、设置闹钟两个工具。搭建的工作流如下图所示。

![img](https://p6-xtjj-sign.byteimg.com/tos-cn-i-73owjymdk6/f00132c1565b4627b3ffea1bf160ad8d~tplv-73owjymdk6-jj-mark-v1:0:0:0:0:5o6Y6YeR5oqA5pyv56S-5Yy6IEAg5bCP55KQ5Lmx5pKe:q75.awebp?rk3s=f64ab15b&x-expires=1756307668&x-signature=pSAGaZhaLPZqUe2i%2F4i9SZgxdSg%3D)

  包环境：

```bash
bash

 体验AI代码助手
 代码解读
复制代码pip install langchain-mcp-adapters
```

  示例代码：

```python
python 体验AI代码助手 代码解读复制代码from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage
from langchain_mcp_adapters.tools import load_mcp_tools
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from langchain_openai import ChatOpenAI
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.prebuilt import ToolNode
import asyncio

# 定义状态
class MyState(TypedDict):
    messages: Annotated[list, add_messages]

# 定义 agent 结点
def agent(state: MyState) -> MyState:
    return {"messages": [model.invoke(state["messages"])]}

# LLM 是否调用工具
def is_call_tools(state: MyState) -> bool:
    last_message = state["messages"][-1]
    if last_message.tool_calls:
        return True
    return False

# 定义 mcp server 参数
mcp_server_param = StdioServerParameters(
    command="java",
    args=["-jar", "/Users/zhanghaisen/Desktop/develop/code/zhs/mcp/mcp-server/target/mcp-server-0.0.1-SNAPSHOT.jar"],
    env=None
)

# 声明模型
model = ChatOpenAI(
    model_name="DeepSeek-V3",
    openai_api_key="", # 填写 api-key
    openai_api_base="" # 填写 base-url
)

async def main():
    async with stdio_client(mcp_server_param) as (read, write):
        async with ClientSession(read, write) as session:
            # 绑定工具
            await session.initialize()
            tools = await load_mcp_tools(session)
            print("tools:", [tool.name for tool in tools])
            global model
            model = model.bind_tools(tools)

            # 定义图结构
            workflow = StateGraph(MyState)
            workflow.add_node("agent", agent)
            workflow.add_node("tools", ToolNode(tools=tools))
            workflow.add_edge(START, "agent")
            workflow.add_conditional_edges("agent", is_call_tools, {
                True: "tools",
                False: END
            })
            workflow.add_edge("tools", "agent")

            # 构建图
            graph = workflow.compile()
            # 调用图
            state = await graph.ainvoke({"messages": [HumanMessage("设置一个时间为现在的闹钟，并告诉我时间")]})
            print(state['messages'][-1].content)


if __name__ == '__main__':
    asyncio.run(main())
```

  运行结果：

![img](https://p6-xtjj-sign.byteimg.com/tos-cn-i-73owjymdk6/b4aded06816d448bb9cf96eb478130c7~tplv-73owjymdk6-jj-mark-v1:0:0:0:0:5o6Y6YeR5oqA5pyv56S-5Yy6IEAg5bCP55KQ5Lmx5pKe:q75.awebp?rk3s=f64ab15b&x-expires=1756307668&x-signature=3H7myMHFri8Kvb4Y6ImOn7qFXV0%3D)

## 四、LangGraph 并行机制

### 4.1 状态聚合

  在 LangGraph 中，当多个节点并发执行，并尝试更新同一个字段时，LangGraph 需要一个机制来合并这些更新值，而 **Redcuer** 就是 LangGraph 完成合并动作的一种**合并策略**，可以在并行的场景中解决因字段更新冲突而引发的问题。LangGraph 使用 Python 的 **Annotated** 类型注解为每个状态字段指定 Reducer，例如：

```python
python 体验AI代码助手 代码解读复制代码from typing_extensions import TypedDict
from typing import Annotated
from langgraph.graph.message import add_messages

class MyState(TypedDict):
    messages: Annotated[list, add_messages]  # 使用 Reducer 合并消息
    counter: int  # 没有指定 Reducer，默认为覆盖
```

  当存在并发工作的两个节点时，对于 `messages` 字段，此时会调用 `add_messages` 这一 Reducer 来完成合并动作，将两个节点返回的值合并为同一个列表，而不是直接覆盖；对于 `counter` 字段，最终的结果取决于两个节点的实际执行顺序，执行靠后的节点会覆盖掉之前的值。

### 4.2 分支上节点数量相同的并行执行

  在添加边时，当某个节点连接了多个后继节点，LangGraph 会同时激活所有符合条件的分支节点，并将当前状态传递给他们进行并行处理。现在以下图构建的工作流作为示例：

![img](https://p6-xtjj-sign.byteimg.com/tos-cn-i-73owjymdk6/229b964759744610823a788c8f870dbd~tplv-73owjymdk6-jj-mark-v1:0:0:0:0:5o6Y6YeR5oqA5pyv56S-5Yy6IEAg5bCP55KQ5Lmx5pKe:q75.awebp?rk3s=f64ab15b&x-expires=1756307668&x-signature=c9%2FdPA%2B47Bs8bg25uzj9i7gBeeM%3D)

```python
python 体验AI代码助手 代码解读复制代码import operator
from langgraph.graph import StateGraph, START, END
from typing import Annotated
from typing_extensions import TypedDict

# 定义状态
class MyState(TypedDict):
    aggregate: Annotated[list, operator.add]

# 定义结点
class Node:
    def __init__(self, value: str):
        self.value = value

    def __call__(self, state: MyState) -> MyState:
        print(f"add {self.value} to {state['aggregate']}")
        return {"aggregate": [self.value]}

# 声明图
workflow = StateGraph(MyState)
# 在图中添加节点
workflow.add_node("A", Node("A"))
workflow.add_node("B", Node("B"))
workflow.add_node("C", Node("C"))
workflow.add_node("D", Node("D"))
# 在图中添加边
workflow.add_edge(START, "A")
workflow.add_edge("A", "B")
workflow.add_edge("A", "C")
workflow.add_edge("B", "D")
workflow.add_edge("C", "D")
workflow.add_edge("D", END)

# 构建图
graph = workflow.compile()
# 调用图
state = graph.invoke({"aggregate": []})
print(f"final state = {state['aggregate']}")
```

  运行结果：

![img](https://p6-xtjj-sign.byteimg.com/tos-cn-i-73owjymdk6/f9a1573cb8484fe09e897d5ca6b6bd38~tplv-73owjymdk6-jj-mark-v1:0:0:0:0:5o6Y6YeR5oqA5pyv56S-5Yy6IEAg5bCP55KQ5Lmx5pKe:q75.awebp?rk3s=f64ab15b&x-expires=1756307668&x-signature=ib95GDDyAfWdTW5%2F14w8I7%2BQOUQ%3D)

  这里可以看到，B、C 节点运行时，状态都为 ['A']，说明两个节点并行执行了，而 D 节点能接收到 B、C 节点的合并结果的原因是设置了 Reducer 为 `operator.add`，该函数会将每个节点返回的状态都添加到列表中，且无并发安全问题。

### 4.3 分支上节点数量不同的并行执行

  上面的案例中，两条并行分支的节点数量是一致的，但如果数量不一致，聚合节点的执行会出现意料之外的情况。这里构建如下图所示的工作流：

![img](https://p6-xtjj-sign.byteimg.com/tos-cn-i-73owjymdk6/7a62a77d69b0401aac62fc07d4cd6620~tplv-73owjymdk6-jj-mark-v1:0:0:0:0:5o6Y6YeR5oqA5pyv56S-5Yy6IEAg5bCP55KQ5Lmx5pKe:q75.awebp?rk3s=f64ab15b&x-expires=1756307668&x-signature=wf8n0OSMyfYP3FiOBF6X025iLwA%3D)

```python
python 体验AI代码助手 代码解读复制代码import operator
from langgraph.graph import StateGraph, START, END
from typing import Annotated
from typing_extensions import TypedDict

# 定义状态
class MyState(TypedDict):
    aggregate: Annotated[list, operator.add]

# 定义结点
class Node:
    def __init__(self, value: str):
        self.value = value

    def __call__(self, state: MyState) -> MyState:
        print(f"add {self.value} to {state['aggregate']}")
        return {"aggregate": [self.value]}

# 构建图
workflow = StateGraph(MyState)
workflow.add_node("A", Node("A"))
workflow.add_node("B", Node("B"))
workflow.add_node("B2", Node("B2")) # 新增 B2 节点
workflow.add_node("C", Node("C"))
workflow.add_node("D", Node("D"))

workflow.add_edge(START, "A")
workflow.add_edge("A", "B")
workflow.add_edge("B", "B2") # 新增 B-> B2 边
workflow.add_edge("B2", "D") # 新增 B2 -> D 边
workflow.add_edge("A", "C")
workflow.add_edge("C", "D")
workflow.add_edge("D", END)

# 构建图
graph = workflow.compile()
# 调用图
state = graph.invoke({"aggregate": []})
print(f"final state = {state['aggregate']}")
```

  运行结果：

![img](https://p6-xtjj-sign.byteimg.com/tos-cn-i-73owjymdk6/fda6d6fedb664064a5824dd15345483b~tplv-73owjymdk6-jj-mark-v1:0:0:0:0:5o6Y6YeR5oqA5pyv56S-5Yy6IEAg5bCP55KQ5Lmx5pKe:q75.awebp?rk3s=f64ab15b&x-expires=1756307668&x-signature=4%2BOxugmb%2FcNmJJt1Ic%2Fmm%2FEQnGo%3D)

  这里可以看到，B、C 节点确实都并行执行了，但是 **D 节点最终运行了两次**，这里以回合制执行节点的方式来分析原因：

```bash
bash 体验AI代码助手 代码解读复制代码第一回合：A
第二回合：B、C
第三回合：B2、D
第四回合：D
```

  归根结底是 A -> C -> D 这条分支运行到 D 节点后，没有等待另一条分支运行到 D 节点，就直接执行了 D 节点。如果我们想要 D 节点只执行一次，就需要加强 B2、C、D 三个节点之间的关系，修改边的定义方式，具体做法如下：

```python
python 体验AI代码助手 代码解读复制代码workflow.add_edge(START, "A")
workflow.add_edge("A", "B")
workflow.add_edge("B", "B2")
workflow.add_edge("A", "C")
# 起始节点修改为列表
workflow.add_edge(["B2", "C"], "D")
workflow.add_edge("D", END)
```

  运行结果：

![img](https://p6-xtjj-sign.byteimg.com/tos-cn-i-73owjymdk6/3457a6fd3a3345b3b4a3e4e042b9f43c~tplv-73owjymdk6-jj-mark-v1:0:0:0:0:5o6Y6YeR5oqA5pyv56S-5Yy6IEAg5bCP55KQ5Lmx5pKe:q75.awebp?rk3s=f64ab15b&x-expires=1756307668&x-signature=zdiBlY2IRtZot9s7bhU5wjvldOU%3D)

  可以看到，D 节点只运行了一次，因为这里增强了 B2、C、D 之间的依赖关系，要求 D 节点需要等待 B2、C 节点同时运行完才能继续执行，而之前的定义方式只能定义两条独立的边，没有强调依赖关系。

### 4.4 并行执行 & 条件分支

  如果需要执行的并行分支不确定，可以使用**带条件判定的边来实现**，例如，对于下图所示的工作流，可以使用条件分支来选择并行执行 B、C 节点还是 C、D 节点。

![img](https://p6-xtjj-sign.byteimg.com/tos-cn-i-73owjymdk6/c9909b0b42004a158a74dfc4c9cb6561~tplv-73owjymdk6-jj-mark-v1:0:0:0:0:5o6Y6YeR5oqA5pyv56S-5Yy6IEAg5bCP55KQ5Lmx5pKe:q75.awebp?rk3s=f64ab15b&x-expires=1756307668&x-signature=UYF44FnFjfgwcSsB4lYlaiZuh3E%3D)

```python
python 体验AI代码助手 代码解读复制代码import operator
from langgraph.graph import StateGraph, START, END
from typing import Annotated
from typing_extensions import TypedDict, Sequence

# 定义状态
class MyState(TypedDict):
    aggregate: Annotated[list, operator.add]
    switch: str # 开关，"BC" / "CD"

# 定义结点
class Node:
    def __init__(self, value: str):
        self.value = value

    def __call__(self, state: MyState) -> MyState:
        print(f"add {self.value} to {state['aggregate']}")
        return {"aggregate": [self.value]}

# 路由函数
def router(state: MyState) -> Sequence[str]:
    if state["switch"] == "CD":
        return ["C", "D"]
    return ["B", "C"]

# 定义图结构
workflow = StateGraph(MyState)
workflow.add_node("A", Node("A"))
workflow.add_node("B", Node("B"))
workflow.add_node("C", Node("C"))
workflow.add_node("D", Node("D"))
workflow.add_node("E", Node("E"))
workflow.add_edge(START, "A")
# 条件边
workflow.add_conditional_edges("A", router, ["B", "C", "D"])
workflow.add_edge("E", END)

# 构建图
graph = workflow.compile()
# 调用图
state = graph.invoke({"aggregate": [], "switch": "BC"})
print(f"final state = {state['aggregate']}")
```

  运行结果：

![img](https://p6-xtjj-sign.byteimg.com/tos-cn-i-73owjymdk6/f7a3e87686884b7c932463cc169deb20~tplv-73owjymdk6-jj-mark-v1:0:0:0:0:5o6Y6YeR5oqA5pyv56S-5Yy6IEAg5bCP55KQ5Lmx5pKe:q75.awebp?rk3s=f64ab15b&x-expires=1756307668&x-signature=PAl%2FQGf2kL0kFF5ZNxsGVGKLvM8%3D)

  由于输入的 `switch` 是 "BC"，所以最终路由到了 B、C 节点。需要注意的是，对于 `add_conditional_edges` 方法的第三个参数，可以使用 3.2 节示例中的字典（dict）形式，也可以使用这里的**列表（list）形式**，如果是列表，表示直接列出所有可能的后继节点列表。这种方式的编写会**更适合动态分支**，因为不需要提前知道有哪些分支的组合，而是由路由函数来动态选择。

## 五、多智能体编排案例

### 5.1 人机交互（Human-in-the-loop）

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

### 5.2 案例设计方案

  这里设计如下图所示的工作流：

![img](https://p6-xtjj-sign.byteimg.com/tos-cn-i-73owjymdk6/17029f8a312d440d9e15bfa9a43c47be~tplv-73owjymdk6-jj-mark-v1:0:0:0:0:5o6Y6YeR5oqA5pyv56S-5Yy6IEAg5bCP55KQ5Lmx5pKe:q75.awebp?rk3s=f64ab15b&x-expires=1756307668&x-signature=qkQl8Gq2ZJ0N2nlx6Xvt%2FSPMQIE%3D)

- **supervisor_node**：管理者节点，负责调度智能体的执行顺序和整体流程控制，本质上也是个智能体。
- **human_node**：人机交互节点，用户在此节点上决定是否接受管理者的调度建议。
- **agent1_node**：技术专家智能体，从技术实现角度分析问题。
- **agent2_node**：产品专家智能体，从用户需求和产品设计角度分析问题。
- **agent3_node**：市场专家智能体，从市场趋势与商业价值角度分析问题。
- **aggregate_node**：聚合节点，用于整合已有信息并输出最终答案，本质上也是个智能体。

流程说明：

  用户提出问题后，`supervisor_node` 首先会并行调度 `agent1_node`、`agent2_node`、`agent3_node`，分别从不同专业角度生成初步分析结果。三个智能体执行完成后，控制权回到 `supervisor_node`，由其选择下一步的处理方向。每一次从 `supervisor_node` 发起的跳转都必须经过 `human_node` 审核，即用户需人工确认是否接受该调度。若用户同意，系统将进入指定智能体节点继续处理，并再次返回 `supervisor_node`，进入下一轮调度；若用户不同意，或 `supervisor_node` 判断当前信息已经足够，无需进一步吹了，则流程转入 `aggregate_node`，整合已有内容并生成最终输出。

### 5.3 案例实现

#### 5.3.1 状态设计

```python
python 体验AI代码助手 代码解读复制代码class GraphState(TypedDict):
    input: str # 用户输入
    messages: Annotated[List, add_messages] # llm 历史交互消息
    direction: List[str] # 节点走向
    result: str # 最终结果
```

#### 5.3.2 节点设计

```python
python 体验AI代码助手 代码解读复制代码# 人机交互节点，决定是否接受 supervisor 的调度
def human_node(state: GraphState) -> Command[Literal["agent1_node", "agent2_node", "agent3_node", "aggregate_node"]]:
    decision = interrupt({
        "question": "你同意后续的操作吗？",
        "llm_output": state["messages"],
        "direction": state["direction"]
    })
    if decision == "approve":
        return Command(goto=state["direction"])
    else:
        return Command(goto="aggregate_node")


# agent1 负责从技术实现角度分析问题
def agent1_node(state: GraphState) -> GraphState:
    print(f"agent1 run")
    response = model.invoke([
        *state["messages"],
        SystemMessage(content="你是一位资深技术专家，擅长分析和解决复杂的工程与系统设计问题。"
                              + "请根据用户提出的问题，从技术角度进行分析或提供解决方案，重点关注可行性、"
                              + "实现方式、技术选型与潜在的技术挑战。请避免讨论产品设计、用户体验或市场相关内容。")
    ])
    return {"messages": [response]}


# agent2 负责从用户需求和产品设计角度分析问题
def agent2_node(state: GraphState) -> GraphState:
    print(f"agent2 run")
    response = model.invoke([
        *state["messages"],
        SystemMessage(content="你是一位经验丰富的产品经理，负责洞察用户需求并设计有价值的产品方案。"
                              + "请根据用户提出的问题，从产品角度进行分析，关注用户体验、功能设计、易用性以及是否解决用户痛点。"
                              + "请避免涉及技术实现细节或市场分析。")
    ])
    return {"messages": [response]}

# agen3 负责从市场趋势与商业价值角度分析问题
def agent3_node(state: GraphState) -> GraphState:
    print(f"agent3 run")
    response = model.invoke([
        *state["messages"],
        SystemMessage(content="你是一位市场与商业策略专家，擅长从市场趋势、用户需求和商业价值的角度分析问题。"
                              + "请根据用户提出的问题，从市场和业务的角度进行分析，重点关注商业机会、"
                              + "目标用户、市场规模与竞争情况。请避免讨论技术可行性或具体产品功能。")
    ])
    return {"messages": [response]}


# supervisor 负责决策下一个步骤：继续 agent 分析或结束流程
def supervisor_node(state: GraphState) -> GraphState:
    print(f"supervisor run")
    # 首次运行，返回并行执行决策
    if len(state["messages"]) == 0:
        return {"messages": [HumanMessage(state["input"])], "direction": ["agent1_node", "agent2_node", "agent3_node"]}
    response = model.invoke([
        *state["messages"],
        SystemMessage(content="你是一位智能协作流程的管理者，负责统筹多个专家智能体的输出，并根据当前对话内容做出下一步决策。"
                              + "你需要综合 agent1（技术智能体）、agent2（产品智能体）、agent3（市场智能体）提供的信息，"
                              + "判断是否需要继续深入某个角度，或是结束分析并汇总已有结论。你的任务是明确指出接下来应该由哪个智能体继续处理，或是否可以直接结束流程进行汇总。"
                              + "如果继续处理，返回：'agent1_node' 或 'agent2_node' 或 'agent3_node'；"
                              + "如果结束，返回：'aggregate_node'。")
    ])
    text = response.content
    return {"direction": [text]}

# aggregate 符合整合信息并输出最终答案
def aggregate_node(state: GraphState) -> GraphState:
    print(f"aggregate run")
    response = model.invoke([
        *state["messages"],
        SystemMessage(content="你是一位综合分析专家，负责汇总来自技术、产品和市场三个智能体的回答内容，"
                              + "并为用户生成一份结构清晰、重点突出的综合结论。请合理整合不同视角的信息，"
                              + "避免重复，突出各角度的要点，并以通俗易懂的方式呈现给用户。"
                              + "最终输出应包括简洁的总结与各角度的关键信息，避免加入你自己的推测或意见。")
    ])
    return {"result": response.content}
```

#### 5.3.4 显式边设计

```python
python 体验AI代码助手 代码解读复制代码workflow.add_edge(START, "supervisor_node")
workflow.add_edge("supervisor_node", "human_node")
workflow.add_edge("agent1_node", "supervisor_node")
workflow.add_edge("agent2_node", "supervisor_node")
workflow.add_edge("agent3_node", "supervisor_node")
workflow.add_edge("aggregate_node", END)
```

#### 5.3.4 完整代码

```python
python 体验AI代码助手 代码解读复制代码from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END, START
from langgraph.types import Command, interrupt
from typing import Annotated, Literal, TypedDict, List
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver

# 初始化模型
model = ChatOpenAI(
    model_name="DeepSeek-V3",
    openai_api_key="", # 填写 api-key
    openai_api_base="", # 填写 base-url
    temperature=0.7
)

# 状态
class GraphState(TypedDict):
    input: str # 用户输入
    messages: Annotated[List, add_messages] # llm 历史交互消息
    direction: List[str] # 节点走向
    result: str # 最终结果

# 人机交互节点，决定是否接受 supervisor 的调度
def human_node(state: GraphState) -> Command[Literal["agent1_node", "agent2_node", "agent3_node", "aggregate_node"]]:
    decision = interrupt({
        "question": "你同意后续的操作吗？",
        "llm_output": state["messages"],
        "direction": state["direction"]
    })
    if decision == "approve":
        return Command(goto=state["direction"])
    else:
        return Command(goto="aggregate_node")


# agent1 负责从技术实现角度分析问题
def agent1_node(state: GraphState) -> GraphState:
    print(f"agent1 run")
    response = model.invoke([
        *state["messages"],
        SystemMessage(content="你是一位资深技术专家，擅长分析和解决复杂的工程与系统设计问题。"
                              + "请根据用户提出的问题，从技术角度进行分析或提供解决方案，重点关注可行性、"
                              + "实现方式、技术选型与潜在的技术挑战。请避免讨论产品设计、用户体验或市场相关内容。")
    ])
    return {"messages": [response]}


# agent2 负责从用户需求和产品设计角度分析问题
def agent2_node(state: GraphState) -> GraphState:
    print(f"agent2 run")
    response = model.invoke([
        *state["messages"],
        SystemMessage(content="你是一位经验丰富的产品经理，负责洞察用户需求并设计有价值的产品方案。"
                              + "请根据用户提出的问题，从产品角度进行分析，关注用户体验、功能设计、易用性以及是否解决用户痛点。"
                              + "请避免涉及技术实现细节或市场分析。")
    ])
    return {"messages": [response]}

# agen3 负责从市场趋势与商业价值角度分析问题
def agent3_node(state: GraphState) -> GraphState:
    print(f"agent3 run")
    response = model.invoke([
        *state["messages"],
        SystemMessage(content="你是一位市场与商业策略专家，擅长从市场趋势、用户需求和商业价值的角度分析问题。"
                              + "请根据用户提出的问题，从市场和业务的角度进行分析，重点关注商业机会、"
                              + "目标用户、市场规模与竞争情况。请避免讨论技术可行性或具体产品功能。")
    ])
    return {"messages": [response]}


# supervisor 负责决策下一个步骤：继续 agent 分析或结束流程
def supervisor_node(state: GraphState) -> GraphState:
    print(f"supervisor run")
    # 首次运行，返回并行执行决策
    if len(state["messages"]) == 0:
        return {"messages": [HumanMessage(state["input"])], "direction": ["agent1_node", "agent2_node", "agent3_node"]}
    response = model.invoke([
        *state["messages"],
        SystemMessage(content="你是一位智能协作流程的管理者，负责统筹多个专家智能体的输出，并根据当前对话内容做出下一步决策。"
                              + "你需要综合 agent1（技术智能体）、agent2（产品智能体）、agent3（市场智能体）提供的信息，"
                              + "判断是否需要继续深入某个角度，或是结束分析并汇总已有结论。你的任务是明确指出接下来应该由哪个智能体继续处理，或是否可以直接结束流程进行汇总。"
                              + "如果继续处理，返回：'agent1_node' 或 'agent2_node' 或 'agent3_node'；"
                              + "如果结束，返回：'aggregate_node'。")
    ])
    text = response.content
    return {"direction": [text]}

# aggregate 符合整合信息并输出最终答案
def aggregate_node(state: GraphState) -> GraphState:
    print(f"aggregate run")
    response = model.invoke([
        *state["messages"],
        SystemMessage(content="你是一位综合分析专家，负责汇总来自技术、产品和市场三个智能体的回答内容，"
                              + "并为用户生成一份结构清晰、重点突出的综合结论。请合理整合不同视角的信息，"
                              + "避免重复，突出各角度的要点，并以通俗易懂的方式呈现给用户。"
                              + "最终输出应包括简洁的总结与各角度的关键信息，避免加入你自己的推测或意见。")
    ])
    return {"result": response.content}

# 构建图
workflow = StateGraph(GraphState)

# 添加节点
workflow.add_node("human_node", human_node)
workflow.add_node("agent1_node", agent1_node)
workflow.add_node("agent2_node", agent2_node)
workflow.add_node("agent3_node", agent3_node)
workflow.add_node("supervisor_node", supervisor_node)
workflow.add_node("aggregate_node", aggregate_node)

# 添加边
workflow.add_edge(START, "supervisor_node")
workflow.add_edge("supervisor_node", "human_node")
workflow.add_edge("agent1_node", "supervisor_node")
workflow.add_edge("agent2_node", "supervisor_node")
workflow.add_edge("agent3_node", "supervisor_node")
workflow.add_edge("aggregate_node", END)

# 记忆管理
checkpointer = MemorySaver()
config = {"configurable": {"thread_id": 1}}

# 编译图
graph = workflow.compile(checkpointer=checkpointer)

# 循环运行图
question = input("请输入你的问题：")
inputs = {"input": question}
result = graph.invoke(inputs, config=config)
while "__interrupt__" in result:
    human_input = input(f"下一个运行的节点是{result["direction"]}，是否同意该决策（approve/direct）：")
    result = graph.invoke(Command(resume=human_input), config=config)

# 输出最终结果
print(result["result"])
```

### 5.4 测试结果

```markdown
markdown 体验AI代码助手 代码解读复制代码请输入你的问题：我们正在考虑开发一个面向高校学生的AI学习助手，帮我分析一下

supervisor run

下一个运行的节点是['agent1_node', 'agent2_node', 'agent3_node']，是否同意该决策（approve/direct）：approve

agent2 run
agent1 run
agent3 run

supervisor run

下一个运行的节点是['aggregate_node']，是否同意该决策（approve/direct）：approve

aggregate run

### 综合结论：高校AI学习助手可行性分析  

#### **核心结论**  
开发面向高校学生的AI学习助手具备明确市场需求和技术可行性，但需在**教育属性**和**商业价值**间找到平衡。关键在于：  
1. **解决高频刚需**（如笔记整理、即时答疑），而非泛功能覆盖。  
2. **差异化设计**：强化学科垂直能力（如数学推导、代码Debug），与通用AI工具形成区隔。  
3. **渐进式落地**：从单一学科/功能试点，逐步扩展至全场景。  

---

### **各角度关键信息整合**  

#### **1. 技术实现要点**  
- **架构设计**：微服务架构（Spring Cloud/Django）+ Kubernetes，便于功能模块化扩展。  
- **核心AI能力**：  
  - **NLP**：基于BERT/GPT优化教育场景语义理解（如数学公式混合文本）。  
  - **知识图谱**：构建学科本体（Neo4j），支持逻辑推理类问题解答。  
- **关键挑战**：  
  - 多模态处理（教科书图表、手写笔记识别）。  
  - 回答准确性验证（需引入教师审核机制）。  

#### **2. 产品设计重点**  
- **核心功能优先级**：  
  - **第一梯队**：智能笔记（自动生成知识图谱）、24小时问答（限学科边界）。  
  - **第二梯队**：个性化学习路径、作业思路引导（避免直接给答案）。  
- **用户体验关键**：  
  - **自然交互**：支持语音/图片提问，模拟“请教学长”体验。  
  - **防沉迷设计**：例如每日答疑次数限制，鼓励自主思考。  

#### **3. 市场与商业化策略**  
- **目标用户**：  
  - **主攻本科生**（通识课需求集中），逐步渗透研究生（科研辅助）。  
  - **拓展留学生**：语言润色、作业翻译是明确付费点。  
- **变现路径**：  
  - **基础功能免费**（引流），高级功能订阅（如考试预测、一对一答疑）。  
  - **B2B合作**：与学校合作嵌入教学系统，按年收费。  
- **竞争壁垒**：  
  - **数据积累**：独家接入教材、历年考题库，提升回答精准度。  
  - **场景绑定**：与课程表同步、作业截止日提醒等高频场景深度结合。  

#### **4. 主要风险与应对**  
- **学术诚信争议**：  
  - 通过功能设计规避（如禁止直接生成论文）。  
  - 主动与高校合作，定位为“教学辅助工具”。  
- **需求周期性**：  
  - 扩展职业规划、技能培训等非周期模块，提升用户留存。  

---

### **下一步行动建议**  
1. **MVP开发**：选择计算机或数学学科，优先实现“代码Debug+解题步骤引导”最小闭环。  
2. **校园试点**：通过3-5所高校的种子用户测试，收集真实反馈。  
3. **资源合作**：联系教材出版商（如高等教育出版社）获取内容授权。  

（注：以上分析仅基于提供的多视角信息整合，无额外推测。）
```

## 六、LangGraph 实时可视化监控

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

## 七、展望

  作为一个多智能体编排框架，LangGraph 正在重塑我们构建和组织多智能体系统的方式，它以图结构直观表达工作流，使智能体之间的协作更清晰、更可控。未来，随着大模型能力的不断增强，LangGraph 有望在多模态交互、动态调度、流程可视化等方面持续进化，推动智能系统从线性调用走向高度并行、可解释、可插拔的图式编排范式，重塑智能应用的开发范式与运行架构。

作者：小璐乱撞
链接：https://juejin.cn/post/7530094533712887827
来源：稀土掘金
著作权归作者所有。商业转载请联系作者获得授权，非商业转载请注明出处。