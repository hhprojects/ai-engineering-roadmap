# Lesson 13 — Prompt Programming Frameworks

> **The single sentence version:** Instead of writing and maintaining prompts by hand, you can describe what you want in structured Python code and let a framework — DSPy, Instructor, BAML, LangChain, LlamaIndex — generate and optimize the prompts for you; this is the future of serious LLM engineering, but you should only adopt it when hand-written prompts stop scaling.

Everything in this module so far has assumed you're writing prompts directly: you craft a string, you parametrize it, you test it, you iterate. That's the normal way, and it's the right way for small applications.

But as systems grow, hand-written prompts start to creak. You have 40 templates, each tuned for one specific model, and you need to migrate to a new provider. You spent three weeks engineering a prompt for task A and now you need to engineer another one for task B. Every prompt depends on the model under the hood, so every provider switch means rewriting everything.

A different approach has been gaining ground: **program language models instead of prompting them**. Describe *what* you want in typed code, let a framework produce the prompts, let an optimizer improve them. This chapter is a survey of the frameworks that implement this approach, what they give you, and when they're worth the switch.

---

## The philosophical shift

A traditional prompt-engineering workflow looks like this:

1. Write a prompt string.
2. Test it on some inputs.
3. Notice it fails on certain cases.
4. Reword the prompt.
5. Test again.
6. Repeat until good enough.

The prompt is a string. You tune it by hand. When the model changes, you tune again.

The *programming* approach inverts this. It separates three concerns that are usually mashed together:

- **The interface** — what goes in, what comes out (types, schemas)
- **The strategy** — how to prompt the model (CoT, few-shot, ReAct, etc.)
- **The prompt text itself** — the specific words that implement the strategy

Traditional prompting conflates all three into one string. The programming approach lets you declare the interface and strategy cleanly and *compile* them into prompts automatically. The prompt becomes a build artifact, not source code.

DSPy is the most prominent example of this approach. Its tagline — "programming, not prompting" — captures the shift.

---

## DSPy: the compiler for language models

**DSPy** (Stanford NLP, originally called Demonstrate-Search-Predict) is a declarative framework for building LLM applications. Instead of writing prompt strings, you write Python code that declares the shape of your task.

The three core abstractions are **signatures**, **modules**, and **optimizers**.

### Signatures

A signature is a typed interface for a task. It says what goes in, what comes out, and what the task is about — without saying how to prompt.

```python
import dspy

class ClassifySupportTicket(dspy.Signature):
    """Classify a support ticket into one of the standard categories."""

    ticket: str = dspy.InputField(
        description="The customer's support message, as plain text"
    )
    category: str = dspy.OutputField(
        description="One of: billing, technical, general, spam"
    )
    confidence: float = dspy.OutputField(
        description="Confidence in the classification, 0.0 to 1.0"
    )
```

No prompt. Just: here's the input, here's the output, here's what the task is (in the docstring). This is a *signature* in the programming sense — the type signature of the operation.

You can also express signatures as terse strings:

```python
classify = dspy.Predict("ticket -> category, confidence: float")
```

Both forms work. The Python class form is more documented and explicit.

### Modules

A module is *how* to invoke the signature. The interesting part of DSPy is that modules come in varieties that correspond to the reasoning techniques you learned earlier in this module:

- **`dspy.Predict(sig)`** — direct prompting, no extra reasoning
- **`dspy.ChainOfThought(sig)`** — prompts the model to reason step-by-step before answering (Lesson 5)
- **`dspy.ReAct(sig, tools=...)`** — reason + act loop with tool calls (Lesson 6)
- **`dspy.ProgramOfThought(sig)`** — model emits code that's executed to produce the answer
- **`dspy.BestOfN(sig, n=5)`** — sample N times and pick the best

Swap modules without changing the signature:

```python
# Try it zero-shot
classifier = dspy.Predict(ClassifySupportTicket)

# Add CoT
classifier = dspy.ChainOfThought(ClassifySupportTicket)

# Add tool use
classifier = dspy.ReAct(ClassifySupportTicket, tools=[lookup_account, fetch_order_history])
```

Each module generates different prompts under the hood — plain for `Predict`, step-by-step with reasoning examples for `ChainOfThought`, Thought/Action/Observation loops for `ReAct`. You don't have to write any of them. The framework does.

### Composing modules

You build larger programs by composing modules into Python classes. Each class is a DSPy module that can use other modules:

```python
class SupportBot(dspy.Module):
    def __init__(self):
        self.classify = dspy.ChainOfThought(ClassifySupportTicket)
        self.respond_billing  = dspy.ChainOfThought("ticket -> response")
        self.respond_technical = dspy.ReAct(
            "ticket -> response",
            tools=[fetch_docs, run_diagnostic],
        )
        self.respond_general = dspy.Predict("ticket -> response")

    def forward(self, ticket):
        result = self.classify(ticket=ticket)
        if result.category == "billing":
            return self.respond_billing(ticket=ticket).response
        elif result.category == "technical":
            return self.respond_technical(ticket=ticket).response
        elif result.category == "spam":
            return "[spam — no response]"
        else:
            return self.respond_general(ticket=ticket).response

bot = SupportBot()
response = bot(ticket="The app crashes when I click login")
```

Each component is a module. The overall flow is Python. There are *no prompt strings anywhere in the code* — DSPy generates them from the signatures and module types.

### Optimizers (teleprompters)

This is the differentiator. DSPy can **automatically improve** the prompts it generates, given training data and a metric.

The optimizer pattern:

```python
# Your training set (same shape as an eval set from Lesson 12)
trainset = [
    dspy.Example(ticket="Why was I charged twice?", category="billing").with_inputs("ticket"),
    dspy.Example(ticket="App keeps crashing",       category="technical").with_inputs("ticket"),
    # ... 20-100 examples
]

# Your evaluation metric
def accuracy_metric(example, prediction, trace=None):
    return prediction.category == example.category

# Compile (optimize) the program against the training data
optimizer = dspy.BootstrapFewShot(metric=accuracy_metric)
optimized_bot = optimizer.compile(SupportBot(), trainset=trainset)

# Use the optimized version
response = optimized_bot(ticket="The app crashes when I click login")
```

What happened during `compile`:

1. DSPy ran the bot on every training example.
2. For each one, it recorded the intermediate reasoning, the module calls, and the final output.
3. It picked the ones that matched `accuracy_metric` (the correct classifications) and used those as few-shot examples for the next round.
4. It regenerated the prompts with the chosen examples baked in.
5. Now every call uses a prompt that's been optimized against your actual data.

Different optimizers do different things:
- **`BootstrapFewShot`** — synthesizes good in-context examples
- **`MIPROv2`** — generates and tests many variations of the instructions themselves
- **`BootstrapFinetune`** — goes further and actually fine-tunes the underlying model using the synthesized data
- **`KNNFewShot`** — picks few-shot examples dynamically based on similarity to the current input

A typical optimization run costs a few dollars in API credits and 10-20 minutes. In exchange, you get prompts tuned to your specific data — without writing them.

### Why DSPy is compelling

The headline claim: **you can swap models without rewriting prompts.** Your signatures and modules stay the same. When you switch from Claude to Llama, DSPy generates new prompts appropriate for the new model. When you upgrade from GPT-5 to GPT-5.1, DSPy can re-optimize against your metric with the new model's behavior.

Your "prompt engineering" work now happens in two places:

1. **Designing good signatures and module structures** — this is real engineering.
2. **Writing good metrics** — which is really writing good evals from Lesson 12.

The prompt strings themselves become artifacts, not source code. You don't check them into git; they're regenerated on demand.

### When DSPy makes sense

- You're building a system with many LLM-touching components (not a single prompt)
- You have real training data you can use as an eval set
- You're likely to switch models or upgrade them
- You're comfortable with Python and typed interfaces
- Performance matters enough to justify optimization

### When DSPy is overkill

- You have a single prompt that works fine
- You don't have training data and can't easily generate any
- Your team isn't familiar with Python type annotations and declarative frameworks
- You're exploring / prototyping and need to see prompts directly

DSPy has a learning curve. It's not a drop-in replacement for string templates — it's a different way of thinking about LLM programming. Adopt it when you've felt the pain of hand-written prompts at scale.

---

## Instructor: structured outputs as a framework

We covered Instructor in Lesson 8. It's worth mentioning again here as the simplest of the prompt-programming libraries: it doesn't aim to optimize prompts, only to make *structured outputs* robust across providers.

The core abstraction is a Pydantic model → typed output. The framework handles:

- Provider-agnostic API (same interface for OpenAI, Claude, Gemini, DeepSeek, Ollama)
- Automatic retries with validation feedback
- Streaming partial objects
- Type-safe outputs in Python

```python
import instructor
from pydantic import BaseModel

class Ticket(BaseModel):
    category: str
    urgency: int

client = instructor.from_provider("anthropic/claude-sonnet-4-6")

ticket = client.create(
    response_model=Ticket,
    messages=[{"role": "user", "content": "The payment failed and it's urgent"}],
    max_retries=3,
)
```

Instructor is the right choice when:

- You just want reliable structured outputs across providers
- You don't need prompt optimization
- You're not yet ready for DSPy's level of abstraction

Many teams use Instructor alongside hand-written prompts: the framework handles the structured-output plumbing, you handle the prompt crafting. It's a lighter-weight option than DSPy and easier to introduce incrementally.

---

## BAML: prompts as a DSL

**BAML** (Boundary Markup Language, from BoundaryML) takes a different approach: instead of writing prompts in Python, you write them in a domain-specific language that compiles to type-safe clients in your language of choice (TypeScript, Python, etc.).

A BAML file looks like this:

```baml
class SupportTicket {
  category string @description("billing, technical, general, or spam")
  urgency int @description("1 to 5, 5 is most urgent")
}

function ClassifyTicket(ticket: string) -> SupportTicket {
  client Claude
  prompt #"
    Classify the following support ticket.

    Ticket:
    {{ ticket }}

    {{ ctx.output_format }}
  "#
}
```

And the generated Python client:

```python
from baml_client import b

ticket = b.ClassifyTicket(ticket="Payment failed and it's urgent")
# ticket is a fully-typed SupportTicket
```

What BAML gives you:

- **Prompts as first-class files** with syntax highlighting, linting, and IDE support
- **Strong types across the boundary** — Python and TypeScript clients are auto-generated
- **Built-in test runners** — write tests alongside prompts and run them with the CLI
- **Structured outputs with any provider** — BAML handles the schema negotiation

Compared to DSPy, BAML is less about *optimization* and more about *engineering ergonomics*. It doesn't automatically improve prompts; it makes hand-writing them more pleasant and testable. It's popular with teams that want explicit prompts (no magic) but more structure than f-strings.

Worth investigating if: you want to see the prompts, you're a multi-language team (Python + TypeScript), you want CI-testable prompts with fewer moving parts than DSPy.

---

## LangChain and LlamaIndex: the big-ecosystem frameworks

**LangChain** and **LlamaIndex** are the two most widely-deployed frameworks for LLM applications, and they have the biggest ecosystems. They're also the most controversial — many experienced engineers consider them over-engineered, with abstraction layers that obscure what's actually happening.

### What they give you

- **Prompt templates** with variable substitution (Lesson 9's Level 2)
- **Chain abstractions** — compose LLM calls into pipelines
- **Retrieval integrations** — hundreds of supported vector databases, document loaders, text splitters
- **Tool integrations** — hundreds of pre-built tools and plugins
- **Memory management** — conversation history, summarization, vector stores
- **Agent abstractions** — pre-built ReAct, plan-and-execute, etc.

The ecosystems are genuinely vast. If you need to integrate with an obscure vector database or an unusual data source, LangChain or LlamaIndex probably has a pre-built connector.

### The controversy

The criticism is that these frameworks often make simple things complex. A task that would take 50 lines of plain Python calling the provider SDK can become 200 lines of framework-specific classes, with multiple layers of abstraction hiding what's actually in the prompt or what's being retrieved.

For small projects, the overhead rarely pays off. For large projects with many moving parts, the ecosystem can be genuinely useful — but at the cost of framework lock-in and harder debugging.

### When to use them

- You're building a RAG system and need to integrate with specific data sources they support (Module 3 goes deeper on this)
- You're adopting an agent harness and want pre-built pieces
- You value ecosystem breadth over minimalism
- You're comfortable debugging abstractions

### When to avoid them

- You're prototyping and need to see what's in the prompt
- You're building something simple (one or two LLM calls)
- You value having full control over every API call
- Your team isn't already familiar with the framework

A pragmatic compromise many teams settle on: **use LangChain's/LlamaIndex's retrieval and text-splitting tools, but write your own prompts and LLM calls.** You get the ecosystem benefits without the full-stack framework lock-in.

---

## Other tools worth knowing

A quick survey of other frameworks you'll encounter:

### Guidance (Microsoft)

Embeds prompts in a constrained template language that can force specific tokens at specific positions. Strong at enforcing format. Not widely adopted outside Microsoft's own tooling.

### Outlines

A Python library for constrained generation using regex, context-free grammars, or JSON schema. Produces outputs that are *guaranteed* to match the constraint by masking out invalid tokens at each step. Strong for open-weight models where strict-mode structured outputs aren't available natively.

### Mirascope

A Python library that tries to split the difference between Pydantic/Instructor minimalism and LangChain's ecosystem. Template-based, typed, multi-provider. Less mature than the incumbents but actively developed.

### LangGraph

LangChain's successor framework for agent workflows. Models your application as a state machine with nodes and edges. Better than LangChain for multi-step agent orchestration, with clearer state transitions. Worth knowing if you're in LangChain's orbit.

### Haystack (deepset)

A Python framework for building search and RAG applications. Strong on retrieval pipelines. Popular in enterprise RAG deployments, particularly in Europe.

### LLM-as-a-Library pattern

An increasingly common approach: don't use a framework at all, just write small Python modules around the provider SDKs. Keep your prompts as strings (or Jinja2 templates), your validation as Pydantic, your retries as custom code, your LLM calls as direct SDK calls. This is how many experienced practitioners ship production LLM apps in 2026. It's not a framework — it's a deliberate rejection of frameworks.

---

## Choosing a framework (or not)

A decision tree to find your fit:

1. **Is this a small project with one or two LLM calls?**
   → Use the provider SDK directly. Maybe Instructor for structured outputs. No framework.

2. **Do you need multi-provider support and reliable structured outputs?**
   → Use Instructor. It's minimal, composable, and doesn't lock you in.

3. **Do you want explicit, testable prompts in a multi-language codebase?**
   → Use BAML. Compiled clients, CI-friendly, explicit.

4. **Are you building complex RAG with many moving parts?**
   → Consider LangChain or LlamaIndex for the retrieval and integration ecosystem. Write your own prompts and LLM calls to keep control.

5. **Are you building a large system with many LLM touch points that needs to be maintainable long-term?**
   → Consider DSPy. It's the most opinionated framework and has the steepest learning curve, but it's the only one that automatically optimizes prompts against your data.

6. **Are you building an agent harness with a complex state machine?**
   → LangGraph for the state-machine model. Or roll your own (Module 4's harness project).

7. **When in doubt?**
   → Start with SDK + Instructor. Upgrade when you feel actual pain from not having more structure. Don't pre-adopt.

---

## The meta-skill

Whichever direction you take, the real skill is knowing when you need more structure and when you need less. Frameworks solve some problems and create others. Hand-written prompts have their own problems. The job is to pick the right tool for the job you're doing right now, not the job you imagine you'll be doing in a year.

A few principles that stand up across tools:

- **Start simple.** f-strings and a for-loop are a legitimate production pattern.
- **Adopt abstractions only after feeling their absence.** Framework adoption is cheap early and expensive late — but adopting frameworks you don't need is the most expensive option of all.
- **Prefer composable tools over monolithic frameworks.** Instructor + plain SDK + your own retry logic composes well. A kitchen-sink framework locks you in.
- **Always be able to see the actual prompt.** Debug logging, tracing, or whatever — if you can't see what's being sent to the model, you can't debug. This is a hard requirement.
- **Don't treat frameworks as insurance.** A framework doesn't save you from writing evals or thinking about failure modes. Nothing does.

---

## Common pitfalls

- **Adopting DSPy before you have training data.** Without a metric and a dataset, DSPy's optimization can't run. Use it once you have evals.
- **Using LangChain for a 50-line script.** It'll become a 200-line script with extra dependencies.
- **Not looking at the actual prompts.** Every framework should give you a way to see the prompts being sent. If it doesn't, you'll eventually regret adopting it.
- **Assuming "the framework handles it."** Prompt injection, rate limits, cost, evals — none of these are the framework's job. They're yours.
- **Framework hopping.** Starting with LangChain, switching to LlamaIndex, then to DSPy, then to BAML, burning time on migrations each time. Pick one, use it long enough to know its tradeoffs, then decide if it's worth switching.
- **Believing the marketing.** Every framework's website claims to solve prompt engineering. None of them do. They make specific parts easier and other parts harder.
- **Not reading the source code.** Especially for LangChain and LlamaIndex, reading how the framework actually constructs prompts is essential. You'll discover both the clever parts and the worrying parts.

---

## What to remember from this lesson

- Prompt programming frameworks separate the interface (types), strategy (CoT, ReAct), and prompt text from each other — letting you change one without breaking the others.
- **DSPy** is the most ambitious framework: declarative signatures, composable modules, automatic optimization. Use when you have training data and scale pain.
- **Instructor** is the minimalist option: Pydantic-based structured outputs with retries, across providers. Great starting point and composable with other tools.
- **BAML** gives you prompts-as-a-DSL with typed generated clients. Good for multi-language teams who want explicit prompts with structure.
- **LangChain / LlamaIndex** are the big-ecosystem frameworks. Useful for their integrations (especially in RAG), controversial for their abstraction overhead.
- **Don't adopt frameworks preemptively.** Start with SDK + Instructor. Upgrade when you feel the pain.
- Whichever tool you pick, always be able to see the actual prompts being sent. Debuggability is non-negotiable.
- The meta-skill is knowing when more structure helps and when it hurts.

This closes Module 2. You now have a full toolkit for prompt engineering — from the basics of message structure through advanced reasoning, structured outputs, multimodal, security, evaluation, and programming-not-prompting. The next module goes into retrieval-augmented generation, where prompts meet external knowledge.

---

## References

- DSPy, *Programming language models*. https://dspy.ai/
- DSPy paper, *DSPy: Compiling Declarative Language Model Calls into Self-Improving Pipelines*. https://arxiv.org/abs/2310.03714
- Instructor, *Python library for structured outputs*. https://python.useinstructor.com/
- BAML, *Boundary Markup Language*. https://docs.boundaryml.com/
- LangChain, *Framework documentation*. https://python.langchain.com/docs/introduction/
- LangGraph, *Stateful agent workflows*. https://langchain-ai.github.io/langgraph/
- LlamaIndex, *Data framework for LLMs*. https://docs.llamaindex.ai/
- Outlines, *Constrained generation*. https://github.com/outlines-dev/outlines
- Guidance, *Controlled generation from Microsoft*. https://github.com/guidance-ai/guidance
- Hamel Husain, *Fuck you, show me the prompt* (an argument against over-abstraction). https://hamel.dev/blog/posts/prompt/

---

[← Lesson 12](12-evaluating-prompts.md) | [Back to Prompt Engineering](../README.md) | [Module Complete → Next Module: RAG](../../03-rag/)
