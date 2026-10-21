"""
core/diagrams.py
================
Small teaching diagrams, one per lesson tab, shown with `show_dg("m3:1")` (module : tab index).
Every picture is drawn by core.ui.diagram so they all share one size, font and colour style.
"""
from __future__ import annotations

from core.ui import diagram

C = dict(blue="#dbeafe", green="#bbf7d0", yellow="#fef9c3", orange="#fed7aa", red="#fecaca",
         grey="#f1f5f9", purple="#e9d5ff", teal="#ccfbf1", white="#ffffff")
LOOP = 'style=dashed color="#dc2626" fontcolor="#dc2626" constraint=false'


def chain(items, rd="LR", loops=(), extra=""):
    """items: [(label, colour[, shape])]; loops: [(from, to, label)] drawn as red dashed return arrows."""
    out = [f"digraph {{ rankdir={rd};"]
    for i, it in enumerate(items):
        lab, col = it[0], it[1]
        shape = f" shape={it[2]}" if len(it) > 2 else ""
        out.append(f'n{i} [label="{lab}" fillcolor="{C[col]}"{shape}];')
    out.append(" -> ".join(f"n{i}" for i in range(len(items))) + ";")
    for a, b, lab in loops:
        out.append(f'n{a} -> n{b} [label=" {lab}" {LOOP}];')
    out.append(extra + "}")
    return "\n".join(out)


def raw(body, rd="LR"):
    return f"digraph {{ rankdir={rd}; {body} }}"


def node(i, label, col="grey", shape=None):
    s = f" shape={shape}" if shape else ""
    return f'{i} [label="{label}" fillcolor="{C[col]}"{s}];'


DG: dict[str, tuple[str, str]] = {}

# ============================== Module 1 ==============================
DG["m1:1"] = (raw(
    node("ml", "Machine learning", "blue") +
    node("sup", "Supervised\\n(has labels)", "green") + node("uns", "Unsupervised\\n(no labels)", "yellow") + node("rl", "Reinforcement\\n(rewards)", "orange") +
    node("reg", "Regression\\nhouse price", "white") + node("cls", "Classification\\nspam or not", "white") +
    node("clu", "Clustering\\ncustomer groups", "white") + node("dim", "Dimensionality\\nreduction", "white") +
    node("mb", "Model-based\\nplans with a map", "white") + node("mf", "Model-free\\nlearns by trying", "white") +
    "ml -> sup; ml -> uns; ml -> rl; sup -> reg; sup -> cls; uns -> clu; uns -> dim; rl -> mb; rl -> mf;", "LR"),
    "The family tree of machine learning. Each branch is a technique with an example below.")
DG["m1:4"] = (chain([("1950s-70s\\nMimic talk\\n(ELIZA, Markov)", "grey"), ("1980s-90s\\nLearn patterns\\n(neural nets)", "blue"),
                     ("2000s\\nMemory\\n(LDA, RNN)", "teal"), ("2014-17\\nImagine\\n(GANs)", "purple"),
                     ("2017-19\\nUnderstand\\n(Transformer, BERT)", "orange"), ("2020-now\\nPartner\\n(GPT, Gemini, Claude)", "green")]),
              "Six phases: from copying conversation to being a creative partner.")
DG["m1:5"] = (chain([("Model scores\\nevery next word", "blue"), ("Temperature\\nreshapes the odds", "orange"), ("Top-k\\nkeep the k best", "yellow"),
                     ("Top-p\\nkeep the top p%", "yellow"), ("Pick one word\\n(random draw)", "green")]),
              "How one word is chosen. Temperature, top-k and top-p are the dials in the middle.")
DG["m1:6"] = (raw(node("ga", "Generative AI", "blue") +
                  node("ben", "Benefits\\l- Productivity\\l- Creativity\\l- Personalisation\\l- Education\\l- Data augmentation\\l", "green") +
                  node("ris", "Challenges\\l- Bias and misinformation\\l- Plagiarism\\l- Job displacement\\l- Security and privacy\\l- Over-reliance\\l", "red") +
                  "ga -> ben; ga -> ris;"), "Powerful, but it needs responsible use.")

# ============================== Module 2 ==============================
DG["m2:0"] = (chain([("Your prompt", "grey"), ("Split into\\ntokens", "yellow"), ("Transformer scores\\nthe next token", "blue"), ("Pick one token", "orange"),
                     ("Answer so far", "green")], loops=[(3, 2, "append the token and repeat")]),
              "An LLM writes one token at a time, feeding its own output back in.")
DG["m2:2"] = (chain([("Text\\n'battery life'", "grey"), ("Tokens\\n[bat][tery][ life]", "yellow"), ("Token IDs\\n[4521][9833][1207]", "orange"),
                     ("Embeddings\\n(lists of numbers)", "purple"), ("Transformer", "blue")]),
              "Machines read numbers, not words.")
DG["m2:3"] = (chain([("4 answers to\\none prompt", "grey"), ("Humans rank\\nthem", "orange"), ("Reward model\\nlearns their taste", "yellow"),
                     ("Scores the LLM's\\nnew answers", "blue"), ("LLM improves\\n(reinforcement)", "green")]),
              "Your ranking is the training data for the reward model.")
DG["m2:4"] = (raw(node("llm", "LLMs", "blue") + node("op", "Open source\\nweights are public", "green") + node("cl", "Closed source\\nused through an app or API", "orange") +
                  node("o1", "Gemma", "white") + node("o2", "LLaMA 3", "white") + node("o3", "Qwen", "white") + node("o4", "BLOOM", "white") + node("o5", "Falcon", "white") +
                  node("c1", "GPT-4", "white") + node("c2", "Claude", "white") + node("c3", "Gemini", "white") +
                  "llm -> op; llm -> cl; op -> o1; op -> o2; op -> o3; op -> o4; op -> o5; cl -> c1; cl -> c2; cl -> c3;", "LR"),
              "Two families of LLMs and well-known examples of each.")
DG["m2:5"] = (raw(node("ly", "Layers", "blue") + node("ne", "Neurons", "blue") + node("wb", "Weights and biases", "yellow") + node("pa", "= Parameters\\n7B means 7 billion", "green") +
                  node("slm", "SLM: 1-8B parameters\\nlaptop, phone, private, cheap", "teal") + node("lm", "LLM: 70B and up\\ncloud, strongest, costly", "purple") +
                  "ly -> ne -> wb -> pa; pa -> slm [style=dotted]; pa -> lm [style=dotted]; slm -> lm [dir=both label=\" trade-off\"];"),
              "Parameters are the learned weights. More of them means more capacity and more hardware.")
DG["m2:6"] = (raw(node("v1", "Vague prompt\\n'tell me about climate change'", "red") + node("v2", "Generic answer", "red") +
                  node("g1", "Specific prompt\\nrole + task + audience + length", "green") + node("g2", "Useful, focused answer", "green") +
                  "v1 -> v2; g1 -> g2;"), "Same model, different prompt, very different result.")
DG["m2:7"] = (raw(node("b", "Bias", "red") + node("b2", "Audit data and outputs", "green") + node("p", "Privacy", "red") + node("p2", "Never paste secrets", "green") +
                  node("i", "Intellectual property", "red") + node("i2", "License data, label AI content", "green") +
                  node("m", "Misinformation", "red") + node("m2", "Fact-check every claim", "green") +
                  node("d", "Deepfakes", "red") + node("d2", "Watermark and detect", "green") +
                  "b -> b2; p -> p2; i -> i2; m -> m2; d -> d2;"), "Each risk has a practical safeguard.")

# ============================== Module 3 ==============================
DG["m3:0"] = (chain([("Prompt\\n(your instruction)", "yellow"), ("LLM", "blue"), ("Response", "green")]), "A clearer prompt gives a better response.")
DG["m3:1"] = (raw(node("ro", "1 Role\\n'You are a career coach'", "yellow") + node("ta", "2 Task\\n'Improve this resume'", "orange") +
                  node("co", "3 Context\\nthe resume text", "teal") + node("fo", "4 Format\\n'bullets, under 150 words'", "purple") +
                  node("pr", "Assembled prompt", "blue") + node("ll", "LLM", "green") +
                  "ro -> pr; ta -> pr; co -> pr; fo -> pr; pr -> ll;"), "Four blocks come together to make one prompt.")
DG["m3:2"] = (chain([("Start of prompt\\nstrong attention", "green"), ("Middle\\nweakest attention", "red"), ("End of prompt\\nstrong attention", "green")]),
              "Put the most important instructions at the start or the end.")
DG["m3:3"] = (chain([("Vague prompt", "red"), ("Add who, what,\\nfor whom, format", "yellow"), ("Clear prompt", "green"), ("Better answer", "blue")],
                    loops=[(3, 1, "still off? refine")]), "Crafting a prompt is a small loop of try and refine.")
DG["m3:4"] = (chain([("Zero-shot\\ninstruction only", "grey"), ("One-shot\\n1 example", "yellow"), ("Few-shot\\nseveral examples", "orange"), ("Role-based\\n'act as an expert'", "purple")]),
              "More guidance usually means more reliable output.")
DG["m3:5"] = (raw(node("t", "Template\\n'Summarise {doc} for {audience}'", "yellow") + node("v", "Variables\\ndoc, audience", "orange") +
                  node("f", "Filled prompt", "teal") + node("l", "LLM", "blue") + node("a", "Tailored answer", "green") +
                  "t -> f; v -> f; f -> l; l -> a;"), "One template, many prompts, filled in at run time.")

# ============================== Module 4 ==============================
DG["m4:0"] = (raw(node("d1", "Direct prompt", "grey") + node("d2", "LLM guesses", "red") + node("d3", "Answer: maybe wrong,\\nno reasoning shown", "red") +
                  node("c1", "CoT prompt", "grey") + node("c2", "Step 1", "yellow") + node("c3", "Step 2", "yellow") + node("c4", "Answer you can check", "green") +
                  "d1 -> d2 -> d3; c1 -> c2 -> c3 -> c4;"), "Basic prompting can jump to an answer. Step-by-step reasoning shows its work.")
DG["m4:1"] = (chain([("Question", "grey"), ("'Let's think\\nstep by step'", "yellow"), ("Step 1", "blue"), ("Step 2", "blue"), ("Final answer", "green")]), "Chain-of-Thought adds the steps between question and answer.")
DG["m4:2"] = (raw(node("e", "Worked examples\\nquestion -> steps -> answer", "yellow") + node("q", "New question", "grey") + node("l", "LLM copies the\\nreasoning style", "blue") + node("a", "Answer", "green") +
                  node("b", "Bias risk: skewed or ordered\\nexamples skew the answer", "red") + "e -> l; q -> l; l -> a; b -> e [style=dashed];"),
              "Show the model how to reason. Choose balanced examples.")
DG["m4:3"] = (chain([("Many\\nquestions", "grey"), ("Cluster by\\nsimilarity", "yellow"), ("Pick 1 per\\ncluster", "orange"), ("LLM writes its\\nown reasoning", "blue"),
                     ("Use as few-shot\\nexamples", "purple"), ("Solve the\\ntarget question", "green")]), "The model builds its own examples, then uses them.")
DG["m4:4"] = (raw(node("q", "Question", "grey") + node("p1", "Reasoning path 1\\nanswer 42", "blue") + node("p2", "Reasoning path 2\\nanswer 42", "blue") + node("p3", "Reasoning path 3\\nanswer 40", "blue") +
                  node("v", "Majority vote", "orange") + node("a", "Final answer: 42", "green") + "q -> p1; q -> p2; q -> p3; p1 -> v; p2 -> v; p3 -> v; v -> a;"),
              "Ask several times and trust the answer most paths agree on.")
DG["m4:7"] = (raw(
    node("s", "Start", "grey", "ellipse") + node("q1", "Simple, one fact?", "yellow", "diamond") + node("z", "Zero-shot", "green") +
    node("q2", "Needs a fixed\\nstyle or format?", "yellow", "diamond") + node("f", "Few-shot", "green") +
    node("q3", "Needs multi-step\\nreasoning?", "yellow", "diamond") + node("cot", "Chain-of-Thought", "green") +
    node("q4", "Must be very\\naccurate?", "yellow", "diamond") + node("sc", "Self-Consistency", "green") +
    node("tot", "Tree-of-Thought\\n(plan or compare options)", "green") +
    's -> q1; q1 -> z [label=" yes"]; q1 -> q2 [label=" no"]; q2 -> f [label=" yes"]; q2 -> q3 [label=" no"]; '
    'q3 -> q4 [label=" yes"]; q3 -> tot [label=" no"]; q4 -> sc [label=" yes"]; q4 -> cot [label=" no"];', "TB"),
    "A quick way to choose a technique.")

# ============================== Module 5.1 ==============================
DG["m51:1"] = (chain([("Big file", "grey"), ("Split into\\nnumbered packets", "yellow"), ("Packets travel by\\ndifferent routes", "blue"), ("Re-assembled\\nin order", "green")]),
               "Small numbered packets can be re-sent individually if one is lost.")
DG["m51:3"] = (raw(node("c1", "Client", "grey") + node("s1", "Serverful\\nalways-on server you manage", "orange") +
                   node("c2", "Client", "grey") + node("s2", "Serverless\\ncode runs only when called", "teal") + node("p", "Pay per call", "green") +
                   "c1 -> s1; c2 -> s2 -> p;"), "Serverful: always running. Serverless: runs on demand and you pay per call.")
DG["m51:4"] = (raw(node("o", "{ } JSON object", "blue") + node("k1", 'name: Asha', "white") + node("k2", 'age: 21', "white") + node("k3", 'skills: [ ]', "yellow") +
                   node("k4", 'address: { }', "yellow") + node("s1", 'Python', "white") + node("s2", 'SQL', "white") + node("a1", 'city: Pune', "white") +
                   "o -> k1; o -> k2; o -> k3; o -> k4; k3 -> s1; k3 -> s2; k4 -> a1;", "LR"), "JSON is keys and values. Values can be lists [ ] or nested objects { }.")
DG["m51:5"] = (chain([("Request\\nmethod, URL,\\nheaders, body", "yellow"), ("API server", "blue"), ("Response\\nstatus code,\\nJSON body", "green")]), "Every API call is a request and a response.")
DG["m51:6"] = (raw(node("r", "HTTP status code", "grey") + node("a", "2xx Success\\n200 OK", "green") + node("b", "3xx Redirect\\n301 Moved", "yellow") +
                   node("c", "4xx Your mistake\\n400, 401, 404, 429", "orange") + node("d", "5xx Server's mistake\\n500, 503", "red") + "r -> a; r -> b; r -> c; r -> d;"),
               "The first digit tells you who should act.")

# ============================== Module 5.2 ==============================
DG["m52:0"] = (chain([("Your app", "yellow"), ("HTTPS API call", "grey"), ("Black-box LLM\\nweights, GPUs, serving\\nare hidden", "purple"), ("Text answer", "green")]),
               "You send a request and get an answer without running the model yourself.")
DG["m52:1"] = (raw(node("a", "Your app\\nrequest + API key", "yellow") + node("s", "Server checks the key", "blue", "diamond") +
                   node("ok", "Valid: answer returned", "green") + node("no", "Invalid: 401 error", "red") + node("e", "Key lives in .env,\\nnever in your code", "teal") +
                   'a -> s; s -> ok [label=" yes"]; s -> no [label=" no"]; e -> a [style=dashed];'), "The API key is your app's ID card.")
DG["m52:2"] = (chain([("Your code\\nclient = genai.Client(key)", "yellow"), ("generate_content(\\nmodel, contents)", "blue"), ("Gemini API", "purple"), ("resp.text", "green")]),
               "The shape of your first programmatic call.")
DG["m52:3"] = (raw(node("i", "Input tokens\\n(your prompt)", "yellow") + node("o", "Output tokens\\n(the answer)", "orange") + node("p", "x price per token", "grey") + node("b", "Your bill", "green") +
                   "i -> p; o -> p; p -> b;"), "You pay for tokens going in and tokens coming out.")
DG["m52:5"] = (raw(node("r", "Request", "grey") + node("q", "Seen this\\nbefore?", "yellow", "diamond") + node("c", "Cached answer\\nfree and instant", "green") +
                   node("l", "Call the LLM", "blue") + node("s", "Store answer", "teal") + 'r -> q; q -> c [label=" yes"]; q -> l [label=" no"]; l -> s -> c [style=dotted];'), "Do not pay twice for the same answer.")
DG["m52:6"] = (chain([("Call", "grey"), ("Fails\\n429 or 5xx", "red"), ("Wait 1 s", "yellow"), ("Retry", "blue"), ("Wait 2 s", "yellow"), ("Retry", "blue"), ("Wait 4 s", "yellow"), ("Give up or\\nuse fallback", "orange")]),
               "Exponential backoff: wait longer after each failure.")
DG["m52:7"] = (raw(node("u", "User input", "grey") + node("v", "Check length\\nand patterns", "yellow", "diamond") + node("sn", "Sanitise", "teal") + node("pt", "Prompt template", "blue") +
                   node("ll", "LLM", "purple") + node("oc", "Check output", "yellow", "diamond") + node("ok", "User", "green") + node("bad", "Reject", "red") +
                   'u -> v; v -> sn [label=" ok"]; v -> bad [label=" bad"]; sn -> pt -> ll -> oc; oc -> ok [label=" safe"]; oc -> bad [label=" unsafe"];'), "Validate before the model and check after it.")
DG["m52:8"] = (chain([("Every LLM call", "grey"), ("Log: time, model,\\ntokens, latency, errors", "yellow"), ("Dashboard", "blue"), ("Alerts", "orange")]), "You cannot fix what you cannot see.")

# ============================== Module 6 ==============================
DG["m6:0"] = (chain([("Prompt template", "yellow"), ("LLM", "blue"), ("Output parser", "teal"), ("Structured result", "green")]), "LangChain snaps these parts together into a chain.")
DG["m6:1"] = (raw(node("lc", "LangChain", "blue") + "".join(node(f"c{i}", t, "white") for i, t in enumerate(
    ["LLM integration", "Prompt templates", "Chains", "Memory", "Tools", "Agents", "Retrieval", "Streaming", "Batch", "Deployment"])) +
                   "".join(f"lc -> c{i};" for i in range(10))), "Ten building blocks in one framework.")
DG["m6:2"] = (raw(node("l", "LLM", "blue") + node("c1", "chunk 1", "yellow") + node("c2", "chunk 2", "yellow") + node("c3", "chunk 3", "yellow") + node("u", "Screen fills\\nas chunks arrive", "green") +
                   "l -> c1 -> c2 -> c3 -> u;"), "Streaming shows words as they are written.")
DG["m6:3"] = (raw(node("i", "100 inputs", "grey") + node("b", ".batch()", "yellow") + node("a", "LLM call", "blue") + node("b2", "LLM call", "blue") + node("c", "LLM call", "blue") + node("o", "100 outputs", "green") +
                   "i -> b; b -> a; b -> b2; b -> c; a -> o; b2 -> o; c -> o;"), "Run many inputs together instead of one at a time.")
DG["m6:4"] = (raw(node("t", "LLM text\\n(hopefully JSON)", "grey") + node("p", "Pydantic schema\\nchecks types and fields", "yellow", "diamond") + node("ok", "Valid object", "green") +
                   node("er", "ValidationError", "red") + node("rt", "Retry with the\\nerror message", "orange") + 't -> p; p -> ok [label=" valid"]; p -> er [label=" invalid"]; er -> rt; rt -> t [style=dashed];'),
               "Validate model output before the rest of your code trusts it.")
DG["m6:5"] = (chain([("Call model", "grey"), ("Error?", "yellow", "diamond"), ("Retry", "blue"), ("Fallback model", "orange"), ("Safe default message", "green")]), "Plan for failure at each step.")
DG["m6:6"] = (chain([("User input", "grey"), ("Input guardrail", "yellow"), ("LLM", "blue"), ("Output guardrail", "yellow"), ("User", "green")],
                    extra='bad [label="Blocked or rewritten" fillcolor="#fecaca"]; n1 -> bad [style=dashed]; n3 -> bad [style=dashed];'), "Guardrails check both what goes in and what comes out.")

# ============================== Module 7 ==============================
DG["m7:0"] = (raw(node("t1", "Turn 1\\n'I am Priya'", "grey") + node("l1", "LLM", "blue") + node("t2", "Turn 2\\n'What is my name?'", "grey") + node("l2", "LLM\\n(starts fresh)", "blue") +
                   node("r", "'I don't know'", "red") + node("fix", "Fix: send the earlier\\nturns again with each request", "green") + "t1 -> l1; t2 -> l2 -> r; fix -> l2 [style=dashed];"),
               "A model has no memory of its own. Each request starts blank.")
DG["m7:1"] = (raw(node("m1", "system\\n'Be concise and simple'", "purple") + node("m2", "user\\n'What is a loop?'", "yellow") + node("m3", "assistant\\n'A loop repeats...'", "green") +
                   node("m4", "user\\n'Show an example'", "yellow") + node("ll", "LLM", "blue") + node("m5", "next assistant\\nmessage", "green") + "m1 -> m2 -> m3 -> m4 -> ll -> m5;"),
               "A chat is a list of role-tagged messages. The whole list goes to the model each time.")
DG["m7:2"] = (raw(node("s", "System prompt\\n'Reply in 2 lines, friendly'", "purple") + node("u1", "User message 1", "yellow") + node("u2", "User message 2", "yellow") +
                   node("r1", "Reply 1\\nfollows the rule", "green") + node("r2", "Reply 2\\nfollows the rule", "green") + "s -> u1 [style=dashed]; s -> u2 [style=dashed]; u1 -> r1; u2 -> r2;"),
               "One system instruction shapes every reply.")
DG["m7:3"] = (chain([("Messages", "yellow"), ("Template turns them\\ninto one text", "grey"), ("Tokens", "orange"), ("Predict next token", "blue"), ("Reply, token by token", "green")],
                    loops=[(3, 3, "")]), "A chat reply is still next-token prediction.")
DG["m7:4"] = (raw(node("a", "Turn 1", "red") + node("b", "Turn 2", "red") + node("c", "Turn 3", "yellow") + node("d", "Turn 4", "yellow") + node("e", "Turn 5", "yellow") + node("w", "Inside the context window\\nthe model can use these", "green") +
                   node("f", "Forgotten:\\nfell out of the window", "red") + "a -> b -> c -> d -> e; a -> f [style=dotted]; c -> w [style=dotted];", "LR"),
               "The context window is a fixed-size whiteboard. The oldest lines get wiped.")
DG["m7:5"] = (raw(node("h", "Long history", "grey") + node("s1", "Full history", "white") + node("s2", "Truncate\\nlast N turns", "white") + node("s3", "Summarise\\nolder turns", "white") + node("s4", "Prioritise\\nrelevant turns", "white") +
                   node("p", "Prompt", "blue") + node("l", "LLM", "green") + "h -> s1; h -> s2; h -> s3; h -> s4; s1 -> p; s2 -> p; s3 -> p; s4 -> p; p -> l;"), "Four ways to fit a long chat in the window and cut cost.")
DG["m7:6"] = (chain([("User message", "yellow"), ("Chatbot replies", "green"), ("Every N turns:\\nsummarise old turns", "orange"), ("Summary + latest turns", "teal"), ("Next prompt", "blue")],
                    loops=[(4, 0, "conversation continues")]), "An auto-summary chatbot keeps the gist and drops the clutter.")
DG["m7:7"] = (raw(node("c", "Your OpenAI-style code", "yellow") + node("u", "Change base_url\\nand api_key", "orange") + node("o", "OpenAI endpoint", "grey") + node("g", "Gemini endpoint", "blue") +
                   "c -> u; u -> o [style=dotted]; u -> g;"), "Same code, different endpoint.")

# ============================== Module 8 ==============================
DG["m8:0"] = (raw(node("l1", "LLM alone\\nknowledge cutoff", "red") + node("a1", "Old or guessed answer", "red") + node("l2", "LLM + tools", "blue") + node("t1", "Search", "white") + node("t2", "Weather API", "white") +
                   node("t3", "Calculator", "white") + node("a2", "Fresh, correct answer", "green") + "l1 -> a1; l2 -> t1; l2 -> t2; l2 -> t3; t1 -> a2; t2 -> a2; t3 -> a2;"), "Tools connect the model to today's world.")
DG["m8:1"] = (raw(node("l", "LLM", "blue") + node("a", "Information tools\\nsearch, lookup, retrieval", "yellow") + node("b", "Compute tools\\ncalculator, code", "orange") + node("c", "API tools\\nweather, maps, payments", "teal") + node("d", "Action tools\\nsend email, book, update", "purple") +
                   "l -> a; l -> b; l -> c; l -> d;"), "Four kinds of tools.")
DG["m8:2"] = (raw(node("t", "Tool card", "blue") + node("n", "name\\nget_weather", "white") + node("d", "description\\nwhen to use it", "yellow") + node("p", "parameters (schema)\\ncity: string", "orange") +
                   node("l", "LLM reads the card\\nand decides to call it", "green") + "t -> n; t -> d; t -> p; n -> l; d -> l; p -> l;"), "The LLM sees only the card, not your code.")
DG["m8:4"] = (chain([("User question", "grey"), ("LLM picks a tool\\nand arguments", "blue"), ("Your code\\nruns the tool", "yellow"), ("Result goes back\\nto the LLM", "orange"), ("Final answer", "green")]),
              "The LLM chooses, your code runs, the LLM answers.")
DG["m8:5"] = (chain([("User", "grey"), ("LLM tool call", "blue"), ("get_weather()", "yellow"), ("Result", "orange"), ("LLM writes\\nthe answer", "green")],
                    extra='bug [label="Common bug: result printed,\\nnever sent back to the LLM" fillcolor="#fecaca"]; n3 -> bug [style=dashed color="#dc2626"];'),
              "Close the loop: the tool result must go back to the model.")
DG["m8:6"] = (raw(node("s1", "Sequential", "grey") + node("a1", "Tool A 2s", "yellow") + node("b1", "Tool B 2s", "yellow") + node("c1", "Tool C 2s", "yellow") + node("t1", "Total 6s", "red") +
                   node("s2", "Fan-out / fan-in", "grey") + node("a2", "Tool A 2s", "yellow") + node("b2", "Tool B 2s", "yellow") + node("c2", "Tool C 2s", "yellow") + node("t2", "Total about 2s", "green") +
                   "s1 -> a1 -> b1 -> c1 -> t1; s2 -> a2; s2 -> b2; s2 -> c2; a2 -> t2; b2 -> t2; c2 -> t2;"), "Independent tools can run at the same time.")
DG["m8:7"] = (raw(node("c", "Tool call", "grey") + node("t", "Try it", "yellow", "diamond") + node("ok", "Result to LLM", "green") + node("er", "Catch the error", "red") +
                   node("j", "Send error as JSON to the LLM", "orange") + node("l", "LLM explains or retries", "blue") + 'c -> t; t -> ok [label=" works"]; t -> er [label=" fails"]; er -> j -> l;'), "Fail at the tool level, not the app level.")

# ============================== Module 9 ==============================
DG["m9:0"] = (chain([("Gap in knowledge", "grey"), ("Model fills it with\\nsomething plausible", "yellow"), ("Confident but\\nwrong answer", "red")],
                    extra='fx [label="Ground in sources, ask for citations,\\nverify" fillcolor="#bbf7d0"]; n2 -> fx [style=dashed];'), "Hallucination: fluent, confident and false.")
DG["m9:1"] = (raw(node("m", "LLM limits", "blue") + node("a", "Outdated knowledge", "white") + node("b", "Bias", "white") + node("c", "Limited context window", "white") + node("d", "Reasoning gaps", "white") + node("e", "Different answer each run", "white") +
                   "m -> a; m -> b; m -> c; m -> d; m -> e;"), "Know the limits before you trust the output.")
DG["m9:2"] = (chain([("Quirk tests\\nedge, long, tricky,\\nambiguous inputs", "yellow"), ("Your app", "blue"), ("Pass or fail", "orange", "diamond"), ("Fix the prompt\\nor the code", "green")],
                    loops=[(3, 1, "re-test")]), "Stress-test before shipping, like testing a phone for overheating.")
DG["m9:3"] = (raw(node("o", "Model output", "blue") + node("r", "Reference answer", "yellow") + node("m1", "Accuracy / exact match", "white") + node("m2", "Precision and recall", "white") + node("m3", "BLEU / ROUGE\\n(word overlap)", "white") + node("s", "Score", "green") +
                   "o -> m1; o -> m2; o -> m3; r -> m1; r -> m2; r -> m3; m1 -> s; m2 -> s; m3 -> s;"), "Classic metrics compare output with a known answer.")
DG["m9:4"] = (chain([("Tier 1\\nautomated signals\\nfast, cheap", "blue"), ("Tier 2\\nhuman judgment\\nslower, nuanced", "orange"), ("Tier 3\\naggregate decision\\nship or fix", "green")]), "Three tiers of evaluation.")
DG["m9:5"] = (raw(node("a", "Accuracy 1-5", "white") + node("b", "Relevance 1-5", "white") + node("c", "Clarity 1-5", "white") + node("d", "Safety 1-5", "white") + node("h", "Human rater\\napplies the rubric", "yellow") + node("s", "Average score", "green") +
                   "h -> a; h -> b; h -> c; h -> d; a -> s; b -> s; c -> s; d -> s;"), "A rubric turns opinions into comparable scores.")
DG["m9:7"] = (chain([("Write prompt", "yellow"), ("Run it", "blue"), ("Evaluate output", "orange"), ("Tweak prompt", "green")], loops=[(3, 1, "repeat until good")]), "Prompts are never set and forget.")
DG["m9:8"] = (raw(node("p", "Weak prompt", "red") + node("a", "Add context", "white") + node("b", "Add constraints", "white") + node("c", "Add examples", "white") + node("d", "Change the format", "white") + node("e", "Split into steps", "white") + node("g", "Stronger prompt", "green") +
                   "p -> a; p -> b; p -> c; p -> d; p -> e; a -> g; b -> g; c -> g; d -> g; e -> g;"), "Five edits that usually help.")


def show_dg(key: str, max_w: float = 8.5, max_h: float = 5.0):
    """Show the diagram stored under `key` (e.g. 'm3:1') with its caption."""
    dot, caption = DG[key]
    diagram(dot, caption, max_w, max_h)

DG["m4:6"] = (chain([("Question", "grey"), ("Thought", "blue"), ("Action\\n(tool call)", "orange"), ("Observation\\n(tool result)", "green"), ("Final answer", "grey")],
                    loops=[(3, 1, "think again")]),
              "ReAct loop: the model thinks and picks an action, your code runs the tool, the observation goes back in.")
