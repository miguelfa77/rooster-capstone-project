# Rooster — Concept

A plain-language description of what Rooster is and how it fits together, at the level of ideas rather than code.

## What it is for

Rooster helps people who invest in residential property in Valencia (Spain) answer market questions quickly. It has two faces:

- **Ask Rooster** — a chat where you ask, in Spanish and in your own words, things like "¿qué barrios tienen mejor rentabilidad?", "enséñame pisos en venta en Ruzafa por debajo de 250.000 €" or "de estos, quédate con los que rinden más del 6 %". Rooster answers with a short explanation plus whatever form fits: key figures, a table, a chart, a map of neighborhoods or a map of individual listings.
- **Intelligence** — a fixed dashboard of the city: three headline market figures, a city map you can color by a metric and overlay with transit stops, tourist apartments, listings and land parcels, a ranking of neighborhoods by investment appeal, and a yield-versus-liquidity chart.

It covers one city and one domain on purpose. It is a public demo with no user accounts.

## The core idea: one brain, thin clients

All the intelligence lives in one place, the **agent**, behind one **API**. Everything a person sees or hears is a **client** that asks the API and presents what comes back. The web app is the first client. A phone line is the planned second one: it will turn speech into a question and speak the answer.

Clients hold no business logic. They don't fetch data, compute metrics, or decide which rows belong in which chart. They present the finished, self-describing answer pieces they receive.

## What the model knows about the data

The model is not left to guess what exists. With every question it gets a **catalogue**: each metric with its Spanish and English names, meaning and formula; compositional ideas such as "buena zona" written out as filters over those metrics; and place references such as "el centro" resolved to neighborhoods. The catalogue is written by hand because it holds things the metric layer doesn't (synonyms, phrasing, local knowledge). A test fails the build if it names a metric the metric layer doesn't produce.

## How a question gets answered

1. **Choosing the data.** The question, the conversation so far and the conversation's memory go to a reasoning language model with a small catalogue of tools. The memory holds the person's stated budget, neighborhoods in focus and priorities, and it belongs to that one conversation. The tools are: look up neighborhood metrics, search listings, get a time series, compute an aggregate, find transit stops or tourist apartments, resolve a place such as "el centro", or narrow the results already shown ("de estos…"). The model never writes database queries; the system turns each tool call into a safe query.
2. **Learning from mistakes.** If a tool call is wrong (an unknown metric, a bad filter), the system returns the real error together with the valid options. The model reads it and tries again, a few times at most. There is no second "checker" model; the model is given better information instead.
3. **Composing the answer.** A separate call to the model turns the retrieved data into an ordered list of **answer blocks**. It is separate for two reasons. Choosing how to present data that is already known is a different job from choosing which data to fetch. And its output is held to a strict structure that clients can rely on, which a tool-using loop's output is not.
   - The block kinds are prose, key figures, tables, standard charts (bar, scatter, line), neighborhood maps and listing maps.
   - Each block names what it shows, and the system attaches the exact rows it needs, so every block is complete on its own.
   - For the phone line, each message will say whether it wants a *visual* answer (today's only kind) or a *spoken* one. A spoken answer adds no new vocabulary. It is just prose blocks, a few short sentences with rounded numbers, written by the same step from the same data and streamed as they are written. The caller hears the progress notes while data is fetched and the first sentence as soon as it exists.
   - **Custom charts** are an optional extra, kept because investors ask for price distributions (histograms), neighborhood-by-month heatmaps and side-by-side comparisons, which the five standard charts can't draw. For these the model may write chart code, which runs in a separate, isolated sandbox. If the sandbox fails or isn't running, that block becomes a table, so Rooster works fully without it. The sandbox has to keep earning its place. A custom chart that shows up often in the evaluation runs is promoted to a standard block. If custom charts stop appearing, the sandbox is removed.
4. **Streaming.** While this happens the client receives progress notes ("buscando anuncios en Ruzafa…") and the prose as it is written, so nobody waits in silence.
5. **Remembering.** After the prose has streamed, so the person is already reading the answer, a small and cheaper model updates the conversation's memory. If it fails, the previous memory is kept. The turn completes when the memory is saved along with the answer. The rows behind the latest answer are kept, and only those, so "de estos…" can filter what was just shown without asking the database again.

## Turns and conversations

- **One turn at a time.** A conversation lives on the server and runs one turn at a time. If you send while Rooster is still answering, the message is turned away with "Rooster sigue respondiendo…", your text stays in the box, and the page checks every few seconds and shows the answer once it is saved.
- **All or nothing.** The finished turn is saved in a single database transaction: the new messages with their ordered blocks, the updated memory, and the rows behind the answer. Only after that is the client told the turn is done. If the service dies mid-answer, nothing half-written is stored, and you see "La respuesta se interrumpió antes de terminar." and can ask again.
- **No stuck conversations.** While a turn runs, it holds a lock in the database on a connection of its own, kept apart from the connections the agent uses for queries. No transaction stays open meanwhile. If the service dies, the connection drops and the database releases the lock by itself, so nothing needs to expire or be cleaned up.
- **Survives navigation and reloads.** You can leave for the dashboard and come back, or reload, and the conversation is still there. An answer looks the same live and after a reload, because both come from the same saved, ordered blocks.
- **Plain failures.** When something fails, Rooster says so in Spanish with a suggestion for what to try next, instead of showing a broken result.

## Trust, abuse and cost

- **Secret key on the server.** The API accepts only callers that hold its secret key: the web app's server and, later, the phone line. Browsers never talk to the API or see the key.
- **Unguessable conversation IDs.** Without accounts, a conversation's random ID is its only key. Anyone holding the ID can read it, which is acceptable for a demo.
- **Spending cap.** Each turn has a fixed ceiling on model use: a message is at most 2,000 characters, the tool loop takes a few steps at most, and a failed custom chart is retried at most once. On top of that, one limit at the API caps how many answers may start per minute across all clients. Past it, people are told to try again in a minute. Together these cap model spend whichever client sends the questions. A conversation can't run two turns at once, but one person could open many conversations. For a demo, the global cap is the backstop.
- **Sandboxed custom charts.** Model-written chart code runs only in the sandbox, and its output is shown in an isolated frame.

## What the data is

A pipeline collects listings for sale and for rent, neighborhood boundaries, cadastral parcels, streets, public transit stops and registered tourist apartments. It cleans them, places each listing on the map and matches it to its neighborhood.

A layer of named, tested metrics sits on top: rental yield, median prices, price per square meter, transit access, tourist density and an investment score. The agent and the dashboard both read only these metrics, so a definition like "gross yield" exists in exactly one place. The one deliberate exception is a data-confidence label (high, medium or low, by number of listings). It is computed where results are assembled, not in the metric layer, because it must also apply to row sets the metric layer never sees, such as a filtered "de estos…" result. That lets Rooster warn when a figure rests on few listings.

The data is refreshed by re-running the pipeline. Rooster describes the market as of the last refresh.

## Knowing it works

- **Evaluation set.** A fixed set of Spanish questions, including follow-ups and questions marked as answerable by voice, is run against the live model. Each run records which tools each question used and which kinds of blocks came back. A run is compared to a recorded baseline, and any question whose tools or block kinds changed is reported for a person to judge. It catches routing and format regressions. It does not grade the prose, and it doesn't yet check figures. The next step is to check a few key numbers against direct queries of the metric layer.
- **Logging.** Every stage of every turn is logged.
- **Contract checks.** The web app's types are generated from the API's published description of every message, so if one side changes in a way the other doesn't expect, the build fails.

## The pieces

| Piece | Responsibility |
|---|---|
| Data pipeline + metric layer | Turn raw sources into trustworthy, named metrics. |
| Agent | Understand questions, use tools, compose answer blocks, keep conversation memory. |
| API | The agent's single front door: conversations, streamed turns, dashboard data, the spending cap. It publishes a precise description of everything it sends. |
| Sandbox (optional) | Run model-written chart code in isolation. |
| Web app | Chat page and dashboard; its server holds the API key and relays requests and streams. |

## Principles

- **Trust the model, inform it well.** When something goes wrong, the fix is better information at the failing step, not another layer of validation.
- **Narrow, honest plumbing.** The system builds safe queries, reports real errors and draws from structured descriptions. It doesn't guess and doesn't hide failures.
- **One definition of everything.** Metrics are defined once, in the metric layer. Answer shapes are defined once, in the agent, and shared with clients through the published contract.
- **Thin clients.** A new client needs only the API.
- **Spanish product, English engineering.**
- **Portable.** Each service is a container configured only by environment variables. The whole system runs on one machine with one command, and nothing ties it to a particular host. Rooster runs one API instance by design. The one-turn-at-a-time rule lives in the database and would survive more instances, but the spending cap is counted inside the instance and would not. Moving that count into the database is the one change needed before scaling out.

## Deliberately out of scope (for now)

User accounts and private conversations, other cities, new data sources, and the phone line itself. The phone line is designed for but not built; beyond the spoken mode it will need low first-word latency, speaking the prose as it streams, letting a caller interrupt, and one conversation per call.
