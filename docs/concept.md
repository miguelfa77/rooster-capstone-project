# Rooster — Concept

A plain-language description of what Rooster is and how it fits together, at the level of ideas rather than code.

## What it is for

Rooster helps people who invest in residential property in Valencia (Spain) answer market questions quickly. It has two faces:

- **Ask Rooster** — a chat where you ask, in Spanish and in your own words, things like "¿qué barrios tienen mejor rentabilidad?", "enséñame pisos en venta en Ruzafa por debajo de 250.000 €" or "de estos, quédate con los que rinden más del 6 %". Rooster answers with a short explanation plus whatever form fits: key figures, a table, a chart, a map of neighborhoods or a map of individual listings.
- **Intelligence** — a fixed dashboard of the city: three headline market figures, a city map you can color by a metric and overlay with transit stops, tourist apartments, listings and land parcels, a ranking of neighborhoods by investment appeal, and a yield-versus-liquidity chart.

It covers one city and one domain on purpose. It is a public demo with no user accounts.

## The shape: one front door, thin clients

```
data pipeline → metric layer ─┬─ dashboard queries ─┐
                              └─ agent ─────────────┴─ API ── clients (web app; phone line later)
                                   └─ sandbox (optional, custom charts)
```

- The **metric layer** is the single source of numbers.
- The **API** is the single front door. It serves two kinds of request.
  - **Questions** go to the agent: a language model with tools.
  - **Dashboard requests** are fixed queries with no model involved, because the dashboard always asks the same things. Their figures come from the metric layer. Their map overlays (boundaries, stops, tourist apartments, listings, parcels) are plain geography, read from the cleaned data beneath it.
- **Clients** hold no business logic. They don't fetch data, compute metrics, or decide which rows belong in which chart. They present the finished, self-describing pieces the API returns. The web app is the first client; a phone line is the planned second.

## What the model knows about the data

With every question the model gets a **catalogue**:
- each metric with its Spanish and English names, meaning and formula;
- compositional ideas such as "buena zona", written out as filters over those metrics;
- place references such as "el centro", resolved to neighborhoods.

The catalogue is written by hand because it holds what the metric layer doesn't: synonyms, phrasing and local knowledge. A test fails the build if it names a metric the metric layer doesn't produce.

## How a question gets answered

1. **Choosing the data.** The question, the conversation so far and the conversation's memory go to a reasoning model with a small set of tools. The memory holds the person's budget, the neighborhoods in focus and their priorities, and it belongs to that one conversation.
   - The tools: look up neighborhood metrics, search listings, get a time series, compute an aggregate, find transit stops or tourist apartments, resolve a place, or narrow a result already shown.
   - The model never writes database queries; the system turns each tool call into a safe query.
2. **Learning from mistakes.** A wrong tool call (an unknown metric, a bad filter) returns the real error together with the valid options. The model reads it and tries again, a few times at most. There is no second "checker" model; the model is given better information instead.
3. **Composing the answer.** A separate model call turns the retrieved data into an ordered list of **answer blocks**: prose, key figures, tables, bar/scatter/line charts, neighborhood maps and listing maps.
   - Why separate: choosing how to present known data is a different job from choosing what to fetch, and this output must follow a strict structure that clients rely on.
   - Each block names what it shows, and the system attaches the exact rows it needs, so every block is complete on its own.
   - **Custom charts (optional).** Investors also ask for price distributions, month-by-neighborhood heatmaps and side-by-side comparisons, which the standard charts can't draw. For these the model may write chart code. It runs in an isolated sandbox with no internet access and a 30-second limit, and the result is shown in an isolated frame. If the sandbox fails or isn't running, the block becomes a table, so Rooster works fully without it. The sandbox must keep earning its place:
     - a custom chart that shows up often in evaluation runs becomes a standard block;
     - if custom charts stop appearing, the sandbox is removed.
4. **Streaming.** While this happens, the client receives progress notes ("buscando anuncios en Ruzafa…") and the prose as it is written. Streamed prose is provisional. When the turn is saved, the client replaces everything with the saved blocks, so the finished answer and a reloaded one are identical.
5. **Remembering.** After the prose has streamed, a small, cheaper model updates the conversation's memory. If that fails, the previous memory is kept. The rows behind the latest answer are kept too, one set per tool that produced them. A follow-up such as "de estos…" names which of those results it narrows, and the narrowing happens without asking the database again.

## Turns and conversations

- **One turn at a time.** A conversation lives on the server and runs one turn at a time. A message sent while Rooster is answering is turned away ("Rooster sigue respondiendo…"). The text stays in the box, and the page checks every few seconds and shows the answer once it is saved.
- **All or nothing.** The finished turn is saved in one database transaction: the new messages with their ordered blocks, the updated memory, and the rows behind the answer. Only then is the client told the turn is done. If the turn fails or the service dies, nothing of it is stored. The client keeps what it had already shown, marks it as interrupted ("La respuesta se interrumpió antes de terminar."), and the person can ask again.
- **No stuck conversations.** A running turn holds a lock in the database on a connection of its own. That connection is separate from the ones used for queries, and it keeps no transaction open. If the service dies, the connection drops and the database releases the lock by itself.
- **Stopping a turn (planned, for the phone line).** A caller interrupting is a request to stop. The agent checks for it between stages (before each tool step, before composing), discards the turn and releases the lock. The web app's "stop" will use the same operation.
- **Plain failures.** When something fails, Rooster says so in Spanish with a suggestion for what to try next, instead of showing a broken result.

## Trust, abuse and cost

- **Secret key.** The API accepts only callers holding its secret key: the web app's server and, later, the phone line. Browsers never talk to the API or see the key.
- **Conversation IDs.** Without accounts, a conversation's random, unguessable ID is its only key. Anyone holding it can read the conversation, which is acceptable for a demo.
- **Data stays data.** Scraped listing data reaches the model only as tool results, never as instructions. Model-written code runs only in the sandbox.
- **Cost is bounded per turn and in total.**
  - Per turn: a message is at most 2,000 characters, the tool loop is a few steps at most, and a failed custom chart is retried at most once.
  - In total: a per-minute limit on how many answers may start, counted in the database across every API instance and every client. Past it, people are told to try again in a minute.
  - One person could still open many conversations. For a demo, the global limit is the backstop.

## What the data is

- **Sources.** A pipeline collects listings for sale and for rent, neighborhood boundaries, cadastral parcels, streets, public transit stops and registered tourist apartments. It cleans them, places each listing on the map and matches it to its neighborhood.
- **Metrics.** The metric layer defines named, tested metrics: rental yield, median prices, price per square meter, transit access, tourist density and an investment score. Every figure the agent or the dashboard shows comes from these, so a definition like "gross yield" exists in exactly one place.
- **Data confidence.** Each neighborhood result carries a confidence label: strong, adequate or low. It is not a metric of its own. It is read off two metrics, the number of listings for sale and for rent, by one shared rule. It lets Rooster warn when a figure rests on few listings.
- **Freshness.** The data is refreshed by re-running the pipeline, and Rooster describes the market as of the last refresh.

## Knowing it works

- **Evaluation set.** A fixed set of Spanish questions is run against the live model. It includes follow-ups and questions marked as answerable by voice. Each run records the tools each question used and the kinds of blocks it got back, and is compared to a recorded baseline. Any question whose tools or block kinds changed is reported for a person to judge. This catches routing and format regressions. It doesn't grade the prose and doesn't yet check figures; the next step is to check key numbers against direct queries of the metric layer.
- **Logs.** Every stage of every turn is logged.
- **Contract checks.** The web app's types are generated from the API's published description of every message. If one side changes in a way the other doesn't expect, the build fails.

## Principles

- **Trust the model, inform it well.** When something goes wrong, the fix is better information at the failing step, not another validation layer.
- **Narrow, honest plumbing.** Safe queries, real errors, structured answers. Nothing guessed, nothing hidden.
- **One definition of everything.** Numbers come from the metric layer. Answer shapes are defined in the agent and shared through the published contract. Limits and locks live in the database.
- **Thin clients.** A new client needs only the API.
- **Spanish product, English engineering.**
- **Portable.** Each service is a container configured only by environment variables. The whole system runs on one machine with one command, and nothing ties it to a particular host.

## Deliberately out of scope (for now)

User accounts and private conversations, other cities, new data sources, and the phone line itself. The phone line is designed for but not built. Beyond the spoken mode and stopping a turn, it will need a fast first word, speaking the prose as it streams, and one conversation per call.

**Spoken mode.** For the phone line, each message will say whether it wants a *visual* answer (today's only kind) or a *spoken* one. A spoken answer adds no new vocabulary. It is just prose blocks: a few short sentences with rounded numbers, written by the same step from the same data and streamed as they are written. The caller hears the progress notes while data is fetched, then the first sentence as soon as it exists.
