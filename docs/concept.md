# Rooster — Concept

A plain-language description of what Rooster is and how it is put together, at the level of ideas rather than code.

## What it is for

Rooster helps people who invest in residential property in Valencia (Spain) answer market questions quickly. It has two faces:

- **Ask Rooster** — a chat where you ask, in Spanish and in your own words, things like "¿qué barrios tienen mejor rentabilidad?", "enséñame pisos en venta en Ruzafa por debajo de 250.000 €" or "de estos, quédate con los que rinden más del 6 %". Rooster answers with a short explanation plus whatever form fits the question best: key figures, a table, a chart, a map of neighborhoods, a map of individual listings, or occasionally a custom chart.
- **Intelligence** — a fixed dashboard of the city: three headline market figures, a city map you can color by a metric and overlay with transit stops, tourist apartments, listings and land parcels, a ranking of neighborhoods by investment appeal, and a yield-versus-liquidity chart.

It covers one city and one domain on purpose. There are no user accounts.

## The core idea: one brain, thin clients

All the intelligence lives in one place, the **agent**. Everything a person sees is a **client** that asks the agent questions and draws what it sends back. The web app is the first client; a phone line that speaks answers is planned as a second one, using the same agent.

The clients never contain business logic. They don't decide what data to fetch, compute metrics, or choose which numbers belong in which chart. They receive finished, self-describing answer pieces and render them.

## How a question gets answered

1. **Understanding.** The question, the recent conversation and a small memory of what this person seems to care about (budget, neighborhoods in focus, priorities) go to a reasoning language model.
2. **Using tools.** The model doesn't write database queries. It picks from a small catalogue of well-defined tools: look up neighborhood metrics, search listings, get a time series, compute an aggregate, find transit stops or tourist apartments, resolve a place like "el centro", or narrow down the results it already showed ("de estos…"). The system turns each tool call into a safe query.
3. **Learning from mistakes.** If a tool call is wrong (an unknown metric, a bad filter), the system returns the real error together with the valid options. The model reads it and tries again, a few times at most. There is no second "checker" model; the model is given better information instead.
4. **Composing the answer.** A second model call writes the answer as an ordered list of **answer blocks**: prose, key figures, tables, standard charts (bar, scatter, line), neighborhood maps and listing maps. Each block names what it shows; the system attaches the exact rows it needs, so every block is complete on its own. For the rare visual that the standard blocks can't express, the model can write chart code. That code runs in an isolated **sandbox** service, and if it fails the answer falls back to a table.
5. **Streaming.** While this happens, the person sees progress notes ("buscando anuncios en Ruzafa…") and the prose as it is written, so there is never a silent wait.
6. **Remembering.** The finished answer is saved. The memory of the person's interests is updated, and the data just shown is kept so a follow-up can filter it without asking the database again.

## What the data is

A pipeline collects listings for sale and for rent, neighborhood boundaries, cadastral parcels, streets, public transit stops and registered tourist apartments. It cleans them, places each listing on the map, and matches it to its neighborhood. A layer of named, tested metrics sits on top: rental yield, median prices, price per square meter, transit access, tourist density, an investment score and a data-confidence indicator. The agent and the dashboard read only these metrics, so a definition like "gross yield" exists in exactly one place.

## The pieces

| Piece | Responsibility |
|---|---|
| Data pipeline + metric layer | Turn raw sources into trustworthy, named metrics. |
| Agent | Understand questions, use tools, compose answer blocks, remember context. |
| API service | The agent's front door: start a conversation, send a message and stream back progress, text and answer blocks, reload a conversation, read dashboard data. It publishes a precise description of every message it sends. |
| Sandbox service | Run model-written chart code in isolation, only when needed. |
| Web app | Chat page and dashboard. Its server holds the API's secret key and relays requests and streams. The browser never talks to the API directly. |

The web app's types are generated from the API's published description, so a change on one side that the other doesn't expect is caught at build time.

## Conversations

A conversation lives on the server. You can reload the page, or leave for the dashboard and come back, and the conversation, including an answer still being written, is still there. While Rooster answers, a second message waits; you're told Rooster is still replying, and your text is kept. A finished answer looks the same live and after a reload, because both are drawn from the same saved, ordered blocks. When something fails, Rooster says so plainly, in Spanish, with a suggestion for what to do next, instead of showing a broken result.

## Principles

- **Trust the model, inform it well.** When something goes wrong, the fix is better information at the failing step, not another layer of validation.
- **Narrow, honest plumbing.** The system builds safe queries, reports real errors, and draws from structured descriptions. It doesn't guess, and it doesn't hide failures.
- **One definition of everything.** Metrics are defined once in the metric layer. Answer shapes are defined once in the agent and shared with clients through the published contract.
- **Thin clients.** A new client (the phone line) should need only the API, not a copy of the agent's logic.
- **Spanish product, English engineering.** Everything a user sees is in Spanish.
- **Portable.** Each service is a container configured only by environment variables. The whole system runs on one machine with one command. Hosting is chosen later, and nothing ties the system to a particular host.

## Deliberately out of scope (for now)

User accounts and multi-tenancy, other cities, new data sources, and the phone line itself (designed for, not yet built).
