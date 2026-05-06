# 🎯 DaanvEdge
### DaanvEdge — The Calculated Bet on IPL Player Value

> *"Every IPL auction is a betting table. Most teams bet on reputation.
> DaanvEdge bets on data."*

---

## What is DaanvEdge?

**DaanvEdge** is a Moneyball-style player intelligence engine for the Indian
Premier League. The word *daanv* (दांव) means a calculated bet or strategic
stake — the kind a chess player places three moves ahead, not a gamble.

That is exactly what this project models: the **informed, data-driven wager**
on a player's true worth — past, present, and future — before the auction
hammer falls or the fantasy deadline closes.

---

## The Three-Horizon Model

| Horizon    | Question it answers                          | Output                        |
|------------|----------------------------------------------|-------------------------------|
| **Past**   | What has this player reliably delivered?     | Baseline score, peak phases   |
| **Present**| What is their actual form right now?         | Rolling form index            |
| **Future** | Where is this player heading?                | Trajectory band + value delta |

Most cricket analysis lives entirely in the Past column.
DaanvEdge is built around the gap between Present and Future.

---

## The Moneyball Parallel

| Baseball (Moneyball)                  | Cricket (DaanvEdge)                          |
|---------------------------------------|----------------------------------------------|
| Market overvalues batting average     | Market overvalues big names at auction       |
| OBP was the underpriced metric        | Context-adjusted strike rate is underpriced  |
| Scouts bet on reputation              | Franchises bid on brand, not data            |
| Billy Beane found the mispriced edge  | DaanvEdge finds the mispriced daanv          |

---

## Core Metrics

- **Daanv Score** — composite bet-worthiness rating per player per match context
- **Value Delta** — gap between auction hammer price and modeled true value
- **Form Index** — exponentially weighted rolling performance window
- **Trajectory Band** — Rising / Peak / Declining via age curve modelling
- **Context Splits** — powerplay vs death, home vs away, surface, opposition quality
- **Bet Confidence** — statistical confidence interval on the Daanv Score

---

## Why DaanvEdge?

*Daanv* captures something *alpha* or *edge* alone cannot:
a deliberate, strategic commitment made under uncertainty, with full awareness
of the risk — backed by the best available information.

That is the game IPL auction rooms play every March.
DaanvEdge puts the data on your side of the table.

---

## Roadmap

-  Historical data pipeline (IPL 2008–present via cricsheet)
-  Daanv Score engine (context-adjusted composite)
-  Auction value model (bid price vs true value delta)
-  Trajectory predictor (age curves + role classification)
-  Player comparison dashboard
-  Fantasy league integration (Dream11 / My11Circle context)

---

## Stack

`Python` · `pandas` · `scikit-learn` · `Streamlit` · `PostgreSQL` · `cricsheet`

---

*Built with chai, cricket data, and calculated conviction.*
