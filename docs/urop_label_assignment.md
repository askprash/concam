# Labeling assignment

This is the same task as the earlier round, so this skips the how-to — it's all
below if you ever want a refresher. Log in at
**https://hex.mit.edu/~prash/concam/index.html?user=YOURNAME** — use that exact
link (with `index.html` and the `?user=YOURNAME` on the end) so the export comes
back tagged with your name and you only get prompted for the password once.

**Please label as many days as possible — but only days before April 27th.** Any
date in the calendar view or dropdown that falls before that cutoff is fair game, and all of it
helps. More coverage across different skies and seasons is exactly what we're
short on, so don't ration yourself; the only rule is one day per export file.

Two things worth keeping in mind since they're easy to drift on: go through
*every* pass in the sidebar, not just the ones with a detection line — the system
misses most real contrails, and a clear contrail it scored zero on is the single
most valuable label you can give us. And keep **unsure** to a minimum; before reaching
for it, just ask "is there a line sitting roughly on the flight track, yes or
no?" — if you can answer that at all, pick contrail or no_contrail, and save
unsure for when the sky itself is genuinely unreadable.

When you finish a day, hit **Export** (it downloads `<date>_<yourname>_labels.json`)
and email it back, one file per day. It auto-saves in the browser, so it's
totally fine to split a day across a few sittings — and if you're tight on time,
finish the **daylight hours** of a day rather than skimming the whole 24 h.

## More info

The auto-detector is poorly calibrated right now. It fails two ways, and you should catch both:

- **It misses most real contrails** (caught only ~1 in 15 on our test day). So go through
  *every* pass in the sidebar — including ones with a low/zero score and no detection line —
  and label them. A clear contrail the system scored low is the most valuable label you can
  give us, because it shows exactly where the detector is blind.
- **It also fires on lots of things that aren't contrails** (tons of false positives). So a
  drawn box / detection line is **not** evidence of a contrail. Do **not** rubber-stamp the
  system's calls. If it drew a line but the sky behind the plane is clear, that's
  `no_contrail` — and that's a valuable label too.

Bottom line: **decide each pass on what you see in the video, not on what the box says.**

Use the toggles above the video to control what's drawn on top of it. **ADS-B**
(on by default) draws the flight track and the dot marking where the plane is in
each frame — that's the line a real contrail should sit on. **Detection lines**
overlays the auto-detector's guesses; it's off by default on purpose, since you
should decide each pass on the video itself, but you can flip it on to see what
the system thought. **Measure** lets you drag to measure a feature when you want
to. There's also a **detected only** filter on the pass list if you ever want to
jump straight to the passes the system flagged — just remember the valuable
labels are the ones it *missed* too, so don't stay filtered for long but it could be useful to find the false positive cases first.

Turn on the **Zoom box** to get a floating 2.5× magnifier that follows the plane
around the frame, with a dot inside it pinned to the aircraft's exact position.
Faint, short contrails are often invisible at full-frame scale and only show up
in the zoom, so it's worth leaving on — especially for the low/zero-score passes
where you're checking whether the detector missed something real.

On the right there's a **heatmap** of flight levels × time for the day. Hover any
cell to see which flights contributed to it, and **click a cell to seek the video
straight to that time** — it's a fast way to jump around the day instead of
scrubbing, and to land on the busy stretches. Use the pop-out button to blow the
heatmap up to a larger view if the small one is hard to read.


| Label | Use when |
|-------|----------|
| **contrail** | There is a line-shaped cloud trailing the aircraft, lying *along the flight track* shown by the overlay. Long or short both count. |
| **no_contrail** | The aircraft passes and leaves no line — clear sky behind it, or only unrelated natural cloud. |
| **unsure** | You genuinely cannot tell (aircraft is behind thick cloud, image too washed out, etc.). **Use this sparingly** — see below. |

### Keep "unsure" rare

In an earlier round one reviewer marked ~40% of passes "unsure." That usually means the
*criteria* were unclear, not that the cases were truly ambiguous. Before choosing unsure,
ask: is there a line roughly co-located with the flight track, yes or no? If you can answer
that, pick contrail/no_contrail.

## How to tell a contrail from things that look like one

- **Natural cirrus** is diffuse, patchy, and **not aligned with one flight path**. A contrail
  is a *narrow line that sits on top of the ADS-B track* and (early on) starts right at the
  aircraft. If a streak doesn't line up with the overlay track, it's probably natural cloud →
  `no_contrail` for *this* aircraft.
- **Lens / sensor artifacts** stay in the **same pixel position** frame to frame and don't
  drift with the sky. If a "line" never moves as the day progresses, it's not a contrail.
- **Old, spread-out contrails** from an *earlier* plane may drift across the scene. Only label
  `contrail` if the line belongs to *the aircraft this pass is about* — i.e. it trails that
  specific track. Note unrelated drifting contrails in the notes field instead.


## The persistence slider (1–5)

How long the contrail lasts behind the aircraft:

- **1** — vanishes within a few seconds of forming (short-lived).
- **3** — hangs for a minute or two, stays a clean line.
- **5** — lingers and spreads, broadening into cirrus-like haze over many minutes.

Persistence is what the science cares about most (persistent contrails are the
climate-relevant ones), so it's worth a moment's thought — but don't agonize; a rough 1/3/5
is fine.


## Notes field — use it for anything odd

Especially: the ADS-B dot not sitting on the head of the contrail, two planes' contrails
crossing, a contrail with no plane, or anything that made the call hard. These notes directly
drive our calibration work. A previous labler found that our default filtering of flights further than 100km might have been too premature and some other flights were actually creating visible contrails.



