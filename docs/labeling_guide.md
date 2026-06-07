# ConCam contrail labeling guide (for UROP reviewers)

**Site:** https://hex.mit.edu/~prash/concam/ — log in with your username + password,
then add `?user=YOURNAME` to the URL so your exported file is tagged with your name.

## What you're doing in one sentence

For a given day, the camera saw many aircraft fly through its field of view. The system
drew a box and track for **every** aircraft pass (using its ADS-B transponder position).
Your job is to look at each pass and decide: **did that aircraft leave a contrail or not?**

## The most important rule: judge every pass yourself — the system is wrong in BOTH directions

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

## The three labels

| Label | Use when |
|-------|----------|
| **contrail** | There is a line-shaped cloud trailing the aircraft, lying *along the flight track* shown by the overlay. Long or short both count. |
| **no_contrail** | The aircraft passes and leaves no line — clear sky behind it, or only unrelated natural cloud. |
| **unsure** | You genuinely cannot tell (aircraft is behind thick cloud, image too washed out, etc.). **Use this sparingly** — see below. |

### Keep "unsure" rare

In an earlier round one reviewer marked ~40% of passes "unsure." That usually means the
*criteria* were unclear, not that the cases were truly ambiguous. Before choosing unsure,
ask: is there a line roughly co-located with the flight track, yes or no? If you can answer
that, pick contrail/no_contrail. Reserve unsure for cases where the sky itself is unreadable.

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
drive our calibration work.

## When you're done

Click **Export** — it downloads `<date>_<yourname>_labels.json`. **Email that file to Prash**
(or drop it where agreed). Do one day per file. It's fine to label a day in several sittings;
your progress is auto-saved in the browser, so just reopen the same URL to continue.

## Scope / pace

A full day is several hundred passes — that's a multi-hour task, not a 20-minute one. It's
fine to split a day across sessions. If you're short on time, fully finish the **daylight
hours** of your assigned day rather than skimming the whole 24 h — contrails are only visible
in daylight anyway.
