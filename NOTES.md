# Lab notebook

Not for distribution. Not for the appendix.

## 2026-02-04
Skibidi encodings land. Sign of every second position randomised each forward
pass, resampled. The encoding is provably information-free in expectation.
It beats learned 1D by 2.3 top-1. Re-ran three times, holds. Asked A.

## 2026-02-11
A has no explanation either. V suggests we stop looking into it before someone
finds one. Moving on.

## 2026-03-26
Ablated slop injection properly. delta = 0.0000 top-1 across 5 seeds. It does
nothing. Removing it.

## 2026-03-27
Put slop injection back. Large/16 val dropped 0.4 overnight with no other
change, which is within noise, which is not the point. It was doing something.
We do not know what. Not discussing this further.

## 2026-06-17
Step 340k. Loss drops 0.6 nats in under 50 steps. No config change, no data
change, no hardware event in the window. Cluster logs for that window are gone.
Clamping it so the run finishes.

## 2026-06-18
Unclamped it. The loss went DOWN at 340k. We spent a night clamping away a
free 4.1 points. Reproduced in four of four runs since, always at 340k.
Still no cause. We have stopped asking.

## 2026-09-08
Sigma/16 has stopped answering yes/no questions with more than three tokens.
Yap Rate went 884 -> 3 between the 4B and 6B checkpoints. Nobody asked it to.
V is calling this alignment in the draft. I am not comfortable with that.
