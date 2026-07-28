# Who actually moves the scoreboard? Player value with honest error bars

Every plus-minus argument has the same flaw: the players you share the
floor with. A bench guard who rides shotgun with the MVP posts a glowing
on/off number; a good starter marooned on a bad roster posts a grim one.
This study builds the standard remedy, RAPM-family regularized regression
over every five-man lineup. It builds it twice, in R and Python, for
2023-24 **and** the just-completed 2025-26 season. Then it does the two
things public plus-minus work most often skips: it reports uncertainty
honestly, and it rebuilds the whole thing from a second, independent unit
of observation to see if the answer holds.

<div class="tiles">
<div class="tile"><div class="label">Median interval width</div>
<div class="value">&plusmn;2.5<small> pts/100</small></div>
<div class="sub">One public season cannot separate the 5th-best player from the 30th.</div></div>
<div class="tile"><div class="label">Cross-model agreement</div>
<div class="value">0.96</div>
<div class="sub">Spearman correlation between two independently built models.</div></div>
<div class="tile"><div class="label">Stints parsed</div>
<div class="value">69,767</div>
<div class="sub">Substitution-to-substitution units from all 1,230 games of play-by-play.</div></div>
<div class="tile"><div class="label">Home-court advantage</div>
<div class="value">+2.2<small> pts/100</small></div>
<div class="sub">Estimated as a byproduct of the stint model, not assumed.</div></div>
</div>

<div class="eyebrow beat">The method</div>

## Shrinkage chosen by prediction, not by hand

The model's one free knob, how hard to shrink, is not a taste decision.
Cross-validation chooses the ridge penalty (λ = 3200 in both seasons).
The figure below shows why that choice is defensible rather than
decorative: predictive error falls, bottoms out, and rises again as the
penalty grows. The model simply takes the minimum.

{{fig:2023-24/fig3_cv_curve|The cross-validation curve that picks the ridge penalty. Necessary because shrinkage is the entire difference between this model and raw on/off. This figure is the evidence that the amount of shrinkage was chosen by held-out prediction, not by hand.}}

{{fig:2023-24/fig2_shrinkage|Raw on/off versus the regularized estimate for every qualifying player. The before/after of the method: the players pulled hardest toward zero are exactly the low-minute, star-adjacent profiles whose raw numbers are most contaminated: visible teammate-context correction, one dot per player.}}

<div class="eyebrow beat">The ranking</div>

## The error bars are the real headline

What comes out the other side passes the eye test at the top: Jokić
(+5.5 per 100), Anunoby, Brunson, George, Gilgeous-Alexander lead 2023-24;
Wembanyama (+6.5) and Gilgeous-Alexander lead 2025-26. But the study's
real headline is the **width of the intervals**: the median 95% bootstrap
CI spans about ±2.5 points per 100 possessions. One season of lineup data
genuinely cannot separate the 5th-best player from the 30th, and a model
that pretends otherwise is selling something.

{{fig:2023-24/fig1_player_values|The 2023-24 player values with their bootstrap intervals: the study's most important figure precisely because of the error bars. The ordering is the headline everywhere else; here the honest message is how much the intervals overlap, which is what a front office actually needs to know before acting on a single-season number.}}

<div class="eyebrow beat">The replication</div>

## Two models, no shared plumbing, one answer

A ranking with intervals still deserves suspicion if it only exists in one
model. So the study rebuilds valuation from scratch at a different grain:
69,767 substitution-to-substitution **stints** parsed from all 1,230
games of play-by-play, with an opponent-adjusted design. Its unpenalized
intercept lands at +2.2 points per 100. That is the home-court advantage,
estimated as a byproduct rather than assumed, and a built-in sanity check
that the design matrix means what it claims. The two models share no code
path and no data feed.

{{fig:2023-24/fig4_stint_vs_lineup|Stint-level RAPM against the aggregate lineup model, one dot per player (Spearman 0.96). The convergent-validity figure: two independently built models, different units of observation, different feeds, near-identical player ordering. Agreement this strong is the best public-data evidence that the values reflect the players rather than the plumbing.}}

<div class="eyebrow beat">Five-man units</div>

## Lineup claims are gated by an audit

Lineups themselves get the descriptive treatment, where the volume filter
does the honest work: among five-man units with 1,000+ possessions, the
best of 2023-24 was Milwaukee's Lopez–Lillard–Middleton–Antetokounmpo–
Beasley at +15.3 per 100 over 1,302 possessions. Every lineup total is
gated by exact reconstruction of official team totals. The model is not
allowed to describe a season it cannot re-add.

<div class="eyebrow beat">The takeaway</div>

## Trust the tier, not the rank

The takeaway for a decision-maker is deliberately modest: with one public
season, trust the tier, not the rank; trust a value more when two
independent constructions agree on it; and treat any lineup-value claim
without an error bar as marketing.
