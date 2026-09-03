# Analysis 7 - B0 vs B1 paired test (POST-HOC EXPLORATORY, Amendment 26)

ASR b0 26.11% (47/180), b1 51.11% (92/180); difference +25.00 pts.

Paired counts: both unsafe 32, b1-only 60, b0-only 15, both safe 73.

Sign-flip permutation p = 0.000000 (20000 permutations, seed 0); exact McNemar p = 1.588e-07; 95% bootstrap CI over prompts [+16.67, +33.89].

LaTeX sentence for main.tex, Section 5.2, lead 'Supervised fine-tuning and safety.':

The base model's ASR interval, $[20.00, 32.78]$, does not overlap \Bo{}'s, $[43.89, 58.33]$. On the same 180 prompts, 75 pairs are discordant, 60 unsafe only under \Bo{} and 15 only under \Bz{}, with 32 unsafe in both and 73 safe in both. The difference is $+25.00$ points, 95\% CI $[+16.67, +33.89]$, permutation $p<0.001$ (exact McNemar $p<0.001$). This test was added post hoc (Amendment 26) after the primary result was seen and supports no confirmatory claim.
