# Exported agents

Each file is the val-selected best checkpoint of one training run, chosen as the typical run
among its algorithm's 3 seeds (test return closest to the mean across seeds).
Test numbers: greedy policy on test seeds 1000 to 1019, default market.

| File | Algorithm | Source run | Seed | Test return | Test net worth | Mean +- std over 3 seeds |
|---|---|---|---|---|---|---|
| dqn.pt | dqn | dqn_05_s2 | 2 | 114.43 | 1,791,524 | 114.10 +- 1.24 |
| reinforce.pt | reinforce | reinforce_04_s2 | 2 | 76.78 | 1,437,963 | 76.93 +- 0.92 |
| a2c.pt | a2c | a2c_03_s1 | 1 | 88.83 | 1,532,825 | 91.47 +- 5.72 |
| ppo.pt | ppo | ppo_08_s2 | 2 | 116.93 | 1,812,845 | 114.86 +- 4.60 |

best_agent.pt is a copy of ppo.pt (ppo, run ppo_08_s2): the algorithm with the highest mean test return across its 3 seeds.

Watch it: uv run main.py (play.py loads best_agent.pt by default).
