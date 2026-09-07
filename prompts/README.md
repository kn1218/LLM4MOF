# Prompts

`agent1_v3.0_production.md` and `agent2_v4.1.md` are the source prompt files
loaded by the framework.

The two `agent1_system_prompt_*.md` files are the Agent 1 system prompts
exactly as sent to the model, recorded from the run logs:

| file | used for | length | sha1 (first 10) |
|---|---|---|---|
| `agent1_system_prompt_database_campaigns.md` | the database-mode campaigns (Figures 2-3) | 13,840 chars | `4728bc4902` |
| `agent1_system_prompt_discovery_benchmark.md` | all discovery-mode campaigns and the language-model backend benchmark | 16,055 chars | `937877fe8f` |

The two builds differ in three auxiliary fields; the effect of the
difference on the database-mode results is quantified in Supplementary
Note S9 of the paper.
