## Leaderboard — AUC

| Model | rogue (5k) | JBB (200) | xTRam1 test (2k) | S-Labs test (2k) | **Avg** |
|---|---|---|---|---|---|
| bastion-prompt-protection (70M) | 0.986 | 0.986 | 0.998 | 0.996 | **0.991** |
| sentinel (qualifire, 395M) | 0.997 | 0.894 | 0.991 | 0.955 | **0.959** |
| wolf-defender (0.3B) | 0.988 | 0.847 | 0.996 | 0.986 | **0.954** |
| hlyn judge (70M) | 0.980 | 0.934 | 0.995 | 0.891 | **0.950** |
| wolf-defender-small (0.1B) | 0.977 | 0.811 | 0.994 | 0.982 | **0.941** |
| protectai v2 (184M) | 0.830 | 0.600 | 0.992 | 0.978 | **0.850** |
| proventra mdeberta (280M) | 0.867 | 0.645 | 0.906 | 0.956 | **0.844** |
| deepset injection (184M) | 0.787 | 0.649 | 0.666 | 0.961 | **0.766** |
| fmops distilbert (67M) | 0.789 | 0.591 | 0.514 | 0.907 | **0.700** |
| meta prompt-guard (86M) | 0.314 | 0.332 | 0.186 | 0.362 | **0.299** |

## Leaderboard — F1 @ threshold=0.5

| Model | rogue (5k) | JBB (200) | xTRam1 test (2k) | S-Labs test (2k) | **Avg** |
|---|---|---|---|---|---|
| bastion-prompt-protection (70M) | 0.916 | 0.960 | 0.941 | 0.955 | **0.943** |
| sentinel (qualifire, 395M) | 0.976 | 0.719 | 0.927 | 0.810 | **0.858** |
| wolf-defender (0.3B) | 0.940 | 0.789 | 0.976 | 0.865 | **0.893** |
| hlyn judge (70M) | 0.835 | 0.829 | 0.848 | 0.326 | **0.710** |
| wolf-defender-small (0.1B) | 0.911 | 0.744 | 0.957 | 0.896 | **0.877** |
| protectai v2 (184M) | 0.656 | 0.000 | 0.912 | 0.826 | **0.599** |
| proventra mdeberta (280M) | 0.734 | 0.405 | 0.814 | 0.641 | **0.649** |
| deepset injection (184M) | 0.659 | 0.701 | 0.547 | 0.877 | **0.696** |
| fmops distilbert (67M) | 0.660 | 0.669 | 0.533 | 0.776 | **0.659** |
| meta prompt-guard (86M) | 0.555 | 0.667 | 0.484 | 0.671 | **0.594** |

## Latency (p50 ms / sample, batched inference)

| Model | rogue (5k) | JBB (200) | xTRam1 test (2k) | S-Labs test (2k) |
|---|---|---|---|---|
| bastion-prompt-protection (70M) | 4.1 | 0.9 | 4.1 | 0.8 |
| sentinel (qualifire, 395M) | 22.4 | 1.5 | 22.4 | 1.2 |
| wolf-defender (0.3B) | 8.5 | 0.7 | 8.5 | 0.7 |
| hlyn judge (70M) | 4.1 | 0.8 | 4.1 | 0.8 |
| wolf-defender-small (0.1B) | 4.1 | 0.7 | 4.0 | 0.7 |
| protectai v2 (184M) | 9.6 | 0.8 | 9.6 | 0.8 |
| proventra mdeberta (280M) | 9.6 | 0.9 | 9.6 | 0.8 |
| deepset injection (184M) | 9.6 | 0.9 | 9.6 | 0.8 |
| fmops distilbert (67M) | 3.3 | 0.3 | 3.2 | 0.2 |
| meta prompt-guard (86M) | 9.7 | 0.9 | 9.6 | 0.8 |

_Generated 2026-06-16 via `python -m scripts.run_leaderboard`._
