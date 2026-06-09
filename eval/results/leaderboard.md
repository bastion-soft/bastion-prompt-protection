## Leaderboard — AUC

| Model | rogue (5k) | JBB (200) | xTRam1 test (2k) | S-Labs test (2k) | **Avg** |
|---|---|---|---|---|---|
| bastion multilingual (280M, commercial) | 0.982 | 0.991 | 0.999 | 0.993 | **0.991** |
| bastion-prompt-protection (70M) | 0.972 | 0.970 | 0.997 | 0.996 | **0.984** |
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
| bastion multilingual (280M, commercial) | 0.921 | 0.960 | 0.962 | 0.945 | **0.947** |
| bastion-prompt-protection (70M) | 0.910 | 0.910 | 0.961 | 0.962 | **0.936** |
| sentinel (qualifire, 395M) | 0.976 | 0.719 | 0.927 | 0.810 | **0.858** |
| wolf-defender (0.3B) | 0.940 | 0.789 | 0.976 | 0.865 | **0.893** |
| hlyn judge (70M) | 0.835 | 0.829 | 0.848 | 0.326 | **0.710** |
| wolf-defender-small (0.1B) | 0.911 | 0.744 | 0.957 | 0.896 | **0.877** |
| protectai v2 (184M) | 0.656 | 0.000 | 0.912 | 0.826 | **0.599** |
| proventra mdeberta (280M) | 0.734 | 0.405 | 0.815 | 0.641 | **0.649** |
| deepset injection (184M) | 0.659 | 0.701 | 0.547 | 0.877 | **0.696** |
| fmops distilbert (67M) | 0.660 | 0.669 | 0.533 | 0.776 | **0.659** |
| meta prompt-guard (86M) | 0.555 | 0.667 | 0.484 | 0.671 | **0.594** |

## Latency (p50 ms / sample, batched inference)

| Model | rogue (5k) | JBB (200) | xTRam1 test (2k) | S-Labs test (2k) |
|---|---|---|---|---|
| bastion multilingual (280M, commercial) | 49.8 | 3.1 | 52.1 | 2.9 |
| bastion-prompt-protection (70M) | 17.3 | 0.9 | 17.8 | 0.9 |
| sentinel (qualifire, 395M) | 128.9 | 8.1 | 130.2 | 6.3 |
| wolf-defender (0.3B) | 47.8 | 3.0 | 48.7 | 2.0 |
| hlyn judge (70M) | 20.6 | 0.9 | 20.7 | 0.9 |
| wolf-defender-small (0.1B) | 20.9 | 1.2 | 21.1 | 0.8 |
| protectai v2 (184M) | 51.0 | 2.7 | 52.8 | 2.5 |
| proventra mdeberta (280M) | 52.8 | 3.4 | 53.4 | 2.9 |
| deepset injection (184M) | 53.0 | 2.7 | 53.0 | 2.6 |
| fmops distilbert (67M) | 16.7 | 0.9 | 17.3 | 0.7 |
| meta prompt-guard (86M) | 53.4 | 3.4 | 53.1 | 2.9 |

_Generated 2026-06-09 via `python -m scripts.run_leaderboard`._
