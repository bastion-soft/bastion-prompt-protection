## Leaderboard — AUC

| Model | rogue (5k) | xTRam1 test (2k) | S-Labs test (2k) | JBB (200) | German (9k) | **Avg** |
|---|---|---|---|---|---|---|
| bastion-v1.4 mdeberta (280M) | 0.986 | 0.998 | 0.996 | 0.986 | 0.899 | **0.973** |
| bastion-tiny v1.1 (70M) | 0.972 | 0.997 | 0.996 | 0.970 | 0.897 | **0.966** |
| wolf-defender (0.3B) | 0.988 | 0.996 | 0.986 | 0.847 | 0.966 | **0.957** |
| wolf-defender-small (0.1B) | 0.977 | 0.994 | 0.982 | 0.811 | 0.945 | **0.942** |
| sentinel (qualifire, 395M) | 0.997 | 0.991 | 0.955 | 0.894 | 0.718 | **0.911** |
| proventra mdeberta (280M) | 0.867 | 0.906 | 0.956 | 0.645 | 0.870 | **0.849** |
| piguard (deberta) | 0.839 | 0.912 | 0.902 | 0.644 | 0.830 | **0.825** |
| fmops distilbert (67M) | 0.789 | 0.514 | 0.907 | 0.591 | 0.601 | **0.681** |
| protectai v2 (184M) | 0.830 | 0.992 | 0.978 | 0.600 | 0.878 | **0.855** |
| deepset injection (184M) | 0.787 | 0.666 | 0.961 | 0.649 | 0.667 | **0.746** |
| hlyn judge (70M) | 0.980 | 0.995 | 0.891 | 0.934 | 0.880 | **0.936** |
| meta prompt-guard (86M) | 0.314 | 0.186 | 0.362 | 0.332 | 0.382 | **0.315** |

## Leaderboard — F1 @ threshold=0.5

| Model | rogue (5k) | xTRam1 test (2k) | S-Labs test (2k) | JBB (200) | German (9k) | **Avg** |
|---|---|---|---|---|---|---|
| bastion-v1.4 mdeberta (280M) | 0.916 | 0.941 | 0.955 | 0.960 | 0.608 | **0.876** |
| bastion-tiny v1.1 (70M) | 0.910 | 0.961 | 0.962 | 0.910 | 0.663 | **0.881** |
| wolf-defender (0.3B) | 0.940 | 0.976 | 0.865 | 0.789 | 0.879 | **0.890** |
| wolf-defender-small (0.1B) | 0.911 | 0.957 | 0.896 | 0.744 | 0.855 | **0.873** |
| sentinel (qualifire, 395M) | 0.976 | 0.927 | 0.810 | 0.719 | 0.646 | **0.815** |
| proventra mdeberta (280M) | 0.734 | 0.815 | 0.641 | 0.405 | 0.764 | **0.672** |
| piguard (deberta) | 0.670 | 0.712 | 0.793 | 0.600 | 0.698 | **0.695** |
| fmops distilbert (67M) | 0.660 | 0.533 | 0.776 | 0.669 | 0.571 | **0.642** |
| protectai v2 (184M) | 0.656 | 0.912 | 0.826 | 0.000 | 0.673 | **0.614** |
| deepset injection (184M) | 0.659 | 0.547 | 0.877 | 0.701 | 0.672 | **0.691** |
| hlyn judge (70M) | 0.835 | 0.848 | 0.326 | 0.829 | 0.426 | **0.653** |
| meta prompt-guard (86M) | 0.555 | 0.484 | 0.671 | 0.667 | 0.489 | **0.573** |

## Latency (p50 ms / sample, batched inference)

| Model | rogue (5k) | xTRam1 test (2k) | S-Labs test (2k) | JBB (200) | German (9k) |
|---|---|---|---|---|---|
| bastion-v1.4 mdeberta (280M) | 2.5 | 2.4 | 0.2 | 0.2 | 2.5 |
| bastion-tiny v1.1 (70M) | 2.5 | 2.4 | 0.2 | 0.2 | 2.5 |
| wolf-defender (0.3B) | 3.5 | 3.5 | 0.3 | 0.3 | 3.5 |
| wolf-defender-small (0.1B) | 1.7 | 1.7 | 0.2 | 0.2 | 1.7 |
| sentinel (qualifire, 395M) | 8.1 | 8.1 | 0.5 | 0.6 | 8.1 |
| proventra mdeberta (280M) | 5.4 | 5.4 | 0.4 | 0.4 | 5.5 |
| piguard (deberta) | 5.4 | 5.4 | 0.4 | 0.4 | 5.5 |
| fmops distilbert (67M) | 1.4 | 1.4 | 0.1 | 0.1 | 1.4 |
| protectai v2 (184M) | 5.4 | 5.4 | 0.4 | 0.4 | 5.4 |
| deepset injection (184M) | 5.4 | 5.4 | 0.4 | 0.4 | 5.5 |
| hlyn judge (70M) | 2.5 | 2.4 | 0.2 | 0.2 | 2.5 |
| meta prompt-guard (86M) | 5.5 | 5.4 | 0.4 | 0.4 | 5.5 |

_Generated 2026-06-13 via `python -m scripts.run_leaderboard`._
