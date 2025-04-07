# MMMResearch
Weekly Findings:
After running the autotuner and then manually testing the top 7-8 results, there
are two configs that tie each other on duration via the NVIDIA profiler

Config One (warptiling_one):
  const uint K10_NUM_THREADS = 128;
  const uint K10_BN = 64;
  const uint K10_BM = 128;
  const uint K10_BK = 16;
  const uint K10_WN = 32;
  const uint K10_WM = 64;
  const uint K10_WNITER = 2;
  const uint K10_TN = 4;
  const uint K10_TM = 4;

Config Two (warptiling_two):
  const uint K10_NUM_THREADS = 128;
  const uint K10_BN = 64;
  const uint K10_BM = 128;
  const uint K10_BK = 8;
  const uint K10_WN = 32;
  const uint K10_WM = 64;
  const uint K10_WNITER = 2;
  const uint K10_TN = 4;
  const uint K10_TM = 4;


