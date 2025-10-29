#pragma once

struct PreprocessParams {
    static constexpr int NUM_THREADS = 256;
    static constexpr int BM = 128;
    static constexpr int BN = 128;
    static constexpr int BK = 8;
    static constexpr int WM = 32;
    static constexpr int WN = 64;
    static constexpr int WNITER = 8;
    static constexpr int TN = 8;
    static constexpr int TM = 1;
    static constexpr int WARPSIZE = 32;

    static constexpr int WMITER = (WM * WN) / (WARPSIZE * TM * TN * WNITER);
    static constexpr int WSUBM = WM / WMITER;

    static constexpr int MAX_SPARSE_OFFSETS = BK / 2;
    static constexpr int ELEMENTS_PER_PATTERN = 1 + MAX_SPARSE_OFFSETS;

    static constexpr int denseListSize(int K) {
        return (K / BK) * WMITER * ELEMENTS_PER_PATTERN;
    }
};

