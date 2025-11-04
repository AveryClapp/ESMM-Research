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
    static constexpr int NUM_WARP_ROWS = (BM + WM - 1) / WM;

    // Bitmask encoding: 1 byte per pattern (8 bits for BK=8)
    // Store as uint8_t, pack 4 masks per int for alignment
    static constexpr int MASKS_PER_INT = 4;

    static constexpr int denseListSize(int K) {
        int numMasks = (K / BK) * NUM_WARP_ROWS * WMITER;
        // Round up to pack 4 masks per int
        return (numMasks + MASKS_PER_INT - 1) / MASKS_PER_INT;
    }
};

