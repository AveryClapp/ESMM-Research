# ESMM CUDA Project Makefile
# Quick compilation with development/production modes

# Compiler and flags
NVCC := nvcc
CCACHE := $(shell which ccache 2>/dev/null)
ifdef CCACHE
    NVCC := $(CCACHE) $(NVCC)
endif
ARCH := -arch=sm_80

# Libraries
LIBS := -lcublas -lcusparse

# Development flags (fast compilation)
DEV_FLAGS := -O1 $(ARCH)
# Production flags (optimized, with profiling info)
PROD_FLAGS := -O3 $(ARCH) -lineinfo --use_fast_math

# Default to development mode
NVCC_FLAGS ?= $(DEV_FLAGS)

# Source files
DRIVER := driver.cu
TARGET := exec_dev
PROD_TARGET := exec_prod

# Header dependencies (if these change, recompile)
HEADERS := $(wildcard include/*.cuh) \
           $(wildcard src/kernels/*.cu) \
           $(wildcard src/kernels/unrolled_kernels/*.cu) \
           $(wildcard src/preprocessors/*.cu) \
           $(wildcard old_kernels/*.cu)

# Default target: fast development build
.PHONY: all
all: dev

# Development build (fast compilation, -O1)
.PHONY: dev
dev: NVCC_FLAGS = $(DEV_FLAGS)
dev: $(TARGET)

$(TARGET): $(DRIVER) $(HEADERS)
	@echo "Building development version (fast compile, -O1)..."
	$(NVCC) $(NVCC_FLAGS) $(DRIVER) -o $(TARGET) $(LIBS)
	@echo "Build complete: ./$(TARGET)"

# Production build (optimized, -O3, lineinfo)
.PHONY: release
release: NVCC_FLAGS = $(PROD_FLAGS)
release: $(PROD_TARGET)

$(PROD_TARGET): $(DRIVER) $(HEADERS)
	@echo "Building release version (optimized, -O3)..."
	$(NVCC) $(NVCC_FLAGS) $(DRIVER) -o $(PROD_TARGET) $(LIBS)
	@echo "Build complete: ./$(PROD_TARGET)"

# Quick build for kernel testing (even faster, -O0)
.PHONY: quick
quick: NVCC_FLAGS = -O0 $(ARCH)
quick: $(TARGET)
	@echo "Ultra-fast build complete (no optimization)"

# Profile build (for ncu/nsys)
.PHONY: profile
profile: NVCC_FLAGS = $(PROD_FLAGS) -lineinfo
profile: $(TARGET)
	@echo "Profile build complete (use with ncu/nsys)"

# NCU profiling targets
PROFILE_TARGET := exec_profile
NCU_FLAGS := --set full --target-processes all
# Default to using sudo (disable with NO_SUDO=1)
NCU_CMD := sudo LD_LIBRARY_PATH=/usr/local/lib64:$LD_LIBRARY_PATH /usr/local/cuda-12.1/bin/ncu -f
ifdef NO_SUDO
    NCU_CMD := ncu
endif

# Build specifically for NCU profiling
.PHONY: build-ncu
build-ncu: NVCC_FLAGS = $(PROD_FLAGS) -lineinfo
build-ncu: $(PROFILE_TARGET)

$(PROFILE_TARGET): $(DRIVER) $(HEADERS)
	@echo "Building for NCU profiling (optimized with lineinfo)..."
	$(NVCC) $(NVCC_FLAGS) $(DRIVER) -o $(PROFILE_TARGET) $(LIBS)
	@echo "Build complete: ./$(PROFILE_TARGET)"

# Run NCU with full metrics (KERNEL can be: 17, "17,18", "17-20", or "all")
.PHONY: ncu-profile
ncu-profile: build-ncu
	@if [ -z "$(KERNEL)" ]; then \
		echo "Error: Please specify KERNEL=<selection>"; \
		echo "Examples:"; \
		echo "  make ncu-profile KERNEL=17          # Single kernel"; \
		echo "  make ncu-profile KERNEL=17,18,19    # Multiple kernels"; \
		echo "  make ncu-profile KERNEL=17-20       # Range"; \
		echo "  make ncu-profile KERNEL=all         # All kernels"; \
		exit 1; \
	fi
	@KERNEL_NUM=$(KERNEL); \
	OUTFILE=profile.ncu-rep; \
	echo "Profiling kernel(s) $${KERNEL_NUM}..."; \
	echo "Output: $${OUTFILE}"; \
	echo "Running: $(NCU_CMD) $(NCU_FLAGS) -o $${OUTFILE} ./$(PROFILE_TARGET) $${KERNEL_NUM}"; \
	$(NCU_CMD) $(NCU_FLAGS) -o $${OUTFILE} ./$(PROFILE_TARGET) $${KERNEL_NUM}

# Run NCU on already-built binary (lighter weight - assumes build exists)
.PHONY: ncu-run
ncu-run:
	@if [ ! -f $(PROFILE_TARGET) ]; then \
		echo "Error: $(PROFILE_TARGET) not found. Run 'make build-ncu' first."; \
		exit 1; \
	fi
	@if [ -z "$(KERNEL)" ]; then \
		echo "Error: Please specify KERNEL=<selection>"; \
		echo "Examples:"; \
		echo "  make ncu-run KERNEL=17			# Single kernel"; \
		echo "  make ncu-run KERNEL=17,18,19		# Multiple kernels"; \
		echo "  make ncu-run KERNEL=17-20	        # Range"; \
		exit 1; \
	fi
	@KERNEL_NUM=$(KERNEL); \
	OUTFILE=profile.ncu-rep; \
	echo "Profiling kernel(s) $${KERNEL_NUM}..."; \
	echo "Output: $${OUTFILE}"; \
	$(NCU_CMD) $(NCU_FLAGS) -o $${OUTFILE} ./$(PROFILE_TARGET) $${KERNEL_NUM}

# Quick NCU profile (fewer metrics, faster) - uses default set instead of full
.PHONY: ncu-quick
ncu-quick: build-ncu
	@if [ -z "$(KERNEL)" ]; then \
		echo "Error: Please specify KERNEL=<selection>"; \
		echo "Examples:"; \
		echo "  make ncu-quick KERNEL=17          # Single kernel"; \
		echo "  make ncu-quick KERNEL=17,18       # Multiple kernels"; \
		exit 1; \
	fi
	@KERNEL_NUM=$(KERNEL); \
	OUTFILE=profile.ncu-rep; \
	echo "Quick profiling kernel(s) $${KERNEL_NUM}..."; \
	echo "Output: $${OUTFILE}"; \
	$(NCU_CMD) --target-processes all -o $${OUTFILE} ./$(PROFILE_TARGET) $${KERNEL_NUM}

# Clean build artifacts
.PHONY: clean
clean:
	rm -f $(TARGET) $(PROD_TARGET) $(TEST_TARGET) $(PROFILE_TARGET) *.o profile.ncu-rep

# Display build modes
.PHONY: help
help:
	@echo "ESMM CUDA Build System"
	@echo ""
	@echo "Build Targets:"
	@echo "  make / make dev    - Fast development build (-O1, ~68s, all kernels)"
	@echo "  make test_kernel   - FASTEST - single kernel test (~3s, edit test_single_kernel.cu)"
	@echo "  make release       - Optimized production build (-O3, ~71s)"
	@echo "  make quick         - Ultra-fast build for testing (-O0, ~68s)"
	@echo "  make profile       - Build for profiling (ncu/nsys)"
	@echo "  make build-ncu     - Build for NCU profiling (same as profile)"
	@echo "  make clean         - Remove build artifacts"
	@echo ""
	@echo "NCU Profiling Targets:"
	@echo "  make ncu-profile KERNEL=<sel>  - Build + profile with --set full"
	@echo "  make ncu-run KERNEL=<sel>      - Profile (no rebuild) with --set full"
	@echo "  make ncu-quick KERNEL=<sel>    - Build + quick profile (default metrics)"
	@echo ""
	@echo "Usage examples:"
	@echo "  make test_kernel && ./test_kernel 100        # FASTEST for development!"
	@echo "  make dev && ./driver 17 100                  # Test specific kernel"
	@echo "  make ncu-profile KERNEL=17                   # Profile single kernel (with sudo)"
	@echo "  make ncu-profile KERNEL=17,18,19             # Profile multiple kernels"
	@echo "  make ncu-profile KERNEL=17-20                # Profile kernel range"
	@echo "  make ncu-profile KERNEL=17 NO_SUDO=1         # Without sudo (if not needed)"
	@echo "  make ncu-run KERNEL=17                       # Re-profile without rebuild"
	@echo "  make ncu-quick KERNEL=17                     # Quick profile"
	@echo ""
	@echo "NCU profiles saved to: ./profile.ncu-rep"
	@echo ""
	@echo "Current settings:"
	@echo "  ARCH: $(ARCH)"
	@echo "  DEV_FLAGS: $(DEV_FLAGS)"
	@echo "  PROD_FLAGS: $(PROD_FLAGS)"
	@echo "  NCU_FLAGS: $(NCU_FLAGS)"
	@echo "  NCU_CMD: $(NCU_CMD)"

.PHONY: info
info: help
