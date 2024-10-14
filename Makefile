# Define variables
NVCC = nvcc
SRC_DIR = fastlap_cuda
BUILD_DIR = lib
NVCC_FLAGS = -lib -Xcompiler -fPIC -O0# -shared

REMOVE_FILES = rm -f

# Get all .cu files in the source directory
SRCS = $(wildcard $(SRC_DIR)/*.cu)

# Create a list of static lib output files by replacing src/ with build/fastlap_cuda/ and changing the extension
OUT_FILES = $(BUILD_DIR)/fastlap.a
# $(patsubst $(SRC_DIR)/%.cu, $(BUILD_DIR)/%.a, $(SRCS))

# Default target
all: $(OUT_FILES)

# Rule to compile each .cu file into a static lib file
$(BUILD_DIR)/%.a: $(SRC_DIR)/%.cu | $(BUILD_DIR)
	$(NVCC) $(NVCC_FLAGS) -o $@ $(SRCS)

# Ensure the build directory exists
$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

# Clean up build files
clean:
	$(REMOVE_FILES) $(BUILD_DIR)/*.a