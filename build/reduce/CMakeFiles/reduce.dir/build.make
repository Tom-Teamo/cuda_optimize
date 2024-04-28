# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.22

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/cwwu/Desktop/Cuda/cuda_code/my_samples

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/cwwu/Desktop/Cuda/cuda_code/my_samples/build

# Include any dependencies generated for this target.
include reduce/CMakeFiles/reduce.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include reduce/CMakeFiles/reduce.dir/compiler_depend.make

# Include the progress variables for this target.
include reduce/CMakeFiles/reduce.dir/progress.make

# Include the compile flags for this target's objects.
include reduce/CMakeFiles/reduce.dir/flags.make

reduce/CMakeFiles/reduce.dir/reduction.cu.o: reduce/CMakeFiles/reduce.dir/flags.make
reduce/CMakeFiles/reduce.dir/reduction.cu.o: ../reduce/reduction.cu
reduce/CMakeFiles/reduce.dir/reduction.cu.o: reduce/CMakeFiles/reduce.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/cwwu/Desktop/Cuda/cuda_code/my_samples/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CUDA object reduce/CMakeFiles/reduce.dir/reduction.cu.o"
	cd /home/cwwu/Desktop/Cuda/cuda_code/my_samples/build/reduce && /usr/local/cuda/bin/nvcc -forward-unknown-to-host-compiler $(CUDA_DEFINES) $(CUDA_INCLUDES) $(CUDA_FLAGS) -MD -MT reduce/CMakeFiles/reduce.dir/reduction.cu.o -MF CMakeFiles/reduce.dir/reduction.cu.o.d -x cu -c /home/cwwu/Desktop/Cuda/cuda_code/my_samples/reduce/reduction.cu -o CMakeFiles/reduce.dir/reduction.cu.o

reduce/CMakeFiles/reduce.dir/reduction.cu.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CUDA source to CMakeFiles/reduce.dir/reduction.cu.i"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_PREPROCESSED_SOURCE

reduce/CMakeFiles/reduce.dir/reduction.cu.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CUDA source to assembly CMakeFiles/reduce.dir/reduction.cu.s"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_ASSEMBLY_SOURCE

# Object files for target reduce
reduce_OBJECTS = \
"CMakeFiles/reduce.dir/reduction.cu.o"

# External object files for target reduce
reduce_EXTERNAL_OBJECTS =

reduce/reduce: reduce/CMakeFiles/reduce.dir/reduction.cu.o
reduce/reduce: reduce/CMakeFiles/reduce.dir/build.make
reduce/reduce: reduce/CMakeFiles/reduce.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/cwwu/Desktop/Cuda/cuda_code/my_samples/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CUDA executable reduce"
	cd /home/cwwu/Desktop/Cuda/cuda_code/my_samples/build/reduce && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/reduce.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
reduce/CMakeFiles/reduce.dir/build: reduce/reduce
.PHONY : reduce/CMakeFiles/reduce.dir/build

reduce/CMakeFiles/reduce.dir/clean:
	cd /home/cwwu/Desktop/Cuda/cuda_code/my_samples/build/reduce && $(CMAKE_COMMAND) -P CMakeFiles/reduce.dir/cmake_clean.cmake
.PHONY : reduce/CMakeFiles/reduce.dir/clean

reduce/CMakeFiles/reduce.dir/depend:
	cd /home/cwwu/Desktop/Cuda/cuda_code/my_samples/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/cwwu/Desktop/Cuda/cuda_code/my_samples /home/cwwu/Desktop/Cuda/cuda_code/my_samples/reduce /home/cwwu/Desktop/Cuda/cuda_code/my_samples/build /home/cwwu/Desktop/Cuda/cuda_code/my_samples/build/reduce /home/cwwu/Desktop/Cuda/cuda_code/my_samples/build/reduce/CMakeFiles/reduce.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : reduce/CMakeFiles/reduce.dir/depend
