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
include flashAtten/CMakeFiles/FA_CPU.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include flashAtten/CMakeFiles/FA_CPU.dir/compiler_depend.make

# Include the progress variables for this target.
include flashAtten/CMakeFiles/FA_CPU.dir/progress.make

# Include the compile flags for this target's objects.
include flashAtten/CMakeFiles/FA_CPU.dir/flags.make

flashAtten/CMakeFiles/FA_CPU.dir/FA.cpp.o: flashAtten/CMakeFiles/FA_CPU.dir/flags.make
flashAtten/CMakeFiles/FA_CPU.dir/FA.cpp.o: ../flashAtten/FA.cpp
flashAtten/CMakeFiles/FA_CPU.dir/FA.cpp.o: flashAtten/CMakeFiles/FA_CPU.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/cwwu/Desktop/Cuda/cuda_code/my_samples/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object flashAtten/CMakeFiles/FA_CPU.dir/FA.cpp.o"
	cd /home/cwwu/Desktop/Cuda/cuda_code/my_samples/build/flashAtten && /usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT flashAtten/CMakeFiles/FA_CPU.dir/FA.cpp.o -MF CMakeFiles/FA_CPU.dir/FA.cpp.o.d -o CMakeFiles/FA_CPU.dir/FA.cpp.o -c /home/cwwu/Desktop/Cuda/cuda_code/my_samples/flashAtten/FA.cpp

flashAtten/CMakeFiles/FA_CPU.dir/FA.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/FA_CPU.dir/FA.cpp.i"
	cd /home/cwwu/Desktop/Cuda/cuda_code/my_samples/build/flashAtten && /usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/cwwu/Desktop/Cuda/cuda_code/my_samples/flashAtten/FA.cpp > CMakeFiles/FA_CPU.dir/FA.cpp.i

flashAtten/CMakeFiles/FA_CPU.dir/FA.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/FA_CPU.dir/FA.cpp.s"
	cd /home/cwwu/Desktop/Cuda/cuda_code/my_samples/build/flashAtten && /usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/cwwu/Desktop/Cuda/cuda_code/my_samples/flashAtten/FA.cpp -o CMakeFiles/FA_CPU.dir/FA.cpp.s

# Object files for target FA_CPU
FA_CPU_OBJECTS = \
"CMakeFiles/FA_CPU.dir/FA.cpp.o"

# External object files for target FA_CPU
FA_CPU_EXTERNAL_OBJECTS =

flashAtten/FA_CPU: flashAtten/CMakeFiles/FA_CPU.dir/FA.cpp.o
flashAtten/FA_CPU: flashAtten/CMakeFiles/FA_CPU.dir/build.make
flashAtten/FA_CPU: flashAtten/CMakeFiles/FA_CPU.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/cwwu/Desktop/Cuda/cuda_code/my_samples/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable FA_CPU"
	cd /home/cwwu/Desktop/Cuda/cuda_code/my_samples/build/flashAtten && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/FA_CPU.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
flashAtten/CMakeFiles/FA_CPU.dir/build: flashAtten/FA_CPU
.PHONY : flashAtten/CMakeFiles/FA_CPU.dir/build

flashAtten/CMakeFiles/FA_CPU.dir/clean:
	cd /home/cwwu/Desktop/Cuda/cuda_code/my_samples/build/flashAtten && $(CMAKE_COMMAND) -P CMakeFiles/FA_CPU.dir/cmake_clean.cmake
.PHONY : flashAtten/CMakeFiles/FA_CPU.dir/clean

flashAtten/CMakeFiles/FA_CPU.dir/depend:
	cd /home/cwwu/Desktop/Cuda/cuda_code/my_samples/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/cwwu/Desktop/Cuda/cuda_code/my_samples /home/cwwu/Desktop/Cuda/cuda_code/my_samples/flashAtten /home/cwwu/Desktop/Cuda/cuda_code/my_samples/build /home/cwwu/Desktop/Cuda/cuda_code/my_samples/build/flashAtten /home/cwwu/Desktop/Cuda/cuda_code/my_samples/build/flashAtten/CMakeFiles/FA_CPU.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : flashAtten/CMakeFiles/FA_CPU.dir/depend
