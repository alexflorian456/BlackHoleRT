rmdir /s /q build
cmake -S . -B build 
cd build
cmake --build .