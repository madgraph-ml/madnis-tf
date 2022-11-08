lhapdf_vectorized.so: lhapdf_vectorized.cpp
	g++ lhapdf_vectorized.cpp -o lhapdf_vectorized.so -fPIC -std=c++11 --shared `lhapdf-config --cflags --ldflags`

.PHONY: clean
clean:
	rm -f lhapdf_vectorized.so
