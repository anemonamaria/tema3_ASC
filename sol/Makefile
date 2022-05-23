COMPILER=nvcc
LIBS=-lm -O0

all: solution

solution: helper.o solution.cu
	$(COMPILER)  solution.cu helper.o -o gpu_sol $(LIBS)

helper.o:
	nvcc -g helper.cpp -c -O0

clean:
	rm -f gpu_sol *.o