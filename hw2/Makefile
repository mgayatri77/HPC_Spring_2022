progs = MMult1 val_test01_solved val_test02_solved omp_solved2 omp_solved3 omp_solved4 omp_solved5 omp_solved6 jacobi2D-omp gs2D-omp
all: $(progs)

MMult1: MMult1.cpp
	g++ -fopenmp -O3 -march=native -o MMult1 MMult1.cpp

val_test01_solved: val_test01_solved.cpp
	g++ val_test01_solved.cpp -o val_test01_solved

val_test02_solved: val_test02_solved.cpp
	g++ val_test02_solved.cpp -o val_test02_solved

omp_solved2: omp_solved2.c
	gcc -fopenmp -o omp_solved2 omp_solved2.c

omp_solved3: omp_solved3.c
	gcc -fopenmp -o omp_solved3 omp_solved3.c

omp_solved4: omp_solved4.c
	gcc -fopenmp -o omp_solved4 omp_solved4.c

omp_solved5: omp_solved5.c
	gcc -fopenmp -o omp_solved5 omp_solved5.c

omp_solved_6: omp_solved6.c
	gcc -fopenmp -o omp_solved6 omp_solved6.c

jacobi2D-omp: jacobi2D-omp.cpp
	g++ -fopenmp -march=native -o jacobi2D-omp jacobi2D-omp.cpp

gs2D-omp: gs2D-omp.cpp
	g++ -fopenmp -march=native -o gs2D-omp gs2D-omp.cpp

clean: 
	rm -f *.o
	rm -f $(progs)