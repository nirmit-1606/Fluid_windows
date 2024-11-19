sample:		sample.cpp
		g++ -fopenmp -I  -o sample   sample.cpp  -lGL -lGLU -lglut  -lm


save:
		cp sample.cpp sample.save.cpp
