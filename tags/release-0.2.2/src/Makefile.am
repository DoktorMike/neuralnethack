MAINTAINERCLEANFILES = Makefile.in
INCLUDES		= -I$(srcdir) -I$(top_srcdir) 
bin_PROGRAMS	= neuralnethack testing
neuralnethack_SOURCES = main.cc \
				  Parser.cc Parser.hh \
				  CrossValidator.cc CrossValidator.hh \
				  Trainer.cc Trainer.hh \
				  GradientDescent.cc GradientDescent.hh \
				  QuasiNewton.cc QuasiNewton.hh \
				  Error.cc Error.hh \
				  SummedSquare.cc SummedSquare.hh \
				  Committee.cc Committee.hh \
				  Config.cc Config.hh \
				  NeuralNetHack.cc NeuralNetHack.hh \
				  MatrixTools.cc MatrixTools.hh
testing_SOURCES = testing.cc \
				  Parser.cc Parser.hh \
				  Supervisor.cc Supervisor.hh \
				  CrossValidator.cc CrossValidator.hh \
				  Trainer.cc Trainer.hh \
				  GradientDescent.cc GradientDescent.hh \
				  QuasiNewton.cc QuasiNewton.hh \
				  Error.cc Error.hh \
				  SummedSquare.cc SummedSquare.hh \
				  Committee.cc Committee.hh \
				  Config.cc Config.hh \
				  NeuralNetHack.cc NeuralNetHack.hh \
				  MatrixTools.cc MatrixTools.hh
LDADD			= ../datatools/libdatatools.a \
				  ../evaltools/libevaltools.a \
				  ../mlp/libmlp.a
