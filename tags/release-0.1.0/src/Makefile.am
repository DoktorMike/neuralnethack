SUBDIRS = evaltools datatools
bin_PROGRAMS = nethack
nethack_SOURCES = \
	main.cc \
	Neuron.cc Neuron.hh \
	Sigmoid.cc Sigmoid.hh \
	TanHyp.cc TanHyp.hh \
	Linear.cc Linear.hh \
	Layer.cc Layer.hh \
	Mlp.cc Mlp.hh \
	Parser.cc Parser.hh \
	Supervisor.cc Supervisor.hh \
	Trainer.cc Trainer.hh \
	GradientDescent.cc GradientDescent.hh \
	QuasiNewton.cc QuasiNewton.hh \
	Error.cc Error.hh \
	SummedSquare.cc SummedSquare.hh \
	Committee.cc Committee.hh \
	Config.cc Config.hh \
	NetHack.cc NetHack.hh \
	MatrixTools.cc MatrixTools.hh
nethack_LDADD = datatools/libdatatools.a evaltools/libevaltools.a
#AM_CXXFLAGS = -Wall -pedantic
