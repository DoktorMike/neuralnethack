# $Id: Makefile.am 1222 2005-03-17 17:18:35Z michael $
MAINTAINERCLEANFILES = Makefile.in
SUBDIRS	= mlp .
#INCLUDES		= -I$(srcdir) -I$(top_srcdir) 
bin_PROGRAMS	= neuralnethack testing
neuralnethack_SOURCES = main.cc \
						Factory.cc Factory.hh \
						PrintUtils.cc PrintUtils.hh \
						Parser.cc Parser.hh \
						ModelEstimator.cc ModelEstimator.hh \
						CrossValidator.cc CrossValidator.hh \
						Bootstrapper.cc Bootstrapper.hh \
						Committee.cc Committee.hh \
						EnsembleBuilder.cc EnsembleBuilder.hh \
						CrossSplitter.cc CrossSplitter.hh \
						Bagger.cc Bagger.hh \
						Config.cc Config.hh \
						ErrorMeasures.cc ErrorMeasures.hh \
						NeuralNetHack.cc NeuralNetHack.hh

testing_SOURCES = testing.cc \
				  Factory.cc Factory.hh \
				  PrintUtils.cc PrintUtils.hh \
				  Parser.cc Parser.hh \
				  ModelEstimator.cc ModelEstimator.hh \
				  CrossValidator.cc CrossValidator.hh \
				  Bootstrapper.cc Bootstrapper.hh \
				  Committee.cc Committee.hh \
				  EnsembleBuilder.cc EnsembleBuilder.hh \
				  CrossSplitter.cc CrossSplitter.hh \
				  Bagger.cc Bagger.hh \
				  Config.cc Config.hh \
				  ErrorMeasures.cc ErrorMeasures.hh \
				  NeuralNetHack.cc NeuralNetHack.hh

LDADD			= ../datatools/libdatatools.a \
				  ../evaltools/libevaltools.a \
				  ../matrixtools/libmatrixtools.a \
				  mlp/libmlp.a
