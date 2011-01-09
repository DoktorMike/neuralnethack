# $Id: Makefile.am 1615 2007-04-17 11:56:40Z michael $

MAINTAINERCLEANFILES = Makefile.in
SUBDIRS	= mlp datatools evaltools matrixtools parser .
INCLUDES = -I$(srcdir) -I$(top_srcdir) 

lib_LTLIBRARIES = libneuralnethack.la

pkginclude_HEADERS = Factory.hh \
					 PrintUtils.hh \
					 ModelSelector.hh \
					 ModelEstimator.hh \
					 Ensemble.hh \
					 EnsembleBuilder.hh \
					 Config.hh \
					 OddsRatio.hh \
					 Saliency.hh \
					 NeuralNetHack.hh


libneuralnethack_la_SOURCES = Factory.cc \
							  PrintUtils.cc \
							  ModelSelector.cc \
							  ModelEstimator.cc \
							  Ensemble.cc \
							  EnsembleBuilder.cc \
							  Config.cc \
							  OddsRatio.cc \
							  Saliency.cc \
							  NeuralNetHack.cc 

libneuralnethack_la_LIBADD = datatools/libdatatools.la \
							 evaltools/libevaltools.la \
							 matrixtools/libmatrixtools.la \
							 mlp/libmlp.la \
							 parser/libparser.la

