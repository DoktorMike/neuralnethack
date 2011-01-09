# $Id: Makefile.am 3344 2009-03-13 00:04:02Z michael $
MAINTAINERCLEANFILES = Makefile.in
INCLUDES= -I. -I.. -I$(top_srcdir)/src

noinst_LTLIBRARIES = libmlp.la

libmlp_la_SOURCES = MultiLayerPerceptron.cc \
					Trainer.cc \
					GradientDescent.cc \
					QuasiNewton.cc \
					Error.cc \
					SummedSquare.cc \
					CrossEntropy.cc \
					Mlp.cc \
					Layer.cc \
					SigmoidLayer.cc \
					TanHypLayer.cc \
					LinearLayer.cc \
					Weights.cc 

include_mlpdir = $(includedir)/neuralnethack/mlp

include_mlp_HEADERS = MultiLayerPerceptron.hh \
					  Trainer.hh \
					  GradientDescent.hh \
					  QuasiNewton.hh \
					  Error.hh \
					  SummedSquare.hh \
					  CrossEntropy.hh \
					  Mlp.hh \
					  Layer.hh \
					  SigmoidLayer.hh \
					  TanHypLayer.hh \
					  LinearLayer.hh \
					  Weights.hh
