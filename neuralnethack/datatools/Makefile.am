# $Id: Makefile.am 1592 2006-11-23 10:40:03Z michael $
MAINTAINERCLEANFILES	= Makefile.in
#INCLUDES				= -I. -I..
noinst_LTLIBRARIES		= libdatatools.la

libdatatools_la_SOURCES	= DataTools.cc \
						  Normaliser.cc \
						  Sampler.cc \
						  HoldOutSampler.cc \
						  DummySampler.cc \
						  BootstrapSampler.cc \
						  CrossSplitSampler.cc \
						  DataManager.cc \
						  CoreDataSet.cc \
						  DataSet.cc \
						  Pattern.cc 

include_datatoolsdir = $(includedir)/neuralnethack/datatools

include_datatools_HEADERS = DataTools.hh \
							Normaliser.hh \
							Sampler.hh \
							HoldOutSampler.hh \
							DummySampler.hh \
							BootstrapSampler.hh \
							CrossSplitSampler.hh \
							DataManager.hh \
							CoreDataSet.hh \
							DataSet.hh \
							Pattern.hh
