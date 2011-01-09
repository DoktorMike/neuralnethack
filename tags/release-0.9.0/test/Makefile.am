# $Id: Makefile.am 1559 2006-05-10 22:41:10Z michael $

MAINTAINERCLEANFILES = Makefile.in
INCLUDES = -I$(srcdir) -I$(top_srcdir) -I$(top_srcdir)/neuralnethack

TESTS = testing testSampler testNormaliser testGof testMatrix \
		testSaliency testParser
check_PROGRAMS = $(TESTS)

testing_SOURCES = testing.cc
testSampler_SOURCES = testSampler.cc
testNormaliser_SOURCES = testNormaliser.cc
testGof_SOURCES = testGof.cc
testMatrix_SOURCES = testMatrix.cc
testSaliency_SOURCES = testSaliency.cc
testParser_SOURCES = testParser.cc

LDADD = $(top_srcdir)/neuralnethack/libneuralnethack.la
