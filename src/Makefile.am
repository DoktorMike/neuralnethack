# $Id: Makefile.am 1606 2007-01-25 01:02:50Z michael $

MAINTAINERCLEANFILES = Makefile.in
INCLUDES = -I$(srcdir) -I$(top_srcdir) -I$(top_srcdir)/neuralnethack -I/home/whitman/michael/include

bin_PROGRAMS = neuralnethack ann modelselector saliency

neuralnethack_SOURCES = neuralnethack.cc
neuralnethack_LDADD	= $(top_srcdir)/neuralnethack/libneuralnethack.la

ann_SOURCES = ann.cc
ann_LDADD = $(top_srcdir)/neuralnethack/libneuralnethack.la

modelselector_SOURCES = modelselector.cc
modelselector_LDADD = $(top_srcdir)/neuralnethack/libneuralnethack.la

saliency_SOURCES = saliency.cc
saliency_LDADD = $(top_srcdir)/neuralnethack/libneuralnethack.la
