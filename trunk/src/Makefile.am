# $Id: Makefile.am 1699 2008-02-02 17:20:39Z michael $

MAINTAINERCLEANFILES = Makefile.in
INCLUDES = -I$(srcdir) -I$(top_srcdir) -I$(top_srcdir)/neuralnethack -I/home/whitman/michael/include

bin_PROGRAMS = neuralnethack ann modelselector featureselector featureselector2 saliency auc

neuralnethack_SOURCES = neuralnethack.cc
neuralnethack_LDADD	= $(top_srcdir)/neuralnethack/libneuralnethack.la

ann_SOURCES = ann.cc
ann_LDADD = $(top_srcdir)/neuralnethack/libneuralnethack.la

modelselector_SOURCES = modelselector.cc
modelselector_LDADD = $(top_srcdir)/neuralnethack/libneuralnethack.la

featureselector_SOURCES = featureselector.cc
featureselector_LDADD = $(top_srcdir)/neuralnethack/libneuralnethack.la

featureselector2_SOURCES = featureselector2.cc
featureselector2_LDADD = $(top_srcdir)/neuralnethack/libneuralnethack.la

saliency_SOURCES = saliency.cc
saliency_LDADD = $(top_srcdir)/neuralnethack/libneuralnethack.la

auc_SOURCES = auc.cc
auc_LDADD = $(top_srcdir)/neuralnethack/libneuralnethack.la
