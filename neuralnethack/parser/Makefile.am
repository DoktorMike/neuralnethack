# $Id: Makefile.am 1555 2006-05-02 14:55:02Z michael $

MAINTAINERCLEANFILES = Makefile.in
INCLUDES = -I$(srcdir) -I../ -I$(top_srcdir) 

noinst_LTLIBRARIES = libparser.la
libparser_la_SOURCES = Parser.cc \
					   NetworkParser.cc

include_parserdir = $(includedir)/neuralnethack/parser

include_parser_HEADERS = Parser.hh \
						 NetworkParser.hh
