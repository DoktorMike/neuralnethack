#                                               -*- Autoconf -*-
# Process this file with autoconf to produce a configure script.

AC_PREREQ(2.59)
AC_INIT(NetHack, 0.1.0, michael@thep.lu.se, nethack)
AC_CONFIG_SRCDIR([src/main.cc])
#This is replaced with AM version since I am using automake. Should I have
#used only autoconf I should use AC version again.
#AC_CONFIG_HEADER(config.h) 
AM_CONFIG_HEADER(config.h)
AC_CONFIG_AUX_DIR(autotools)
AM_INIT_AUTOMAKE($PACKAGE_NAME,$PACKAGE_VERSION,$PACKAGE_BUGREPORT)


# Checks for programs.
AC_PROG_CXX

CXXFLAGS="$CXXFLAGS -Wall -pedantic" # -pg"

# Checks for libraries.
AC_PROG_RANLIB

# Checks for header files.

# Checks for typedefs, structures, and compiler characteristics.
AC_HEADER_STDBOOL
AC_C_CONST

# Checks for library functions.
AC_FUNC_ERROR_AT_LINE
AC_CHECK_FUNCS([pow sqrt])

AC_SUBST(CXXFLAGS)

AC_CONFIG_FILES([Makefile
		 src/Makefile
		 src/datatools/Makefile
		 src/evaltools/Makefile])
AC_OUTPUT
