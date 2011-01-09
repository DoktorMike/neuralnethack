#                                               -*- Autoconf -*-
# Process this file with autoconf to produce a configure script.

AC_PREREQ(2.59)
AC_INIT(NeuralNetHack, 0.2.4, michael@thep.lu.se, neuralnethack)
AC_CONFIG_SRCDIR([src/main.cc])
#This is replaced with AM version since I am using automake. Should I have
#used only autoconf I should use AC version again.
#AC_CONFIG_HEADER(config.h) 
AM_CONFIG_HEADER(config.h)
AC_CONFIG_AUX_DIR(autotools)
AM_INIT_AUTOMAKE($PACKAGE_NAME,$PACKAGE_VERSION,$PACKAGE_BUGREPORT)


# Checks for programs.
AC_PROG_CXX

#-gdwarf-2 is the preferred way of using gdb with C++
#CXXFLAGS="$CXXFLAGS -Wall -pedantic" # -pg "
#CXXFLAGS="$CXXFLAGS -Wall -pedantic -O3 -DNDEBUG" # -pg "
CXXFLAGS="$CXXFLAGS -Wall -pedantic -g3 -gdwarf-2 -D_GLIBCXX_CONCEPT_CHECKS -D_GLIBCXX_DEBUG_PEDANTIC"
LDFLAGS="$LDFLAGS -static"

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
AC_SUBST(LDFLAGS)

AC_CONFIG_FILES([Makefile
		 src/Makefile
		 datatools/Makefile
		 evaltools/Makefile
		 matrixtools/Makefile
		 src/mlp/Makefile])
AC_OUTPUT
