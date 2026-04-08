#                                               -*- Autoconf -*-
# Process this file with autoconf to produce a configure script.

AC_PREREQ(2.59)
AC_INIT([NeuralNetHack], [0.9.5])
AC_CONFIG_SRCDIR([neuralnethack/NeuralNetHack.cc])
AM_CONFIG_HEADER(config.h)
AC_CONFIG_AUX_DIR(autotools)
AM_INIT_AUTOMAKE
#AM_INIT_AUTOMAKE($PACKAGE_NAME,$PACKAGE_VERSION,$PACKAGE_BUGREPORT)

# Added this init since automake complains
#LT_INIT

# Suggested add
AM_MAINTAINER_MODE([disable])

# Added by suggestion of libtool
AC_CONFIG_MACRO_DIRS([m4])

# Set the language
AC_LANG(C++)

# Checks for programs.
AC_PROG_CXX
AC_PROG_CXXCPP
AC_PROG_LIBTOOL
# Make libtool shut up
#LIBTOOL="$LIBTOOL --silent"

# Macro flags for DEBUG mode
CXXFLAGS="-std=c++17 -Wall -pedantic -O3 -march=native -ffast-math -funroll-loops -ftree-vectorize -fno-math-errno"
#CXXFLAGS="$CXXFLAGS -Wall -pedantic -g3 -gdwarf-2 -O0" # GDB mode
#CXXFLAGS="$CXXFLAGS -pg" # Benchmark mode
#CPPFLAGS="-D_GLIBCXX_CONCEPT_CHECKS -D_GLIBCXX_DEBUG_PEDANTIC"

# Macro flags for release mode
#CXXFLAGS="$CXXFLAGS -Wall -pedantic -O3"
CPPFLAGS="$CPPFLAGS -DNDEBUG"

#LDFLAGS="$LDFLAGS -static -all-static"
LDFLAGS="$LDFLAGS"

# Checks for libraries. Not needed if LIBTOOL is used.
# AC_PROG_RANLIB

# Checks for header files.
AC_CHECK_HEADERS([cctype cmath cassert cstdlib numeric utility vector string sstream fstream iomanip iostream istream ostream algorithm functional iterator])

# Checks for typedefs, structures, and compiler characteristics.
#AC_HEADER_STDBOOL
#AC_C_CONST
#AC_C_INLINE

# Checks for library functions.
#AC_FUNC_ERROR_AT_LINE
#AC_CHECK_FUNCS([pow sqrt])
#AC_CHECK_LIB(m, round)

# Check for BLAS/OpenBLAS
AC_ARG_WITH([blas],
  [AS_HELP_STRING([--with-blas], [use BLAS for matrix operations @<:@default=check@:>@])],
  [], [with_blas=check])

BLAS_LIBS=""
have_blas=no
AS_IF([test "x$with_blas" != xno],
  [AC_CHECK_HEADERS([cblas.h], [
    AC_CHECK_LIB([cblas], [cblas_ddot], [have_blas=yes; BLAS_LIBS="-lcblas"],
      [AC_CHECK_LIB([openblas], [cblas_ddot], [have_blas=yes; BLAS_LIBS="-lopenblas"],
        [AC_CHECK_LIB([blas], [cblas_ddot], [have_blas=yes; BLAS_LIBS="-lblas"],
          [AS_IF([test "x$with_blas" = xyes],
            [AC_MSG_ERROR([BLAS requested but not found])])])])])],
    [AS_IF([test "x$with_blas" = xyes],
      [AC_MSG_ERROR([BLAS requested but cblas.h not found])])])])

AS_IF([test "x$have_blas" = xyes],
  [AC_DEFINE([USE_BLAS], [1], [Define to 1 if BLAS is available])
   AC_MSG_NOTICE([BLAS support enabled])],
  [AC_MSG_NOTICE([BLAS support disabled])])

AC_SUBST([BLAS_LIBS])
AC_SUBST(CXXFLAGS)
AC_SUBST(CPPFLAGS)
AC_SUBST(LDFLAGS)

AC_CONFIG_FILES([Makefile
		 neuralnethack/Makefile
		 neuralnethack/datatools/Makefile
		 neuralnethack/evaltools/Makefile
		 neuralnethack/matrixtools/Makefile
		 neuralnethack/mlp/Makefile
		 neuralnethack/parser/Makefile
		 src/Makefile
		 test/Makefile])
AC_OUTPUT
