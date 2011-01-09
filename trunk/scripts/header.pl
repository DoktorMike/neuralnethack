#!/usr/bin/perl -w

# $Id: header.pl 1546 2006-04-18 08:38:01Z michael $ 

use strict;
use File::Find;

sub process_file; 

my @gplhead = <STDIN>;

@ARGV = qw(.) unless @ARGV;
find \&process_file, @ARGV;

sub process_file {
	return unless (-f $_ and $_ =~ /\.cc$|\.hh$/ );
	print $File::Find::name, "\n";
	open(my $fh1, "<$_") or die "Couldn't open file: $File::Find::name: $!";
	open(my $fh2, ">new$_") or die "Couldn't open file: new$_: $!";
	my @tmp = <$fh1>;
	my $instring = join('', @tmp);
	if($instring =~ /\$Id.+\$/){
		#print "ENTERING ID\n";
		my $gplstring = join('', @gplhead);
		$instring =~ s/.+(\$Id.+\$).+/\/\*$1\*\/\n\n\/\*\n$gplstring\*\/\n/g;
		print $fh2 $instring;
	}else{
		#print "ENTERING NON ID\n";
		print $fh2 "/*\$Id\$*/\n\n/*\n"; 
		print $fh2 @gplhead;
		print $fh2 "*/\n\n"; 
		print $fh2 $instring;
	}
	close($fh1);
	close($fh2);
	rename("new$_", "$_");
	unlink("new$_");
	#sleep 100;
}

