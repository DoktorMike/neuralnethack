#!/usr/bin/perl -w

use strict;

my @files = glob('msres*');

foreach my $fname (@files){
	open my $fh, "<$fname" or die "Couldn't open $fname: $!\n";
	my $opt_alpha = 0;
	my $opt_trnauc = 0;
	my $opt_tstauc = 0;
	while (my $line = <$fh>){
		next if $line =~ /#/;
		my ($alpha, $trnauc, $tstauc) = split(' ', $line);
		if($tstauc > $opt_tstauc){
			$opt_alpha = $alpha;
			$opt_trnauc = $trnauc;
			$opt_tstauc = $tstauc;
		}
	}
	close $fh;
	printf("%17s: %.3f %.4f %.4f\n", $fname, $opt_alpha, $opt_trnauc, $opt_tstauc);
}
