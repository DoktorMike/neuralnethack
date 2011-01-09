#!/usr/bin/perl -w

use strict;

my %trnTarget;
my %trnOutput;
my %tstTarget;
my %tstOutput;

my $cntr = 0;
while(my $line = <>){
	next if $line =~ /#/;
	if($line =~ />>/){
		$cntr++;
		next;
	}
	my @vals = split(' ', $line);
	$trnTarget{$vals[0]} = $vals[1] if $cntr == 1;
	$tstTarget{$vals[0]} = $vals[1] if $cntr == 2;
	$trnOutput{$vals[0]} = $vals[1] if $cntr == 3;
	$tstOutput{$vals[0]} = $vals[1] if $cntr == 4;
}

foreach my $key (keys %trnOutput){
	print "$trnTarget{$key}\t$trnOutput{$key}\n";
}
