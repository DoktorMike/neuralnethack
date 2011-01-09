#!/usr/bin/perl -w

# This script converts a neuralnethack weightvector, as given in the
# networks.xml file, to the matlab neural network weightvector. This makes
# plotting of hinton diagrams etc easier.
# $Id: nnh2matlab.pl 1602 2007-01-18 19:24:20Z michael $

use strict;

my @arch = @ARGV;
my $tmp = <STDIN>;
my @weights = split(' ', $tmp);

my %weights;

# Parse neural nethack format
for (my $i=1; $i<@arch; ++$i){
	my $nprev = $arch[$i-1];
	my $ncurr = $arch[$i];
	for (my $j=1; $j<=$ncurr; ++$j){
		for (my $k=1; $k<=$nprev + 1; ++$k){
			$weights{"$i $j $k"} = shift @weights;
		}
	}
}


@weights = (); #zero the weight vector.
# Print it in matlab format
for (my $i=1; $i<@arch; ++$i){
	my $nprev = $arch[$i-1];
	my $ncurr = $arch[$i];
	for (my $k=1; $k<=$nprev; ++$k){
		for (my $j=1; $j<=$ncurr; ++$j){
			print $weights{"$i $j $k"}, " ";
			push( @weights, $weights{"$i $j $k"} );
		}
	}
	for (my $j=1; $j<=$ncurr; ++$j){
		my $k = $nprev+1;
		print $weights{"$i $j $k"}, " ";
		push( @weights, $weights{"$i $j $k"} );
	}
	
}
print "\n";

# Generate Matlab code for printing the hinton diagrams
open my $fh, ">hinton.m" or die "Couln't open hinton.m: $!\n";
my $weightstring = '['.join(' ', @weights).']';
my $arch = '['.join(' ', @arch).']';
$arch =~ s/\[\d+\s/\[/;
print $fh <<EOF;

a = [zeros($arch[0], 1) ones($arch[0], 1)];
net = newff(a, $arch);
net = setx(net, $weightstring);
figure;
hintonwb(net.iw{1,1}, net.b{1,1});
print -depsc -tiff hinton1;
EOF
for (my $i=2; $i<@arch; ++$i){
	print $fh "figure;\n";
	print $fh "hintonwb(net.lw{$i,1}, net.b{$i,1});\n";
	print $fh "print -depsc -tiff hinton$i;\n"
}
close $fh;


