#!/usr/bin/perl -w

###############################################################################
# mapsaliencies - a program for mapping saliencies from a neuralnethack run
#                 onto an interpretable format
# $Id: mapsaliencies.pl 1683 2007-10-08 16:39:39Z michael $
###############################################################################

use strict;
use Data::Dumper;

# Getting stuff from config file
sub get_from_config($$); #Get the n:th attribute of the supplied identifier from config file.
sub get_cols(); #Get the columns to use from the config file.
# Getting stuff from data file
sub get_targets(); #Get the targets from the datafile.
sub get_ids(); #Get the IDs from the datafile.
sub get_col_names(); #Get the names of the variables.
# Misc
sub get_saliencies(); #Get the saliencies for corresponding columns.
sub get_ann_outputs(); #Get the ANN outputs for corresponding columns.
sub normalize($); #Normalize each variable.
sub normalize_sigmoid($); #Normalize each variable.
sub normalize_patient($); #Normalize over each patient instead
sub get_code($); #Convert a float into a category.
sub printit(@); #Print data, coded or not.
sub printreduced(@); #Print the 6 most important saliencies.
sub printreducedindices(@); #Print the 6 most important saliencies indices.
sub usage(); #Print usage.

# ARGUMENT PARSING
unless ( @ARGV == 2 ){
	usage();
	exit(1);
}
my $conf_name = shift @ARGV;
my $nvar = shift @ARGV;
my $data_name = get_from_config('FileNameT', 1); # Get the test filename!
my $suffix = get_from_config('Suffix', 1);

# MAIN
my @cols = get_cols();
my @col_names = get_col_names();
my @saliencies = get_saliencies();
my @annouts = get_ann_outputs();
my @targets = get_targets();
my @ids = get_ids();
#print scalar @annouts, " ", scalar @targets, " ", scalar @saliencies, "\n"; exit;

printit("saliencies.$suffix.dat", 0); # print unnormalized
#normalize_sigmoid(\@saliencies);
normalize_patient(\@saliencies);
printit("saliencies_norm.$suffix.dat", 0); # print normalized.
printit("saliencies_norm_code.$suffix.dat", 1); # print coded.
printreduced("saliencies_reduced.$suffix.dat");
printreducedindices("saliencies_reduced_indices.$suffix.dat");

# SUBROUTINES

# print indices in order of decreasing absolute saliency
sub printreducedindices(@){
	open my $tmpfh, "> $_[0]" or die "Couldn't open $_[0]: $!\n";
	my $cntr = 0;
	foreach my $i (0..@saliencies-1){
		my @arr = ();
		for my $j (0..@cols-1){
			push(@{$arr[$j]}, $cols[$j], $saliencies[$i]->[$j]);
		}
		@arr = sort {abs($b->[1]) <=> abs($a->[1])} @arr;
		splice(@arr, $nvar, @arr-$nvar-$nvar);
		foreach my $ref (@arr){
				print $tmpfh $ref->[0], "\t"; 
		}
		print $tmpfh "\n";
	}
	close $tmpfh;
}

sub printreduced(@){
	open my $tmpfh, "> $_[0]" or die "Couldn't open $_[0]: $!\n";
	my $cntr = 0;
	foreach my $i (0..@saliencies-1){
		my @arr = ();
		for my $j (0..@col_names-1){
			push(@{$arr[$j]}, $col_names[$j], $saliencies[$i]->[$j]);
		}
		@arr = sort {$a->[1] <=> $b->[1]} @arr;
		#print "Array is of size: ", scalar @arr, "\n";
		splice(@arr, $nvar, @arr-$nvar-$nvar);
		#print "Array is of size: ", scalar @arr, "\n";
		foreach my $ref (@arr){
			#unless(get_code($ref->[1]) eq '0'){
				my $tmp = length($ref->[0]) + 2;
				printf $tmpfh ("%-$tmp"."s", $ref->[0]); 
				#}
		}
		printf $tmpfh "%-12s\t", "ANN";
		printf $tmpfh "%-12s", "Target";
		printf $tmpfh "%-12s", "Id";
		print $tmpfh "\n";
		foreach my $ref (@arr){ 
			#unless(get_code($ref->[1]) eq '0'){
				#printf("%-3s %.20f\t", get_code($ref->[1]), $ref->[1]); 
				my $tmp = length($ref->[0]) + 2;
				printf $tmpfh ("%-$tmp"."s", get_code($ref->[1])); 
				#}
		}
		printf $tmpfh "%-12f\t", $annouts[$cntr];
		printf $tmpfh "%-12d", $targets[$cntr];
		printf $tmpfh "%-12s", $ids[$cntr++];
		print $tmpfh "\n";
	}
	close $tmpfh;
}

sub printit(@){
	# Print header
	open my $tmpfh, "> $_[0]" or die "Couldn't open $_[0]: $!\n";
	foreach my $col_name (@col_names){
		printf $tmpfh "%-14s", "$col_name";
	}
	printf $tmpfh "%-14s", "ANN";
	printf $tmpfh "%-14s", "Target";
	printf $tmpfh "%-14s", "Id";
	print $tmpfh "\n";

	# Fill file with saliencies
	my $cntr = 0;
	foreach my $aref ( @saliencies ){
		foreach my $sal ( @{$aref} ){
			printf $tmpfh "%-14e", "$sal" if $_[1] == 0;
			printf $tmpfh "%-14s", get_code($sal) if $_[1] == 1;
		}
		printf $tmpfh "%-14f", $annouts[$cntr];
		printf $tmpfh "%-14d", $targets[$cntr];
		printf $tmpfh "%-14s", $ids[$cntr++];
		print $tmpfh "\n";
	}
	close $tmpfh;
}

sub get_from_config($$){
	open my $fh, "<$conf_name" or die "Couldn't open file $conf_name: $!\n";
	my ($id, $index) = @_;
	my @tokens = ();
	while(my $line = <$fh>){
		next unless $line =~ /$id/;
		next if $line =~ /%$id/;
		#print "Searching for: $id\n";
		#print "Found: $line\n";
		@tokens = split(' ', $line);
		last;
	}
	close $fh;
	return $tokens[$index];
}

sub get_cols(){
	my $conf = `cat $conf_name`;
	if($conf =~ /[^%]InCol\s+((\d+|,\s*|-\s*)*).*\n/){
		$conf = $1;
	}
	my @list = split(',', $conf);
	my @cols = ();
	for my $i (@list){
		if ($i =~ /-/){
			my ($start, $stop) = split('-', $i);
			push(@cols, ($start..$stop));
		}else{
			push(@cols, $i);
		}
	}
	return @cols;
}

sub get_id_col(){
	my $conf = `cat $conf_name`;
	if($conf =~ /IdCol\s+(\d+).*\n/){
		$conf = $1;
	}
	return $conf;
}

sub get_target_col(){
	my $conf = `cat $conf_name`;
	if($conf =~ /OutCol\s+(\d+).*\n/){
		$conf = $1;
	}
	return $conf;
}

sub get_col_names(){
	$data_name =~ /(.+)\..+/;
	my $data_col_name = $1.".lbl";
	open my $fh, "<$data_col_name" or die "Couldn't open $data_col_name: $!\n";
	my @col_names = ();
	while( my $line = <$fh> ){
		next unless $line =~ /\d/;
		if($line =~ /^\s*\d+-/){
			my ($lower, $upper) = $line =~ /(\d+)-\s{1,2}(\d+)/;
			foreach my $col (@cols){
				if( $col >= $lower and $col <= $upper ){
					my ($col_name) = $line =~ /\d+-\s{1,2}\d+\s+\b(.+)/;
					$col_name =~ s/\s//g;
					push(@col_names, $col_name);
				}
			}
		}else{
			$line =~ /^(\d+)\s/;
			my $colid = $1;
			#print join(' ', @cols), "\n"; 
			foreach my $col (@cols){
				if( $col == $colid ){
					#print "$col == $colid\n";
					my ($col_name) = $line =~ /^\d+\s+\b(.+)/;
					$col_name =~ s/\s//g;
					#print $col_name, "\n";
					push(@col_names, $col_name);
				}
			}
		}
	}
	#exit;
	close $fh;
	return @col_names;
}

sub usage(){
	print <<EOL
Usage:
	./mapsaliencies configfile nvars

	configfile: The config file used to train the ANN.
	nvars: The number of variables to use for the reduced set.
EOL
}

sub get_saliencies(){

	# Make a tmp file with the relevant columns
	open my $tmpinfh, "<$data_name" or die "Couldn't open $data_name: $!\n";
	open my $tmpoutfh, ">tmp.dat" or die "Couldn't open tmp.dat: $!\n";
	while (my $line = <$tmpinfh>){
		my @vals = split(' ', $line);
		for ( my $i=0; $i<@cols; ++$i ){
			print $tmpoutfh $vals[$cols[$i]-1];
			print $tmpoutfh "\t";
		}
		print $tmpoutfh "\n";
	}
	close($tmpoutfh);
	close($tmpinfh);

	# Run the columns through saliency program and put the results in an array
	my $suffix = get_from_config('Suffix', 1);
	my $xmlfilename = "networks.".$suffix.".xml";
	open my $fh, "saliency $xmlfilename <tmp.dat |" or die "Couldn't open pipe to saliency program: $!\n";
	my @saliencies;
	my $cntr = 0;
	while (my $line = <$fh>){
		my @vals = split(' ', $line);
		for ( my $i=0; $i<@vals; ++$i ){
			push( @{$saliencies[$cntr]}, $vals[$i] );
		}
		++$cntr;
	}
	close $fh;
	unlink('tmp.dat');
	return @saliencies;
}

sub get_ann_outputs(){

	# Make a tmp file with the relevant columns
	open my $tmpinfh, "<$data_name" or die "Couldn't open $data_name: $!\n";
	open my $tmpoutfh, ">tmp.dat" or die "Couldn't open tmp.dat: $!\n";
	while (my $line = <$tmpinfh>){
		my @vals = split(' ', $line);
		for ( my $i=0; $i<@cols; ++$i ){
			print $tmpoutfh $vals[$cols[$i]-1];
			print $tmpoutfh "\t";
		}
		print $tmpoutfh "\n";
	}
	close($tmpoutfh);
	close($tmpinfh);

	# Run the columns through ann program and put the results in an array
	my $suffix = get_from_config('Suffix', 1);
	my $xmlfilename = "networks.".$suffix.".xml";
	open my $fh, "ann $xmlfilename <tmp.dat |" or die "Couldn't open pipe to ann program: $!\n";
	my @annouts = <$fh>;
	close $fh;
	#unlink('tmp.dat');
	return @annouts;
}

sub get_ids(){
	my $id_col = get_from_config('IdCol', 1);

	# If no id column exists use rownumber as id
	if($id_col == 0){
		my $tmp = `wc $data_name`;
		my @tokens = split ' ', $tmp;
		return (1..$tokens[0]);
	}

	# Make a tmp file with the relevant columns
	open my $fh, "<$data_name" or die "Couldn't open $data_name: $!\n";
	my @ids = ();
	while (my $line = <$fh>){
		my @vals = split(' ', $line);
		push(@ids, $vals[$id_col - 1]);
	}
	close($fh);

	return @ids;
}

sub get_targets(){
	my $target_col = get_from_config('OutCol', 1);

	# Make a tmp file with the relevant columns
	open my $fh, "<$data_name" or die "Couldn't open $data_name: $!\n";
	my @targets = ();
	while (my $line = <$fh>){
		my @vals = split(' ', $line);
		push(@targets, $vals[$target_col - 1]);
	}
	close($fh);

	return @targets;
}

sub normalize_patient($){
	my $aref = $_[0];

	foreach my $pat_ref (@{$aref}) {
		my $maxval = 0;
		map { $maxval = abs($_) if abs($_) > $maxval; } @{$pat_ref};
		#print $maxval, "\n";
		unless ($maxval == 0){
			map { $_ /= $maxval; } @{$pat_ref};
		}
	}
}

sub normalize($){
	my $aref = $_[0];

	#define max values and put to zero.
	my @maxvals;
	for ( my $i = 0; $i<@cols; ++$i) { $maxvals[$i] = 0; }

	for (my $i=0; $i < @{$aref}; ++$i){
		my $ref = @{$aref}[$i];
		for (my $j=0; $j < @{$ref}; ++$j){
			if ( abs($ref->[$j]) > $maxvals[$j] ){
				$maxvals[$j] = abs($ref->[$j]);
			}
		}
	}
	for (my $i=0; $i < @{$aref}; ++$i){
		my $ref = @{$aref}[$i];
		for (my $j=0; $j < @{$ref}; ++$j){
			$ref->[$j] /= $maxvals[$j];
		}
	}
}

sub normalize_sigmoid($){
	my $aref = $_[0];
	for (my $i=0; $i < @{$aref}; ++$i){
		my $ref = @{$aref}[$i];
		for (my $j=0; $j < @{$ref}; ++$j){
			my $val = $ref->[$j];
			$ref->[$j] = 1.0 / ( 1.0 + exp(-$val) );
		}
	}
}

sub get_code($){
	my $num = shift;
	if( $num < 0 ){
		return '0' if( abs($num) <= 0.1 );
		return '-' if( abs($num) <= 0.4 );
		return '--' if( abs($num) <= 0.7 );
		return '---';
	}else {
		return '0' if( abs($num) <= 0.1 );
		return '+' if( abs($num) <= 0.4 );
		return '++' if( abs($num) <= 0.7 );
		return '+++';
	}
}
