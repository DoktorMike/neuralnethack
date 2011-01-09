#!/usr/bin/perl -w

##### Program description #######
#
# Title: runann.pl
#
# Author(s): Michael Green
#
# Description: A small script to run a number of neuralnethack(s) while
# varying specified parameters.
#
# Outline:
#
# Ingredients:
#
# Procedure:
#
# $Id: runann.pl 1526 2006-03-10 14:08:02Z michael $
##################################

use strict; #For good programming practice.
use Getopt::Long; #For command line parsing.

sub readCommandLine(@); #Parse the command line.
sub parseVariates(@);
sub parseConfig(@); #Parse the config file.
sub printConfig(@); #Print the config file.
sub usage(@); #Print usage information.
sub changeConfig(@); #Change one of the variables and it's values.
sub varyParameter(@); #Vary the variable specified by --var.
sub varyRecursive(@); #Vary the variable specified by --var.
sub runAnn(@); #Run neuralnethack on the current configuration.

use vars qw(%Opt); # The options read from the commandline
my @confItems; # The order of the variables
my %confItemValues; # The values of the variables
my @varConfItems; # The name of the config items to vary
my @varConfItemNames; # The name to print
my @varConfItemOffsets; # The offsets
my @varConfItemValues; # The values to use for the item


########## Main part of the program ##############

readCommandLine();
parseVariates();
parseConfig();

changeConfig("FileName", 0, "$Opt{'Training'}") unless $Opt{'Training'} eq "";
changeConfig("FileNameT", 0, "$Opt{'Testing'}") unless $Opt{'Testing'} eq "";

open(RESFILE, ">$Opt{'OutputFile'}") or die "Couldn't open file $Opt{'OutputFile'}: $!\n";
open(LOGFILE, ">runann".rand(time).".log") or die "Couldn't open file: $!\n";
varyParameter();
close(LOGFILE);
close(RESFILE);

###################################################

sub readCommandLine(@)
{
	my ($var, $trn, $tst, $suf) = ("", "", "", "txt");
	my $ok = GetOptions( 
		'var=s'	=> \$var,		# --var path
		'trn=s'	=> \$trn,		# --trn path
		'tst=s'	=> \$tst,		# --tst path
		'suf=s'	=> \$suf	 	# --suf=string or --suf string
	);

	usage() unless ($ok && $var); #Must supply a variate file

	$Opt{'Variate'} = $var;
	$Opt{'Training'} = $trn;
	$Opt{'Testing'} = $tst;
	$Opt{'Suffix'} = $suf;
	$Opt{'OutputFile'} = "res.".$suf;

	usage() unless @ARGV == 1;
	$Opt{'InputFile'} = $ARGV[0];
}

sub usage(@)
{
	print "runann.pl - A program running a number of neuralnethack runs.\n\n";
	print "Usage: runann.pl [options] {\"inputfile\"}\n\n";
	print "Common options:\n";
	print "--var       Path to file containing variates.\n";
	print "--trn       Path to the datafile containing the training set.\n";
	print "--tst       Path to the datafile containing the testing set.\n";
	print "--suf       The output file.\n";
	exit;
}

sub parseVariates(@)
{
	open(VARS, "$Opt{'Variate'}") or die "Couldn't open file $Opt{'Variate'}: $!\n";
	my @config = <VARS>;
	close(VARS);

	foreach my $line (@config){
		next if $line =~ /^\s*(#.*|\s*)$/;
		my @tokens = split(' ', $line);
		push(@varConfItems, shift @tokens);
		push(@varConfItemNames, shift @tokens);
		push(@varConfItemOffsets, shift @tokens);
		push(@varConfItemValues, \@tokens);
	}
}

sub parseConfig(@)
{
	my $fileName = $Opt{'InputFile'};
	open(FILE, $fileName) or die "Couldn't open file $fileName: $!\n";
	my @config = <FILE>;
	close(FILE);

	foreach my $line (@config){
		my @tokens = split(' ', $line);

		#Kill off all comments with %
		for(my $i=0; $i<@tokens; ++$i){
			splice(@tokens, $i, @tokens-$i) if $tokens[$i] =~ /%/;
		}
		next if @tokens < 2; 

		my $var = shift @tokens;
		push(@confItems, $var);
		$confItemValues{"$var"} = \@tokens;
	}
}

sub printConfig(@)
{
	my $FH = shift @_;
	foreach my $order (@confItems){
		my $vals = $confItemValues{"$order"};
		print $FH "$order\t", join(' ', @$vals), "\n";
	}
}

sub changeConfig(@)
{
	my $name = shift @_;
	my $offset = shift @_;
	my @params = @_;

	my $valuesref = $confItemValues{"$name"};
	splice(@$valuesref, $offset, @params, @params);
}

sub varyParameter(@)
{
	if(scalar @varConfItems == 0){ # We don't have anything to vary.
		printf RESFILE ("%10s%10s%10s%10s%10s%10s\n", "trnAuc", "valAuc", "tstAuc", "trnCee", "valCee", "tstCee");
		my ($trnAuc, $valAuc, $tstAuc, $trnCee, $valCee, $tstCee) = runAnn();
		printf RESFILE ("%10.6f%10.6f%10.6f%10.6f%10.6f%10.6f\n", $trnAuc, $valAuc, $tstAuc, $trnCee, $valCee, $tstCee);
	}else{
		printf RESFILE ("#%9s", "$varConfItemNames[0]"); 
		for(my $i=1; $i<@varConfItemNames; ++$i){
			my $name = $varConfItemNames[$i];
			printf RESFILE ("%10s", "$name");
		}
		printf RESFILE ("%10s%10s%10s%10s%10s%10s\n", "trnAuc", "valAuc", "tstAuc", "trnCee", "valCee", "tstCee");
		varyRecursive((scalar @varConfItems) - 1, "", "$Opt{'Suffix'}");
	}
}

sub varyRecursive(@)
{
	my $variateIndex = $_[0];
	my $prefix = $_[1];
	my $suffix = $_[2];
	if($variateIndex < 0){
		changeConfig("Suffix", 0, $suffix);
		print RESFILE $prefix;
		my ($trnAuc, $valAuc, $tstAuc, $trnCee, $valCee, $tstCee) = runAnn();
		printf RESFILE ("%10.6f%10.6f%10.6f%10.6f%10.6f%10.6f\n", $trnAuc, $valAuc, $tstAuc, $trnCee, $valCee, $tstCee);
	}else{
		my $valuesref = $varConfItemValues[$variateIndex];
		my $offset = $varConfItemOffsets[$variateIndex];
		my $item = $varConfItems[$variateIndex];
		foreach my $val (@$valuesref){
			my $prefixAdd = sprintf("%10.6s", $val); 
			my $suffixAdd = "-$item"."$varConfItemNames[$variateIndex]"."$val";
			changeConfig($item, $offset, $val);
			varyRecursive($variateIndex - 1, $prefixAdd.$prefix, $suffix.$suffixAdd);
		}
	}
}

sub runAnn(@)
{
	my $randNum = rand(time);
	my $tmpconf = "tmpconf".$randNum.".txt";
	open(OUTFILE, ">$tmpconf");
	printConfig(*OUTFILE);
	close(OUTFILE);
	open(ANNOUT, "neuralnethack $tmpconf |");
	my ($trnAuc, $valAuc, $tstAuc) = ("0", "0", "0");
	my ($trnCee, $valCee, $tstCee) = ("0", "0", "0");
	while(my $line = <ANNOUT>){
		print LOGFILE $line;
		$trnAuc = $1 if($line =~ /.*training AUC: (\d+\.\d+|\d)/);
		$valAuc = $1 if($line =~ /.*validation AUC: (\d+\.\d+|\d)/);
		$tstAuc = $1 if($line =~ /.*testing AUC: (\d+\.\d+|\d)/);
		$trnCee = $1 if($line =~ /.*training CEE: (\d+\.\d+|\d)/);
		$valCee = $1 if($line =~ /.*validation CEE: (\d+\.\d+|\d)/);
		$tstCee = $1 if($line =~ /.*testing CEE: (\d+\.\d+|\d)/);
	}
	close(ANNOUT);
	system("rm $tmpconf");
	return $trnAuc, $valAuc, $tstAuc, $trnCee, $valCee, $tstCee;
}

