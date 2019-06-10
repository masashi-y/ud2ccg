#!/usr/bin/perl

# Assumes just the parser output and not the log information is sent to a file - i.e. don't take stdout since log information will be mixed in.

use strict;

use Getopt::Long;
my $modrecall = 0;
GetOptions("modrecall" => \$modrecall);

if($#ARGV != 2) {die "Usage: evaluateDepsBI.pl testfile goldfile logfile\n";}

my $test_file = $ARGV[0];
my $gold_file = $ARGV[1];
my $log_file = $ARGV[2];

open(BI,$gold_file) || die "Can't open BI file.\n";
open(TEST,$test_file) || die "Can't open test file.\n";
open(LOG,">$log_file") || die "Can't open log file.\n";

our %bi_deps = ();
our %test_deps = ();
our $hkey = "";   # only used for looping
our $hval = "";   # only used for looping

our %connected_bi_tokens = ();  # for each sentence, the gold standard tokens that appear in some relation

our %dep_types = ();  # total number of dependencies of each type
our %deps_correct = (); # number of correct deps of each type
our %deps_precision_errors = (); # number of precision errors for deps of each type
our %deps_recall_errors = (); # number of recall errors for deps of each type
our $total = 0;   # only used for looping
my $dep_type = 0;   # only used for looping

our $deps_correct = 0;
our $precision_errors = 0;
our $recall_errors = 0;

our $sentence = "";  # this is a hack, in order to print out the sentence, one of the subroutines is allowed to modify it
our $sentence_with_cats = "";

our $num = 0;
while (get_next_test_deps()!=0) {
  $num++;
  print LOG $num . "\n\n";
  get_next_bi_deps();

  if (! %test_deps) {
    print LOG "Parser failed on sentence: $sentence\n\n"; 
    if (!$modrecall) {next;}
  }

  while( ($hkey,$hval) = each(%bi_deps) ) {
    my @goldgr = split(/ /, $hkey);
    my $dep_type = $goldgr[0];
    $dep_types{$dep_type}++;
    if ($test_deps{$hkey}) {
      $test_deps{$hkey} = 2;
    }
    else {
      $bi_deps{$hkey} = 0;
    }
  }
  print LOG $sentence . "\n\n";
  print LOG $sentence_with_cats . "\n\n";
  print LOG "Matches:\n";
  while( ($hkey,$hval) = each(%test_deps) ) {
      my @testgr = split(/ /, $hkey);
      my $dep_type = $testgr[0];
      if($hval == 2) {
	  print LOG $hkey . "\n";
	  $deps_correct++;
	  $deps_correct{$dep_type}++;
    }
  }
  print LOG "\n";
  print LOG "In test (parser), not gold (BI) (precision errors):\n";
  while( ($hkey,$hval) = each(%test_deps) ) {
      if($hval == 1) {
	  my @testgr = split(/ /, $hkey);
	  my $dep_type = $testgr[0];

	  # NEW: exclude deps output by parser where one of the tokens is unconnected in BI deps
	  # (following S. Pyysalo pc.)
	  my $dep_source = $testgr[1];
	  my $dep_goal = $testgr[2];
	  if ($connected_bi_tokens{$dep_source} and $connected_bi_tokens{$dep_goal}) {
	      print LOG $hkey . "\n";
	      $precision_errors++;
	      $deps_precision_errors{$dep_type}++;
	    }
      }
  }
  print LOG "\n";
  print LOG "In gold (BI), not test (parser) (recall errors):\n";
  while( ($hkey,$hval) = each(%bi_deps) ) {
      my @goldgr = split(/ /, $hkey);
      my $dep_type = $goldgr[0];
      if($hval == 0) {
	  print LOG $hkey . "\n";
	  $recall_errors++;
	  $deps_recall_errors{$dep_type}++;
      }
  }
  print LOG "\n";
}

close(TEST);
close(BI);

print "Deps correct: $deps_correct, precision errors: $precision_errors, recall errors: $recall_errors\n";
print "Precision: ";
# printf("%.2f", ($deps_correct/($deps_correct+$precision_errors)));
my $precision = ($deps_correct/($deps_correct+$precision_errors));
print $precision;
print "\n";
print "Recall: ";
# printf("%.2f", ($deps_correct/($deps_correct+$recall_errors)));
my $recall = ($deps_correct/($deps_correct+$recall_errors));
print $recall;
print "\n";
print "F-score: ";
print ((2 * $precision * $recall)/($precision+$recall));
print "\n\n";
print "Figures by dep type:\n";
my $dep_precision = 0;
my $dep_recall = 0;
my $dep_fscore = 0;
foreach $dep_type (sort keys %dep_types) {
    print $dep_type . ": " . $dep_types{$dep_type} . " total";
    # ", " . $deps_correct{$dep_type} . " correct"
    print ", P: ";
    if ($deps_correct{$dep_type} != 0 or $deps_precision_errors{$dep_type} != 0) {
	$dep_precision = $deps_correct{$dep_type}/($deps_correct{$dep_type}+$deps_precision_errors{$dep_type});
    }
    else {
	$dep_precision = 0;
    }
    $dep_precision *= 100;
    printf("%.2f", $dep_precision);
    print ", R: ";
    if ($deps_correct{$dep_type} != 0 or $deps_recall_errors{$dep_type} != 0) {
	$dep_recall = $deps_correct{$dep_type}/($deps_correct{$dep_type}+$deps_recall_errors{$dep_type});
    }
    else {
	$dep_recall = "0";
    }
    $dep_recall *= 100;
    printf("%.2f", $dep_recall);
    if (($dep_precision + $dep_recall) != 0) {
        $dep_fscore = (2*$dep_precision*$dep_recall)/($dep_precision+$dep_recall);
    }
    else {
        $dep_fscore = 0;
      }
    print ", F: ";
    printf("%.2f", $dep_fscore);
    print "\n";
}

sub get_next_test_deps { # reads dependencies produced by the parser into a hash

  my $deptype = "";
  my $depsource = "";
  my $depgoal = "";
  my $key = "";

  my $in_messages = 0;  # used to discard the "file generated by" message at top of file

  my $input = "";

  %test_deps = ();

#  while(($input = <TEST>) =~ /^#/) { # look for end of "file generated by" message, after which dependencies begin
#    next;
#  }

  while ($input = <TEST>) {

    if ($input =~ /^#/) { $in_messages = 1; }

    if ($input =~ /<c> (.*)$/) {$sentence_with_cats = $1;}

    if ($input =~ /^\((\S+) (\S+) (\S+)\)$/) {  # Note: gets only binary relations
      $deptype = $1;
      $depsource = $2;
      $depgoal = $3;
     
      $key = "$deptype $depsource $depgoal";
      $test_deps{$key} = 1;
    }
#    return 1 if $input =~ /^$/;
     if ($input =~ /^$/) {
       if ($in_messages) { $in_messages = 0; }
       else { return 1; }
     }
  }
  return 0 if (!defined($input));
  return 1; # this probably never gets called, should delete

}


sub get_next_bi_deps {  # reads stanford dependencies from BioInfer file into hash

  %bi_deps = ();
  %connected_bi_tokens = ();

  my %tokens = ();          # hash of token numbers and words
  my $cur_tok_num = "";     # used to create %tokens
  my $cur_tok_value = "";   # used to create %tokens
  my $deptype = "";         # for reading in dependencies
  my $depdir = "";          # for reading in dependencies
  my $token1 = "";          # for reading in dependencies
  my $token2 = "";          # for reading in dependencies
  my $token1num = "";
  my $token2num = "";
  my $key;

  my $input = "";

  # Put the tokens into a hash so that we can reconstitute the dependencies
  # later in a format that matches the parser output

  while(($input = <BI>) !~ /<linkage type="stanford">/) {
    if ($input =~ /<sentence.*origText="(.*)">/) {
      $sentence = $1; # this is so we can print out the sentence
    }
    if ($input =~ /<token.*id="(t\.\d+\.\d+)"/) {
      $cur_tok_num = $1;
    }
    elsif ($input =~ /<\/token>/) {
      $tokens{$cur_tok_num} = $cur_tok_value;
      $cur_tok_num = "";
      $cur_tok_value = "";
    }
    elsif ($input =~ /<subtoken.*text="([^"]*)"/) {
      $cur_tok_value .= $1;
    }
  } 

  # Having finished tokens we are at the start of the stanford links
  while(($input = <BI>) !~ /<\/linkage>/) {
                                   # do this until end of stanford links

    if($input =~ /type="(\w+)&(\w{2})\;"/) {
      $deptype = $1;   # name of the dependency type
      $depdir = $2;    # which direction: BioInfer codes with &gt; or &lt;
    }
    elsif($input =~ /type="&(\w{2})\;(\w+)"/) {  # can appear before/after dep type
      $deptype = $2;
      $depdir = $1;
    }
    
    # BioInfer contains attributes token1 and token2
    $input =~ /token1="(t\.\d+\.(\d+))"/;
    $token1 = $1;
    $token1num = $2;
    $input =~ /token2="(t\.\d+\.(\d+))"/;
    $token2 = $1;
    $token2num = $2;

    if($depdir eq "gt") {  # gt means forward pointing dep, lt means backward
      $key = $deptype . " " . $tokens{$token1} . "_" . $token1num . " " . $tokens{$token2} . "_" . $token2num;
    }
    else {
      $key = $deptype . " " . $tokens{$token2} . "_" . $token2num . " " . $tokens{$token1} . "_" . $token1num;
    }
    print "($key)\n";
    $bi_deps{$key} = 1;

    $connected_bi_tokens{"$tokens{$token1}_$token1num"} = 1;
    $connected_bi_tokens{"$tokens{$token2}_$token2num"} = 1;

  }
  print "\n\n";

  while(($input = <BI>) !~ /<\/sentence>/) {next;} 
                                   # get down to next sentence

  return 0 if (!defined($input));

  return 1;

}


