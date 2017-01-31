#/bin/bash
datadir=sharedtask-data
#langs=SV
langs="BG CS DE EL ES FR HE HU IT LT MT PL PT RO SL SV TR"
for lang in $langs
do
	echo $lang
	qsub -cwd -pe smp 8 -b y -l mem_free=4G -q "troja-all.q@*" python mwe_tagger.py --data_train="$datadir/$lang/train_test_$lang" --data_dev="$datadir/$lang/dev_$lang" --data_test="$datadir/$lang/test_$lang " --data_test_blind="$datadir/$lang/blindtest_$lang" --threads 8 $@
done

