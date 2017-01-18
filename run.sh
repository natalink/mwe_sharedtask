#$ -cwd
#$ -pe smp 8
#$ -hard
#$ -l mem_free=4G
#$ -q troja-all.q
# source ~/.bashrc
PYTHONIOENCODING=UTF-8 python mwe_tagger.py --threads 8 $@

