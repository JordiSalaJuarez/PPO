for x in jobs/jobscript_*_impala.sh; do 
  bsub < $x
done