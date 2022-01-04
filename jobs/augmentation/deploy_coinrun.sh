for x in jobscript_coinrun*.sh; do 
  bsub < $x
done