for x in jobscript_*.sh; do 
  bsub < $x
done