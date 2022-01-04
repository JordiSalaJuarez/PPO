for x in jobscript_ninja*.sh; do 
  bsub < $x
done