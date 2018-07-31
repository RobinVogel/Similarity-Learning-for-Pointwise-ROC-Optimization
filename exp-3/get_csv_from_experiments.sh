#!/bin/bash
# This script transforms the records into a csv file.
target_folder=""
echo "type_exp,n,B,objective,constraint,time"
cat $target_folder"/"* | tr -s ' ' | sed 's/ /,/g'