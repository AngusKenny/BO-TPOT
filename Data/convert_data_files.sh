# Rename all *.csv to *.data remove "Clean_" and make lowercase
for f in *.csv; do
    f_lc=${f,,}
    f_lc_noclean=${f_lc/clean_/}
    mv -- "$f" "${f_lc_noclean%.csv}.data"
done
