if (( "$#" == 0 )); then 
    ID=$(mkid)
else
    ID=$1
fi

echo "Running with id $ID"

ebatch sd_bios_base slconf            "python -m train experiment=bios_repro"
ebatch sd_bios_mlm slconf             "python -m train experiment=bios_repro l_good_base=0.0 l_mlm=1.0 mlm.distill=False"
ebatch sd_bios_mlm_distill slconf_jag "python -m train experiment=bios_repro l_good_base=0.0 l_mlm=1.0 mlm.distill=True"
