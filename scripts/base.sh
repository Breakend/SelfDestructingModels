if (( "$#" == 0 )); then 
    ID=$(mkid)
else
    ID=$1
fi

echo "Running with id $ID"

# ebatch sd0_bios slconf "python -m train +experiment=bios_repro batch_size_search_space=[8,16,24]"
# ebatch sd1_bios slconf "python -m train +experiment=bios_repro batch_size_search_space=[8,16,24] lr_search_range=[1e-5,1e-2]"
# ebatch sd2_bios slconf_jag "python -m train +experiment=bios_repro"
# ebatch sd3_bios slconf_jag "python -m train +experiment=bios_repro inner_loop_freeze_base=True"
ebatch sd4_bios slconf_jag "python -m train +experiment=bios_repro inner_loop_random_head=True"
ebatch sd5_bios slconf_jag "python -m train +experiment=bios_repro inner_loop_freeze_base=True inner_loop_random_head=True"
