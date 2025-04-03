## Purchase 100 Membership Inference Attack Template Code

To install and run this template code, start by cloning the [MIAdefenseSELENA git repo](https://github.com/inspire-group/MIAdefenseSELENA.git). Then do the following:

1. Run `MIAdefenseSELENA\prepare_dataset.py`. Note you only have to do this once, and it takes a little while.
2. Copy `ces_partition.py` to `MIAdefenseSELENA\purchase\ces_partition.py` and run it to generate victim and shadow classifier datasets. These will be written to the file systems in `MIA_root_dir` and can be reused, though you may want to rerun this if you're interested in trying different (random) dataset partitions.
3. Copy `basic-attack-label-trainloader.py` to `MIAdefenseSELENA\purchase\Undefend`. If you run this code as-is it will generate a victim classifier, shadow classifier, and attack classifier, and evaluate the accuracy of all 3. 
